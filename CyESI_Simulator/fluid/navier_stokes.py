"""P2/P1 axisymmetric FEM with hoop strain and radial integration weights.

The outlet has zero traction. Scaled variables improve matrix conditioning;
the optional Picard iteration includes steady inertial convection.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementTriP2, ElementVector, asm, condense
from skfem.helpers import dot, grad, sym_grad, ddot
from .mesh import liquid_mesh


def annular_profile(r, a, b, flow):
    r = np.asarray(r)
    shape = b*b - r*r - (b*b - a*a) * np.log(b / np.maximum(r, a)) / np.log(b / a)
    integral = np.pi / 2 * (b**4 - a**4 - (b*b - a*a)**2 / np.log(b/a))
    return flow * shape / integral


def annular_gas(config):
    g, f = config.geometry, config.fluid
    a, b = g.d, g.gas_id_um * 0.5e-6
    dp = f.gas_pressure_kpa * 1000
    length = g.gas_supply_length_mm * 1e-3
    q = np.pi * dp / (8 * f.gas_viscosity * length) * (
        b**4-a**4-(b*b-a*a)**2 / np.log(b/a))
    mean = q / (np.pi*(b*b-a*a))
    return {"flow_m3_s": q, "mean_velocity_m_s": mean,
            "reynolds": f.gas_density*mean*2*(b-a)/f.gas_viscosity,
            "mach_estimate": mean/349., "assumed_supply_length_mm": g.gas_supply_length_mm}


@dataclass
class FlowResult:
    mesh_data: object
    velocity_basis: object
    pressure_basis: object
    velocity_coefficients: np.ndarray
    pressure_coefficients: np.ndarray
    velocity_scale: float
    pressure_scale: float
    diagnostics: dict

    def velocity(self, r, z):
        r, z = np.broadcast_arrays(r, z)
        pts = np.vstack([r.ravel(), z.ravel()]) / self.mesh_data.scale
        return (self.velocity_basis.interpolator(self.velocity_coefficients)(pts)
                * self.velocity_scale).reshape((2,) + r.shape)

    def pressure(self, r, z):
        r, z = np.broadcast_arrays(r, z)
        pts = np.vstack([r.ravel(), z.ravel()]) / self.mesh_data.scale
        return (self.pressure_basis.interpolator(self.pressure_coefficients)(pts)
                * self.pressure_scale).reshape(r.shape)


def solve(config, concentration=None, progress=lambda x: None, ionic=None):
    g, f, n = config.geometry, config.fluid, config.numerics
    md = liquid_mesh(config)
    ub = Basis(md.mesh, ElementVector(ElementTriP2()), intorder=5)
    pb = Basis(md.mesh, ElementTriP1(), intorder=5)
    u_scale = (f.qs + f.qh)/(np.pi*g.c*g.c)
    p_scale = f.sample_viscosity*u_scale/g.c
    xq = ub.global_coordinates() * g.c
    if concentration is None:
        fraction = np.where((xq[0] < g.a) & (xq[1] < g.inner_tip), 0.,
                            config.chemistry.sheath_methanol_fraction)
    else:
        fraction = concentration.sample(xq[0], xq[1])
    # Log interpolation is an explicit constitutive approximation, not measured mixture data.
    mu = np.exp((1-fraction)*np.log(f.sample_viscosity) + fraction*np.log(f.sheath_viscosity))
    rho = f.sample_density*(1-fraction) + f.sheath_density*fraction
    mu_ratio = mu/f.sample_viscosity
    re = rho*u_scale*g.c/f.sample_viscosity

    @BilinearForm
    def viscous(u, v, w):
        radius = w.x[0]
        return 2*w.mu*(ddot(sym_grad(u), sym_grad(v))*radius + u[0]*v[0]/radius)

    @BilinearForm
    def divergence(u, q, w):
        return q*((u.grad[0, 0]+u.grad[1, 1])*w.x[0]+u[0])

    @BilinearForm
    def convection(u, v, w):
        return w.re*dot(np.einsum('ij...,j...->i...', grad(u), w.previous), v)*w.x[0]

    A = asm(viscous, ub, mu=mu_ratio)
    B = asm(divergence, ub, pb)
    boundary = ub.get_dofs().all()
    # All boundaries except outlet are prescribed, with only u_r=0 on axis.
    pts = ub.doflocs
    at_outlet = np.isclose(pts[1], 0., atol=1e-9)
    at_axis = np.isclose(pts[0], 0., atol=1e-9)
    radial_component = np.zeros(ub.N, dtype=bool)
    radial_component[ub.split_indices()[0]] = True
    facets = md.mesh.boundary_facets()
    mid = md.mesh.p[:, md.mesh.facets[:, facets]].mean(axis=1)
    inlet_facets = np.isclose(mid[1]*g.c, g.liquid_start, atol=1e-12)
    outlet_facets = np.isclose(mid[1], 0., atol=1e-9)
    axis_facets = np.isclose(mid[0], 0., atol=1e-9)
    walls = facets[~(inlet_facets | outlet_facets | axis_facets)]
    fixed = np.union1d(ub.get_dofs(facets=walls).all(), ub.get_dofs(facets=facets[inlet_facets]).all())
    fixed = np.union1d(fixed, boundary[at_axis[boundary] & radial_component[boundary]])
    prescribed = np.zeros(ub.N+pb.N)
    max_slip = 0.
    if ionic is not None and config.electrical.electroosmosis:
        # Thin EDL is represented by tangential Helmholtz-Smoluchowski slip.
        # Quartz is insulating; no normal liquid current crosses its wall.
        wall_dofs = ub.get_dofs(facets=walls).all()
        rr, zz = pts[0, wall_dofs]*g.c, pts[1, wall_dofs]*g.c
        fraction_wall = concentration.sample(rr, zz) if concentration is not None else np.zeros_like(rr)
        viscosity = np.exp((1-fraction_wall)*np.log(f.sample_viscosity)+fraction_wall*np.log(f.sheath_viscosity))
        mobility = -8.8541878128e-12*config.chemistry.liquid_relative_permittivity*config.electrical.quartz_zeta_mv*1e-3/viscosity
        on_tip = np.isclose(zz, g.inner_tip, atol=1e-12)&(rr>g.a+1e-12)&(rr<g.b-1e-12)
        tangent = np.where(on_tip, radial_component[wall_dofs], ~radial_component[wall_dofs])
        electric_t = np.where(on_tip, ionic.sample(rr, zz, "field_r"), ionic.sample(rr, zz, "field_z"))
        slip = mobility*electric_t*tangent
        prescribed[wall_dofs] = slip/u_scale
        max_slip = float(np.max(np.abs(slip)))
    inlet = np.isclose(pts[1]*g.c, g.liquid_start, atol=1e-12) & ~radial_component
    rin = pts[0]*g.c
    sample = inlet & (rin <= g.a+1e-12)
    sheath = inlet & (rin >= g.b-1e-12)
    prescribed[np.flatnonzero(sample)] = 2*f.qs/(np.pi*g.a*g.a)*(1-(rin[sample]/g.a)**2)/u_scale
    prescribed[np.flatnonzero(sheath)] = annular_profile(rin[sheath], g.b, g.c, f.qh)/u_scale
    previous = prescribed[:ub.N].copy()
    converged = not f.inertial
    error = 0.
    for iteration in range(n.picard_iterations if f.inertial else 1):
        C = asm(convection, ub, re=re, previous=ub.interpolate(previous)) if f.inertial else 0
        system = bmat([[A+C, -B.T], [-B, csr_matrix((pb.N, pb.N))]], format="csr")
        reduced, rhs, expanded, free = condense(system, np.zeros(system.shape[0]), x=prescribed.copy(), D=fixed)
        solution = spsolve(reduced, rhs)
        expanded[free] = solution
        new = expanded[:ub.N]
        error = np.linalg.norm(new-previous)/max(np.linalg.norm(new), 1e-30)
        previous = new.copy() if iteration == 0 else 0.8*new+0.2*previous
        if not f.inertial or (iteration > 0 and error < n.flow_tolerance):
            converged = True
            break
    if not converged:
        raise RuntimeError(f"Navier-Stokes Picard iteration did not converge: {error:.3g}; refine mesh or reduce flow")
    residual = system @ expanded
    free_residual = np.linalg.norm(residual[free])/max(np.linalg.norm(system @ expanded), 1e-30)
    out = FlowResult(md, ub, pb, expanded[:ub.N], expanded[ub.N:], u_scale, p_scale,
                     {"method": "axisymmetric Taylor-Hood P2/P1 FEM; steady NS" if f.inertial else
                      "axisymmetric Taylor-Hood P2/P1 FEM; Stokes", "elements": int(md.mesh.nelements),
                      "velocity_dofs": int(ub.N), "pressure_dofs": int(pb.N),
                      "picard_iterations": iteration+1, "picard_relative_change": float(error),
                      "algebraic_relative_residual": float(free_residual),
                      "electroosmotic_max_wall_slip_m_s": max_slip,
                      "flow_control": "prescribed inlet volume flow; electric slip changes pressure and local profile",
                      "sample_reynolds": float(f.sample_density*2*f.qs/(np.pi*g.a*f.sample_viscosity)),
                      "outlet_reynolds": float(f.sheath_density*2*(f.qs+f.qh)/(np.pi*g.c*f.sheath_viscosity))})
    return out
