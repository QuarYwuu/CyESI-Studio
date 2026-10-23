"""Axisymmetric div(epsilon grad V)=0, quartz included as a dielectric.

Resistive mode imports the conserved ionic current/diffusion potential.
The ideal-conductor sensitivity limit keeps connected liquid equipotential.
The MS inlet is approximated by a grounded plane; finite aperture is absent.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ..geometry.coaxial_geometry import classify, Domain
from ..numerics import metric, elliptic, dirichlet_solve

EPS0 = 8.8541878128e-12


@dataclass
class ElectricResult:
    radial_edges: np.ndarray
    axial_edges: np.ndarray
    potential: np.ndarray
    field_r: np.ndarray
    field_z: np.ndarray
    domains: np.ndarray
    diagnostics: dict
    config: object
    ionic: object = None

    def __post_init__(self):
        self.r, self.z, *_ = metric(self.radial_edges, self.axial_edges)
        self._fields = {}
        e2 = self.field_r**2+self.field_z**2
        self.grad_e2_r, self.grad_e2_z = np.gradient(e2, self.r, self.z, edge_order=2)

    def sample(self, r, z, field="field_z"):
        r,z=np.broadcast_arrays(r,z)
        liquid=(r<self.config.geometry.c)&(z<=0)
        if self.ionic is not None and np.all(liquid):
            return self.ionic.sample(r,z,field)
        if self.config.electrical.liquid_model=='conducting' and field!='potential' and np.all(liquid):
            return np.zeros_like(r,dtype=float)
        if field not in self._fields:
            self._fields[field] = RegularGridInterpolator((self.r, self.z), getattr(self, field),
                                                        bounds_error=False, fill_value=None)
        r, z = np.broadcast_arrays(r, z)
        points = np.c_[np.clip(r.ravel(), self.r[0], self.r[-1]), np.clip(z.ravel(), self.z[0], self.z[-1])]
        out = self._fields[field](points).reshape(r.shape)
        if self.config.electrical.liquid_model == "conducting" and field != "potential":
            liquid = (r < self.config.geometry.c) & (z <= 0)
            out = np.where(liquid, 0., out)
        if self.ionic is not None:
            liquid=(r<self.config.geometry.c)&(z<=0)
            out=np.where(liquid,self.ionic.sample(r,z,field),out)
        return out


def solve(config, ionic=None):
    g, e = config.geometry, config.electrical
    # Resolve the nozzle and tube radii, then stretch into the external domain.
    nr = max(24, config.numerics.radial_cells)
    re = np.unique(np.r_[np.linspace(0, g.gas_od_um*.5e-6, nr+1),
                            np.geomspace(g.gas_od_um*.5e-6, g.external_radius_mm*1e-3, 25),
                            g.a, g.b, g.c, g.d, g.gas_id_um*.5e-6])
    zstart = min(g.liquid_start, -g.gas_recess_mm*1e-3)
    ze = np.unique(np.r_[np.linspace(zstart, 0., 65),
                          np.linspace(-min(.15e-3, -zstart), .2e-3, 40),
                          np.linspace(0., g.ms_gap_mm*1e-3, 65),
                          g.inner_tip, -g.gas_recess_mm*1e-3])
    ze = ze[(ze >= zstart) & (ze <= g.ms_gap_mm*1e-3)]
    r, z, *_ = metric(re, ze)
    R, Z = np.meshgrid(r, z, indexing="ij")
    domains = classify(R, Z, g)
    eps = np.ones(R.shape)
    eps[domains == Domain.QUARTZ] = e.quartz_relative_permittivity
    liquid = (domains == Domain.SAMPLE) | (domains == Domain.SHEATH)
    eps[liquid] = config.chemistry.liquid_relative_permittivity
    volts = e.voltage_kv*1000
    # Insulating outer radial boundary.
    A, rhs = elliptic(re, ze, eps, boundary={"z1": 0.})
    fixed = liquid if e.liquid_model in ("conducting", "resistive") else np.zeros(R.shape, bool)
    if e.liquid_model == "dielectric": fixed[:, 0] = liquid[:, 0]
    values = np.full(R.size, volts)
    if e.liquid_model == "resistive":
        if ionic is None:
            from ..engine import liquid_solution
            _, solvent, _ = liquid_solution(config)
            ionic = solvent.ionic
        values = ionic.sample(R, Z).ravel()
    potential, residual = dirichlet_solve(A, rhs, fixed, values)
    V = potential.reshape(R.shape)
    Er, Ez = [-v for v in np.gradient(V, r, z, edge_order=2)]
    if e.liquid_model == "conducting": Er[liquid] = 0.; Ez[liquid] = 0.
    if e.liquid_model == "resistive":
        Er[liquid] = ionic.sample(R[liquid], Z[liquid], "field_r")
        Ez[liquid] = ionic.sample(R[liquid], Z[liquid], "field_z")
    return ElectricResult(re, ze, V, Er, Ez, domains,
        {"method": "axisymmetric dielectric-weighted Laplace FV", "relative_residual": residual,
         "liquid_model": e.liquid_model, "ms_boundary": "grounded plane; no aperture / vacuum suction",
         "electrode_assumption": "quartz dielectric; remote direct liquid contact" if ionic is not None else "ideal connected liquid equipotential",
         "ionic": None if ionic is None else ionic.diagnostics}, config, ionic)
