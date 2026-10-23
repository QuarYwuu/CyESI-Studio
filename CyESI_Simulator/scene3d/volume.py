"""Cartesian finite-volume flow, chemistry and electrical fields at the tip.

Liquid outlets and quartz surfaces have fixed geometry. Developed-flow
transport supplies the upstream tube extension. Conservative face fluxes
are used for continuity and scalar transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.interpolate import RegularGridInterpolator

EPS0 = 8.8541878128e-12
FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
D_PLUS, D_MINUS = 1.96e-9, 1.45e-9


@dataclass
class VolumeSettings:
    refinement: int = 1
    axial_cells: int = 32
    flow_steps: int = 8
    coupling_iterations: int = 2
    pseudo_step_s: float = .002
    inertia: bool = True
    gas_asymmetry: float = 0.0
    inlet_offset_x_um: float = 0.0
    gas_extent_mm: float = .5
    sample_potential_v: float | None = None

    def validate(self):
        for key in ('refinement', 'axial_cells', 'flow_steps', 'coupling_iterations'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 1 <= self.refinement <= 3 or not 16 <= self.axial_cells <= 128:
            raise ValueError('3-D refinement must be 1..3; axial_cells 16..128')
        if not np.isfinite([self.pseudo_step_s, self.gas_asymmetry,
                            self.inlet_offset_x_um, self.gas_extent_mm]).all():
            raise ValueError('3-D settings must be finite')
        if self.pseudo_step_s <= 0 or self.gas_extent_mm <= 0 or abs(self.gas_asymmetry) > .8:
            raise ValueError('Invalid 3-D step, extent or gas asymmetry')
        if self.sample_potential_v is not None and not np.isfinite(self.sample_potential_v):
            raise ValueError('Sample potential must be finite or null (floating / insulated inlet)')
        return self


@dataclass
class CartesianGrid:
    edges: tuple

    def __post_init__(self):
        self.edges = tuple(np.asarray(e, float) for e in self.edges)
        if len(self.edges) != 3 or any(len(e) < 3 or not np.isfinite(e).all() or
                                     np.any(np.diff(e) <= 0) for e in self.edges):
            raise ValueError('A 3-D grid needs three finite increasing edge arrays')
        self.widths = tuple(np.diff(e) for e in self.edges)
        self.centers = tuple((e[:-1] + e[1:]) * .5 for e in self.edges)
        self.shape = tuple(len(c) for c in self.centers)
        self.volume = (self.widths[0][:, None, None] * self.widths[1][None, :, None]
                       * self.widths[2][None, None, :])

    def faces(self, value=0.):
        return tuple(np.full(tuple(n + (a == k) for a, n in enumerate(self.shape)),
                             value, dtype=float) for k in range(3))

    def area(self, axis):
        shape = [1, 1, 1]
        result = np.ones(tuple(n + (a == axis) for a, n in enumerate(self.shape)))
        for k, widths in enumerate(self.widths):
            if k != axis:
                shape[k] = len(widths)
                result *= widths.reshape(shape)
                shape[k] = 1
        return result

    def net_outflow(self, flux):
        return sum(np.diff(q, axis=k) for k, q in enumerate(flux))


class FiniteVolumeDomain:
    """Conservative orthogonal-grid operators with impermeable masked walls.

    Face values are positive along +x/+y/+z. Dirichlet data live on faces;
    NaN means insulated. This avoids grounding quartz or disconnected drops.
    """
    def __init__(self, grid, active):
        self.grid = grid
        self.active = np.asarray(active, bool)
        if self.active.shape != grid.shape or not self.active.any():
            raise ValueError('Empty or incorrectly shaped 3-D domain')
        self.ids = np.full(grid.shape, -1, int)
        self.ids[self.active] = np.arange(int(self.active.sum()))
        self.n = int(self.active.sum())
        self.face_data = []
        for axis in range(3):
            shape = tuple(n + (a == axis) for a, n in enumerate(grid.shape))
            left = np.full(shape, -1, int)
            right = left.copy()
            s = [slice(None)] * 3
            s[axis] = slice(1, None)
            left[tuple(s)] = self.ids
            s[axis] = slice(None, -1)
            right[tuple(s)] = self.ids
            hleft = np.zeros(shape)
            hright = np.zeros(shape)
            widths = np.broadcast_to(grid.widths[axis].reshape(tuple(
                len(grid.widths[axis]) if k == axis else 1 for k in range(3))), grid.shape)
            s[axis] = slice(1, None)
            hleft[tuple(s)] = .5 * widths
            s[axis] = slice(None, -1)
            hright[tuple(s)] = .5 * widths
            self.face_data.append((left, right, hleft, hright, grid.area(axis)))

    def unpack(self, values, fill=0.):
        result = np.full(self.grid.shape, fill, float)
        result[self.active] = values
        return result

    def diffusion(self, coefficient, boundary=None):
        coefficient = np.broadcast_to(coefficient, self.grid.shape)
        if np.any(coefficient[self.active] <= 0) or not np.isfinite(coefficient[self.active]).all():
            raise ValueError('Diffusion coefficients must be positive on the active domain')
        cv = coefficient[self.active]
        boundary = self.grid.faces(np.nan) if boundary is None else boundary
        diagonal = np.zeros(self.n)
        rhs = np.zeros(self.n)
        rows, cols, values, conductances = [], [], [], []
        for axis, (left, right, hl, hr, area) in enumerate(self.face_data):
            both = (left >= 0) & (right >= 0)
            l, r = left[both], right[both]
            conductance = np.zeros_like(area)
            conductance[both] = area[both] / (hl[both]/cv[l] + hr[both]/cv[r])
            gg = conductance[both]
            np.add.at(diagonal, l, gg)
            np.add.at(diagonal, r, gg)
            rows.extend((l, r)); cols.extend((r, l)); values.extend((-gg, -gg))
            for only, ids, half in (((left >= 0) & (right < 0), left, hl),
                                     ((right >= 0) & (left < 0), right, hr)):
                selected = only & np.isfinite(boundary[axis])
                idx = ids[selected]
                gg = area[selected] * cv[idx] / half[selected]
                conductance[selected] = gg
                np.add.at(diagonal, idx, gg)
                np.add.at(rhs, idx, gg * boundary[axis][selected])
            conductances.append(conductance)
        rows.append(np.arange(self.n)); cols.append(np.arange(self.n)); values.append(diagonal)
        matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(self.n, self.n)).tocsc()
        return matrix, rhs, tuple(conductances)

    def diffusive_flux(self, values, coefficient, boundary=None):
        boundary = self.grid.faces(np.nan) if boundary is None else boundary
        _, _, conductances = self.diffusion(coefficient, boundary)
        v = np.asarray(values)[self.active] if np.shape(values) == self.grid.shape else np.asarray(values)
        flux = self.grid.faces()
        for axis, ((left, right, _, _, _), conductance) in enumerate(zip(self.face_data, conductances)):
            both = (left >= 0) & (right >= 0)
            flux[axis][both] = -conductance[both] * (v[right[both]] - v[left[both]])
            only = (left >= 0) & (right < 0) & np.isfinite(boundary[axis])
            flux[axis][only] = -conductance[only] * (boundary[axis][only] - v[left[only]])
            only = (right >= 0) & (left < 0) & np.isfinite(boundary[axis])
            flux[axis][only] = -conductance[only] * (v[right[only]] - boundary[axis][only])
        return flux

    def advection(self, flux, boundary):
        diagonal = np.zeros(self.n); rhs = np.zeros(self.n)
        rows, cols, values = [], [], []
        for axis, (left, right, _, _, _) in enumerate(self.face_data):
            both = (left >= 0) & (right >= 0)
            l, r, q = left[both], right[both], flux[axis][both]
            pos, neg = np.maximum(q, 0.), np.minimum(q, 0.)
            np.add.at(diagonal, l, pos); np.add.at(diagonal, r, -neg)
            rows.extend((l, r)); cols.extend((r, l)); values.extend((neg, -pos))
            for only, ids, sign in (((left >= 0) & (right < 0), left, 1.),
                                     ((right >= 0) & (left < 0), right, -1.)):
                idx = ids[only]; q = sign * flux[axis][only]
                inflow = q < 0
                bc = boundary[axis][only]
                if np.any(inflow & ~np.isfinite(bc)):
                    raise ValueError('A scalar value is required on every inflow face')
                np.add.at(diagonal, idx, np.maximum(q, 0.))
                np.add.at(rhs, idx[inflow], -q[inflow] * bc[inflow])
        rows.append(np.arange(self.n)); cols.append(np.arange(self.n)); values.append(diagonal)
        return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(self.n, self.n)).tocsc(), rhs

    def solve_scalar(self, diffusion, flux, boundary, advection_boundary=None, source=None):
        d, b, _ = self.diffusion(diffusion, boundary)
        if flux is not None:
            a, ab = self.advection(flux, boundary if advection_boundary is None else advection_boundary)
            d = d + a; b += ab
        if source is not None:
            b += np.asarray(source)[self.active]
        factor = splu(d)
        values = factor.solve(b)
        residual = float(np.linalg.norm(d @ values-b, np.inf) / max(np.linalg.norm(b, np.inf), 1e-30))
        return self.unpack(values), residual

    def predicted_flux(self, velocity, pressure_boundary, fixed_flux):
        flux = self.grid.faces()
        for axis, (left, right, hl, hr, area) in enumerate(self.face_data):
            vv = velocity[..., axis][self.active]
            both = (left >= 0) & (right >= 0)
            flux[axis][both] = area[both] * (hr[both]*vv[left[both]] + hl[both]*vv[right[both]]) / (hl[both]+hr[both])
            for only, ids in (((left >= 0) & (right < 0), left), ((right >= 0) & (left < 0), right)):
                selected = only & np.isfinite(pressure_boundary[axis])
                flux[axis][selected] = area[selected] * vv[ids[selected]]
            selected = np.isfinite(fixed_flux[axis])
            flux[axis][selected] = fixed_flux[axis][selected]
        return flux

    def project(self, flux, coefficient, pressure_boundary, factor=None):
        matrix, bc, _ = self.diffusion(coefficient, pressure_boundary)
        rhs = -self.grid.net_outflow(flux)[self.active] + bc
        factor = splu(matrix) if factor is None else factor
        correction = self.unpack(factor.solve(rhs))
        dq = self.diffusive_flux(correction, coefficient, pressure_boundary)
        corrected = tuple(q+change for q, change in zip(flux, dq))
        imbalance = self.grid.net_outflow(corrected)[self.active]
        scale = max(sum(float(np.sum(np.abs(q))) for q in corrected)/max(self.n, 1), 1e-30)
        return corrected, correction, float(np.max(np.abs(imbalance))/scale)

    def cell_velocity(self, flux):
        result = np.zeros(self.grid.shape+(3,))
        for axis, q in enumerate(flux):
            vv = q / self.grid.area(axis)
            a = [slice(None)]*3; b = a.copy()
            a[axis] = slice(None, -1); b[axis] = slice(1, None)
            result[..., axis] = .5*(vv[tuple(a)]+vv[tuple(b)])
        result[~self.active] = 0.
        return result

    def gradient(self, value, boundary=None):
        boundary = self.grid.faces(np.nan) if boundary is None else boundary
        v = value[self.active]; out = np.zeros(self.grid.shape+(3,))
        for axis, (left, right, hl, hr, _) in enumerate(self.face_data):
            face = np.zeros_like(hl)
            both = (left >= 0) & (right >= 0)
            face[both] = (hr[both]*v[left[both]] + hl[both]*v[right[both]])/(hl[both]+hr[both])
            for only, ids in (((left >= 0) & (right < 0), left), ((right >= 0) & (left < 0), right)):
                face[only] = np.where(np.isfinite(boundary[axis][only]), boundary[axis][only], v[ids[only]])
            h = self.grid.widths[axis].reshape(tuple(len(self.grid.widths[axis]) if a == axis else 1 for a in range(3)))
            out[..., axis] = np.diff(face, axis=axis)/h
        out[~self.active] = 0.
        return out


def make_grid(config, settings):
    g = config.geometry
    rg = g.gas_id_um*.5e-6; ro = g.gas_od_um*.5e-6
    extent = max(settings.gas_extent_mm*.001, ro*1.1)
    anchors = np.array([0., g.a/3, g.a*2/3, g.a, (g.a+g.b)*.5, g.b,
                        (2*g.b+g.c)/3, (g.b+2*g.c)/3, g.c,
                        (g.c+g.d)*.5, g.d, (g.d+rg)*.5, rg, ro, extent])
    positive = np.unique(np.r_[*[np.linspace(a, b, settings.refinement+1)[:-1]
                                 for a, b in zip(anchors[:-1], anchors[1:])], extent])
    xy = np.r_[-positive[:0:-1], positive]
    start = min(g.liquid_start, -g.gas_recess_mm*.001-g.gas_supply_length_mm*.001)
    knots = sorted(set([start, g.inner_tip, -g.gas_recess_mm*.001, 0., g.ms_gap_mm*.001]))
    counts = [max(2, int(settings.axial_cells*(b-a)/(knots[-1]-start))) for a, b in zip(knots[:-1], knots[1:])]
    # Recess mixing must retain multiple genuine axial control volumes even
    # when an extremely long upstream length is requested.
    z = np.unique(np.r_[*[np.linspace(a, b, max(n, 6)+1)[:-1] for a, b, n in zip(knots[:-1], knots[1:], counts)], knots[-1]])
    return CartesianGrid((xy, xy.copy(), z))


def materials(grid, config):
    x, y, z = np.meshgrid(*grid.centers, indexing='ij')
    g = config.geometry; r = np.hypot(x, y)
    quartz = (((r >= g.a) & (r < g.b) & (z < g.inner_tip)) |
              ((r >= g.c) & (r < g.d) & (z < 0.)) |
              ((r >= g.gas_id_um*.5e-6) & (r < g.gas_od_um*.5e-6) & (z < -g.gas_recess_mm*.001)))
    liquid = (((r < g.a) & (z < g.inner_tip)) |
              ((r >= g.b) & (r < g.c) & (z < g.inner_tip)) |
              ((r < g.c) & (z >= g.inner_tip) & (z < 0.))) & ~quartz
    return np.where(quartz, 2, np.where(liquid, 1, 0)).astype(np.int8)


def _phase_boundaries(domain, config, settings, phase):
    grid = domain.grid; g = config.geometry; f = config.fluid
    pbc = grid.faces(np.nan); fixed = grid.faces(np.nan)
    ubc = [grid.faces(np.nan) for _ in range(3)]
    x, y = np.meshgrid(grid.centers[0], grid.centers[1], indexing='ij')
    radius = np.hypot(x, y)
    if phase == 1:
        for wanted, q, profile in (
            (radius < g.a, f.qs, np.maximum(0., 1.-((x-settings.inlet_offset_x_um*1e-6)**2+y*y)/g.a**2)),
            ((radius >= g.b) & (radius < g.c), f.qh, np.maximum(0., (radius-g.b)*(g.c-radius)))):
            wanted &= domain.active[:, :, 0]
            profile = np.where(wanted, profile, 0.)
            integral = float(np.sum(profile*grid.area(2)[:, :, 0]))
            if integral <= 0:
                raise ValueError('3-D inlet is unresolved or displaced outside the sample lumen')
            uz = profile*q/integral
            fixed[2][:, :, 0][wanted] = uz[wanted]*grid.area(2)[:, :, 0][wanted]
            for axis in range(3):ubc[axis][2][:, :, 0][wanted] = uz[wanted] if axis == 2 else 0.
        lip = int(np.argmin(abs(grid.edges[2])))
        selected = domain.active[:, :, lip-1]
        pbc[2][:, :, lip][selected] = f.sample_viscosity*0.+config.supply.outlet_pressure_pa
    else:
        gas = (radius >= g.d) & (radius < g.gas_id_um*.5e-6) & domain.active[:, :, 0]
        pbc[2][:, :, 0][gas] = f.gas_pressure_kpa*1000*(1+settings.gas_asymmetry*x[gas]/max(g.gas_id_um*.5e-6, 1e-30))
        for axis in range(3):
            for side in (0, -1):
                index = [slice(None)]*3; index[axis] = side
                active = np.take(domain.active, 0 if side == 0 else -1, axis=axis)
                if axis == 2 and side == 0:
                    outside = active & (radius >= g.gas_od_um*.5e-6)
                    pbc[axis][tuple(index)][outside] = 0.
                else:pbc[axis][tuple(index)][active] = 0.
    # All material and closed outer faces have zero no-slip momentum data;
    # open pressure boundaries keep homogeneous normal-gradient momentum.
    for axis, (left, right, _, _, _) in enumerate(domain.face_data):
        wall = ((left >= 0) ^ (right >= 0)) & ~np.isfinite(pbc[axis]) & ~np.isfinite(fixed[axis])
        for component in range(3):ubc[component][axis][wall] = 0.
    return pbc, fixed, ubc


def _flow(domain, config, settings, rho, mu, boundaries, initial=None, progress=lambda value: None):
    grid = domain.grid; pbc, fixed, ubc = boundaries
    velocity = np.zeros(grid.shape+(3,)) if initial is None else initial[0].copy()
    pressure = np.zeros(grid.shape) if initial is None else initial[1].copy()
    dt = settings.pseudo_step_s
    coeff = dt / np.broadcast_to(rho, grid.shape)
    zero_pbc = tuple(np.where(np.isfinite(v), 0., np.nan) for v in pbc)
    projection_matrix, _, _ = domain.diffusion(coeff, zero_pbc)
    projection_factor = splu(projection_matrix)
    mass = np.broadcast_to(rho, grid.shape)[domain.active]*grid.volume[domain.active]/dt
    flux = domain.predicted_flux(velocity, pbc, fixed)
    residual = 0.; change = 0.
    for iteration in range(settings.flow_steps):
        if iteration%max(1,settings.flow_steps//5)==0:
            progress(f'3-D momentum projection {iteration+1}/{settings.flow_steps}')
        previous = velocity.copy()
        gradp = domain.gradient(pressure, pbc)
        conv = None
        if settings.inertia and iteration > 0:
            # Scalar density is exact per gas and a frozen mixture approximation
            # for liquid; the volume-continuity flux remains conservative.
            density_faces = []
            rv = np.broadcast_to(rho, grid.shape)[domain.active]
            for left, right, _, _, _ in domain.face_data:
                a = rv[np.maximum(left, 0)]; b = rv[np.maximum(right, 0)]
                density_faces.append(np.where((left >= 0)&(right >= 0), (a+b)*.5,
                                               np.where(left >= 0, a, b)))
            conv = tuple(q*r for q, r in zip(flux, density_faces))
        predicted = np.zeros_like(velocity)
        for component in range(3):
            diffusion, b, _ = domain.diffusion(mu, ubc[component])
            matrix = diffusion+diags(mass)
            rhs = b+mass*velocity[..., component][domain.active]-gradp[..., component][domain.active]*grid.volume[domain.active]
            if conv is not None:
                adv_boundary = tuple(np.where(np.isfinite(v), v, 0.) for v in ubc[component])
                adv, ab = domain.advection(conv, adv_boundary)
                matrix = matrix+adv; rhs += ab
            predicted[..., component] = domain.unpack(splu(matrix.tocsc()).solve(rhs))
        qstar = domain.predicted_flux(predicted, pbc, fixed)
        corr_bc = pbc if initial is None and iteration == 0 else zero_pbc
        flux, correction, residual = domain.project(qstar, coeff, corr_bc, projection_factor)
        pressure += correction
        velocity = domain.cell_velocity(flux)
        change = float(np.linalg.norm((velocity-previous)[domain.active]) / max(np.linalg.norm(velocity[domain.active]), 1e-30))
    return velocity, pressure, flux, dict(continuity_relative_residual=residual,
                                         last_velocity_relative_change=change,
                                         momentum_converged=change<=config.numerics.flow_tolerance,
                                         momentum_relative_tolerance=config.numerics.flow_tolerance,
                                         pseudo_time_advanced_s=settings.flow_steps*settings.pseudo_step_s,
                                         momentum_iterations=settings.flow_steps)


@dataclass
class VolumeResult:
    grid: CartesianGrid
    material: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    methanol: np.ndarray
    salt_mol_m3: np.ndarray
    potential: np.ndarray
    electric: np.ndarray
    face_flux: tuple
    diagnostics: dict
    settings: VolumeSettings
    _samplers: dict = field(default_factory=dict, repr=False)

    def sample(self, points, name):
        value = getattr(self, name)
        if name not in self._samplers:
            self._samplers[name] = RegularGridInterpolator(self.grid.centers, value,
                                                          bounds_error=False, fill_value=np.nan)
        return self._samplers[name](points)


def _electrical(grid, material, config, methanol, salt, liquid, current, settings):
    g = config.geometry; e = config.electrical
    liquidbc = grid.faces(np.nan)
    inlet_sheath = liquid.active[:, :, 0] & (np.hypot(*np.meshgrid(*grid.centers[:2], indexing='ij')) >= g.b)
    contact_remaining = max(e.contact_distance_mm*.001-abs(grid.edges[2][0]), 0.)
    remote_resistance = contact_remaining/(e.sheath_conductivity_s_m*np.pi*(g.c*g.c-g.b*g.b))
    if abs(current*remote_resistance) > abs(e.voltage_kv*1000)+1e-9:
        raise ValueError('Configured spray current and sheath conductivity require an upstream voltage drop larger than the supply; reduce current or provide measured conductivity')
    v_in = e.voltage_kv*1000-current*remote_resistance
    liquidbc[2][:, :, 0][inlet_sheath] = v_in
    if settings.sample_potential_v is not None:
        inlet_sample = liquid.active[:, :, 0] & ~inlet_sheath
        liquidbc[2][:, :, 0][inlet_sample] = settings.sample_potential_v
    supporting = e.sheath_conductivity_s_m
    conductivity = supporting+FARADAY**2*(D_PLUS+D_MINUS)*np.maximum(salt, 0.)/(GAS_CONSTANT*config.fluid.temperature_k)
    prescribed = grid.faces()
    lip = int(np.argmin(abs(grid.edges[2])))
    exit_mask = liquid.active[:, :, lip-1]
    area = grid.area(2)[:, :, lip]
    prescribed[2][:, :, lip][exit_mask] = current*area[exit_mask]/area[exit_mask].sum()
    matrix, rhs, _ = liquid.diffusion(conductivity, liquidbc)
    rhs -= grid.net_outflow(prescribed)[liquid.active]
    if current == 0 and settings.sample_potential_v is None:
        # An insulated, connected liquid with zero specified spray current is
        # exactly equipotential. Do not magnify roundoff into fictional EOF.
        vl = np.where(liquid.active, v_in, 0.)
    else:
        vl = liquid.unpack(splu(matrix).solve(rhs))
    lip_values = vl[:, :, lip-1][exit_mask]
    if current != 0 and settings.sample_potential_v is None and np.any(lip_values*np.sign(e.voltage_kv) < -1e-6):
        raise ValueError('The remote and resolved liquid voltage drops reverse the nozzle potential at the configured current; increase measured conductivity or reduce current')
    liquid_e = -liquid.gradient(vl, liquidbc)
    # Electrostatic dielectric volume around the conductive liquid: all three
    # spatial directions are unknowns, and quartz is a dielectric, never metal.
    permittivity = np.where(material == 2, e.quartz_relative_permittivity,
                            np.where(material == 1, config.chemistry.liquid_relative_permittivity, 1.))
    whole = FiniteVolumeDomain(grid, np.ones(grid.shape, bool))
    bc = grid.faces(np.nan); bc[2][:, :, -1] = 0.
    matrix, rhs, _ = whole.diffusion(permittivity, bc)
    fixed = liquid.active.ravel(); free = ~fixed
    fixed_values = vl.ravel()[fixed]
    rhs_free = rhs[free]-matrix[free][:, fixed]@fixed_values
    values = np.zeros(np.prod(grid.shape)); values[fixed] = fixed_values
    values[free] = splu(matrix[free][:, free].tocsc()).solve(rhs_free)
    potential = values.reshape(grid.shape)
    electric = -whole.gradient(potential, bc)
    electric[liquid.active] = liquid_e[liquid.active]
    current_flux = liquid.diffusive_flux(vl, conductivity, liquidbc)
    current_flux = tuple(q+qb for q, qb in zip(current_flux, prescribed))
    balance = grid.net_outflow(current_flux)[liquid.active]
    absolute = float(np.max(abs(balance)))
    err = float(absolute/max(abs(current), 1e-20))
    return potential, electric, dict(liquid_current_balance_residual=err,
                                     applied_voltage_v=e.voltage_kv*1000,
                                     local_sheath_inlet_voltage_v=v_in,
                                     assumed_spray_current_a=current,
                                     resolved_lip_potential_range_v=[float(np.min(lip_values)), float(np.max(lip_values))],
                                     liquid_current_balance_absolute_a=absolute,
                                     remote_resistance_ohm=remote_resistance,
                                     electrical_load_feasible=abs(current*remote_resistance) <= abs(e.voltage_kv*1000))


def solve_volume(config, settings=None, progress=lambda value: None):
    """Solve Cartesian gas, liquid, species and electrical fields on fixed geometry.

    An optional x-dependent gas-pressure perturbation provides an asymmetric case.
    """
    started = time.perf_counter(); config = deepcopy(config).validate()
    settings = deepcopy(settings or VolumeSettings()).validate()
    grid = make_grid(config, settings); material = materials(grid, config)
    liquid = FiniteVolumeDomain(grid, material == 1)
    gas = FiniteVolumeDomain(grid, material == 0)
    lbc = _phase_boundaries(liquid, config, settings, 1)
    gbc = _phase_boundaries(gas, config, settings, 0)
    f = config.fluid; ch = config.chemistry
    methanol = np.where(liquid.active, ch.sheath_methanol_fraction, 0.)
    x, y = np.meshgrid(*grid.centers[:2], indexing='ij'); radius = np.hypot(x, y)
    methanol[(radius < config.geometry.a)[:, :, None] & (np.meshgrid(*grid.centers, indexing='ij')[2] < config.geometry.inner_tip)] = 0.
    salt = np.where(liquid.active, ch.ammonium_formate_mm*(1-methanol), 0.)
    cbc = grid.faces(np.nan)
    sample = liquid.active[:, :, 0] & (radius < config.geometry.a)
    sheath = liquid.active[:, :, 0] & ~sample
    cbc[2][:, :, 0][sample] = 0.; cbc[2][:, :, 0][sheath] = ch.sheath_methanol_fraction
    sbc = grid.faces(np.nan); sbc[2][:, :, 0][sample] = ch.ammonium_formate_mm; sbc[2][:, :, 0][sheath] = 0.
    # Outlet is diffusive no-flux, but explicit zero concentration is used if
    # a pressure-outlet face reverses (an uncalibrated backflow boundary).
    cab = tuple(np.where(np.isfinite(v), v, 0.) for v in cbc)
    sab = tuple(np.where(np.isfinite(v), v, 0.) for v in sbc)
    liquid_state = None; electric = np.zeros(grid.shape+(3,))
    current = (config.electrical.spray_current_na or 0.)*1e-9*np.sign(config.electrical.voltage_kv)
    remote_length = max(config.electrical.contact_distance_mm*.001-abs(grid.edges[2][0]), 0.)
    remote_r = remote_length/(config.electrical.sheath_conductivity_s_m*np.pi*(config.geometry.c**2-config.geometry.b**2))
    if abs(current*remote_r) > abs(config.electrical.voltage_kv*1000)+1e-9:
        raise ValueError('Configured spray current and sheath conductivity require an upstream voltage drop larger than the supply; reduce current or provide measured conductivity')
    progress('3-D gas momentum and pressure projection')
    ug, pg, qg, gas_diag = _flow(gas, config, settings, f.gas_density, f.gas_viscosity, gbc,progress=progress)
    coupling_history = []
    for iteration in range(settings.coupling_iterations):
        progress(f'3-D liquid / solvent / current coupling {iteration+1}/{settings.coupling_iterations}')
        mu = np.exp((1-methanol)*np.log(f.sample_viscosity)+methanol*np.log(f.sheath_viscosity))
        rho = f.sample_density*(1-methanol)+f.sheath_density*methanol
        if iteration and config.electrical.electroosmosis:
            # Helmholtz-Smoluchowski tangential slip on fixed quartz, using the
            # independently solved 3-D liquid electric field.
            eps = EPS0*ch.liquid_relative_permittivity
            eof = -eps*config.electrical.quartz_zeta_mv*.001*electric/np.maximum(mu[..., None], 1e-30)
            for axis, (left, right, _, _, _) in enumerate(liquid.face_data):
                wall = ((left >= 0) ^ (right >= 0)) & ~np.isfinite(lbc[0][axis]) & ~np.isfinite(lbc[1][axis])
                ids = np.maximum(left, right)
                for component in range(3):
                    if component != axis:lbc[2][component][axis][wall] = eof[..., component][liquid.active][ids[wall]]
        ul, pl, ql, liquid_diag = _flow(liquid, config, settings, rho, mu, lbc, liquid_state,progress=progress)
        liquid_state = (ul, pl)
        previous = methanol.copy()
        methanol, methanol_residual = liquid.solve_scalar(ch.diffusivity_m2_s, ql, cbc, cab)
        ambipolar = 2*D_PLUS*D_MINUS/(D_PLUS+D_MINUS)
        salt, salt_residual = liquid.solve_scalar(ambipolar, ql, sbc, sab)
        potential, electric, electric_diag = _electrical(grid, material, config, methanol, salt, liquid, current, settings)
        coupling_history.append(float(np.max(abs(methanol-previous))))
    velocity = ul+ug; pressure = pl+pg
    total_flux = tuple(a+b for a, b in zip(ql, qg))
    diag = dict(method='Independent 3-D Cartesian cell-centered finite volume; face-flux pressure projection',
        dimensions=3, shape=list(grid.shape), active_liquid_cells=liquid.n, active_gas_cells=gas.n,
        gas=gas_diag, liquid=liquid_diag, electric=electric_diag,
        methanol_linear_residual=methanol_residual, salt_linear_residual=salt_residual,
        methanol_range=[float(np.min(methanol[liquid.active])), float(np.max(methanol[liquid.active]))],
        salt_range_mol_m3=[float(np.min(salt[liquid.active])), float(np.max(salt[liquid.active]))],
        coupling_methanol_changes=coupling_history,
        inlet_sample_flow_m3_s=f.qs, inlet_sheath_flow_m3_s=f.qh,
        resolved_z_range_m=[float(grid.edges[2][0]), float(grid.edges[2][-1])],
        nonaxisymmetric_gas_pressure_fraction=settings.gas_asymmetry,
        elapsed_s=time.perf_counter()-started,
        limits=[
            'Fixed gas/liquid interface and flat liquid outlet; no 3-D free-surface Taylor cone or droplet breakup is claimed.',
            'Gas and liquid are independent pressure-projection domains; resolved gas shear is not yet transmitted across a deforming interface.',
            'Finite pseudo-time iterations need convergence and mesh refinement before quantitative conclusions.',
            'Long upstream capillary uses a separate developed-flow extension, not this resolved tip mesh.',
            'Neutral methanol has no electrophoretic drift. Salt uses an electroneutral ambipolar closure; full Poisson-Nernst-Planck migration is not solved.',
            'Liquid current and dielectric external fields are quasi-steady. Detached-drop free-charge evolution and electrode chemistry are absent.',
            'Helmholtz-Smoluchowski wall slip uses unmeasured quartz zeta; Maxwell free-surface traction and thermal/evaporation feedback remain unresolved.',
            'Staircase quartz geometry and first-order upwind scalar advection require refinement; fields are not experimental calibration.',
            'The ambient gas outlet is a pressure boundary, not a resolved Q Exactive vacuum-pump or heated-inlet model.',
            'Gravity acts in the upstream hydraulic head and finite-sphere transport; buoyancy-driven mixture stratification is not included in local fluid momentum.',
        ])
    return VolumeResult(grid, material, velocity, pressure, methanol, salt, potential, electric, total_flux, diag, settings)
