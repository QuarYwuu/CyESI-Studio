"""Transient Cartesian two-phase flow and charge transport.

Liquid and gas share velocity and pressure fields. Surface tension uses a
continuum surface force; Maxwell stress uses a variable-permittivity potential.
Charge advection is conservative and Ohmic relaxation is implicit. Contact-line
calibration, distributed electrode dynamics and evaporation are unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .volume import (CartesianGrid, FiniteVolumeDomain, VolumeResult, EPS0,
                     FARADAY, GAS_CONSTANT, D_PLUS, D_MINUS, materials)


@dataclass
class PhaseSettings:
    upstream_mm: float = .15
    downstream_mm: float = .6
    radius_mm: float = .4
    axial_cells: int = 12
    duration_us: float = 1.
    max_steps: int = 40
    save_every: int = 4
    electric_update_every: int = 4
    cfl: float = .3
    capillary_safety: float = .15
    surface_tension_enabled: bool = True
    electric_stress_enabled: bool = True
    inertia: bool = True
    electrical_model: str = 'leaky_dielectric'
    remote_circuit: bool = True

    def validate(self):
        if not np.isfinite([self.upstream_mm, self.downstream_mm, self.radius_mm,
                            self.duration_us, self.cfl, self.capillary_safety]).all():
            raise ValueError('Phase inputs must be finite')
        if min(self.upstream_mm, self.downstream_mm, self.radius_mm, self.duration_us) <= 0:
            raise ValueError('Phase dimensions and duration must be positive')
        if not 0 < self.cfl <= .5 or not 0 < self.capillary_safety <= .25:
            raise ValueError('Invalid conservative phase step limits')
        for key in ('axial_cells', 'max_steps', 'save_every', 'electric_update_every'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if not 8 <= self.axial_cells <= 96:
            raise ValueError('Phase axial cells must be 8..96')
        if self.electrical_model not in ('dielectric', 'leaky_dielectric'):
            raise ValueError('Phase electrical_model must be dielectric or leaky_dielectric')
        if not isinstance(self.remote_circuit, bool):raise ValueError('remote_circuit must be true or false')
        return self


@dataclass
class PhaseResult:
    grid: CartesianGrid
    material: np.ndarray
    time_s: np.ndarray
    liquid_fraction: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    potential: np.ndarray
    electric: np.ndarray
    methanol: np.ndarray
    salt_mol_m3: np.ndarray
    charge_density_c_m3: np.ndarray
    diagnostics: dict
    settings: PhaseSettings


def _filled(value, active):
    if active.all():return value.copy()
    nearest = distance_transform_edt(~active, return_distances=False, return_indices=True)
    return value[tuple(nearest)]


def _gradient(value, grid):
    return np.stack(np.gradient(value, *grid.centers, edge_order=1), axis=-1)


def capillary_force(phi, grid, active, surface_tension):
    filled = _filled(phi, active)
    grad = _gradient(filled, grid)
    norm = np.linalg.norm(grad, axis=-1)
    normal = grad/np.maximum(norm[..., None], 1e-12)
    curvature = -sum(np.gradient(normal[..., axis], grid.centers[axis], axis=axis)
                     for axis in range(3))
    force = surface_tension*curvature[..., None]*grad
    force[~active] = 0.
    return force


def maxwell_force(electric, permittivity, grid, active):
    """div(epsilon*(E E - .5 E^2 I)), in N/m^3, all three components."""
    magnitude2 = np.sum(electric**2, axis=-1)
    force = np.zeros(grid.shape+(3,))
    for component in range(3):
        for axis in range(3):
            stress = permittivity*(electric[..., component]*electric[..., axis]
                                   -(.5*magnitude2 if component == axis else 0.))
            force[..., component] += np.gradient(stress, grid.centers[axis], axis=axis)
    force[~active] = 0.
    return force


def dielectric_force(electric, permittivity, grid, active):
    """Charge-free equivalent of div(T), avoiding discrete grad(E) noise.

    With div(epsilon E)=0 and curl(E)=0, div(T)=-|E|^2 grad(epsilon)/2.
    Quartz is filled from the adjacent fluid for the fluid-force gradient;
    forces on solid quartz belong to its mechanical support.
    """
    gradient = _gradient(_filled(permittivity, active), grid)
    force = -.5*np.sum(electric**2, axis=-1)[..., None]*gradient
    force[~active] = 0.
    return force


def _momentum_step(domain, velocity, pressure, rho, mu, force, flux,
                   pbc, fixed, ubc, dt, inertia):
    active = domain.active; grid = domain.grid
    mass = rho[active]*grid.volume[active]/dt
    # Non-incremental Chorin projection: pressure is supplied only by the
    # compatible conservative face correction below. Reusing an independently
    # reconstructed cell-pressure gradient here creates checkerboard feedback
    # at a large gas/liquid density jump on a collocated mesh.
    gradp = np.zeros_like(velocity)
    predicted = np.zeros_like(velocity)
    density_flux = None
    if inertia:
        rv = rho[active]; density_flux = []
        for qq, (left, right, _, _, _) in zip(flux, domain.face_data):
            donor = np.where(qq >= 0, left, right)
            donor = np.where(donor >= 0, donor, np.maximum(left, right))
            density_flux.append(qq*rv[np.maximum(donor, 0)])
    diffusion, _, _ = domain.diffusion(mu, ubc[0])
    matrix = diffusion+diags(mass)
    if density_flux is not None:
        adv, _ = domain.advection(density_flux, tuple(np.where(np.isfinite(v), v, 0.) for v in ubc[0]))
        matrix = matrix+adv
    factor = splu(matrix.tocsc())
    for component in range(3):
        _, rhs, _ = domain.diffusion(mu, ubc[component])
        rhs += mass*velocity[..., component][active]+(force[..., component][active]-gradp[..., component][active])*grid.volume[active]
        if density_flux is not None:
            _, ab = domain.advection(density_flux, tuple(np.where(np.isfinite(v), v, 0.) for v in ubc[component]))
            rhs += ab
        predicted[..., component] = domain.unpack(factor.solve(rhs))
    qstar = domain.predicted_flux(predicted, pbc, fixed)
    return domain.project(qstar, dt/rho, pbc)


def scalar_face_flux(domain, value, flux, boundary):
    transport = domain.grid.faces()
    cells = value[domain.active]
    for axis, (left, right, _, _, _) in enumerate(domain.face_data):
        both = (left >= 0)&(right >= 0)
        q = flux[axis][both]
        donor = np.where(q >= 0, left[both], right[both])
        transport[axis][both] = q*cells[donor]
        for only, ids, sign in (((left >= 0)&(right < 0), left, 1.),
                                 ((right >= 0)&(left < 0), right, -1.)):
            qq = flux[axis][only]; incoming = sign*qq < 0
            bc = boundary[axis][only]
            if np.any(incoming & ~np.isfinite(bc)):
                raise ValueError('Missing phase value at an incoming boundary')
            transport[axis][only] = qq*np.where(incoming, np.nan_to_num(bc), cells[ids[only]])
    return transport


def outflow_frequency(domain, flux):
    result = np.zeros(domain.grid.shape)
    for axis, q in enumerate(flux):
        a = [slice(None)]*3; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        result += np.maximum(q[tuple(b)], 0.)+np.maximum(-q[tuple(a)], 0.)
    return result/domain.grid.volume


def advect_fraction(domain, phi, flux, dt, boundary):
    """Bounded donor-cell transport; preserve source/outlet mass explicitly."""
    if dt <= 0 or not np.isfinite(dt):raise ValueError('Phase step must be positive')
    courant = float(np.max(outflow_frequency(domain, flux)[domain.active])*dt)
    if courant > .95:raise ValueError('Phase CFL exceeded; reduce the explicit transport step')
    transport = scalar_face_flux(domain, phi, flux, boundary)
    change = dt*domain.grid.net_outflow(transport)
    new = phi-change/domain.grid.volume
    new[~domain.active] = 0.
    if new[domain.active].min() < -1e-7 or new[domain.active].max() > 1+1e-7:
        raise FloatingPointError('Unbounded phase: projection or step control failed')
    clipped = np.clip(new, 0., 1.)
    clipping = float(np.sum((clipped-new)*domain.grid.volume))
    old_mass = float(np.sum(phi*domain.grid.volume))
    new_mass = float(np.sum(clipped*domain.grid.volume))
    net_out = float(np.sum(change))
    return clipped, transport, dict(courant=courant, boundary_net_out_volume_m3=net_out,
        clipping_volume_m3=clipping, balance_error_m3=new_mass-old_mass+net_out-clipping)


def _electric(volume, config, grid, material, phi, advected_charge=None, dt=None, conductivity=None, circuit=None):
    whole = FiniteVolumeDomain(grid, np.ones(grid.shape, bool))
    relative = np.where(material == 2, config.electrical.quartz_relative_permittivity,
                         1+(config.chemistry.liquid_relative_permittivity-1)*phi)
    boundary = grid.faces(np.nan)
    for axis in range(3):
        axes = list(grid.centers); axes[axis] = grid.edges[axis]
        xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        value = volume.sample(xyz, 'potential')
        for side in (0, -1):
            sl = [slice(None)]*3; sl[axis] = side
            prescribed = value[tuple(sl)]
            if not np.isfinite(prescribed).all():raise ValueError('Phase electric boundary lies outside the 3-D tip field')
            boundary[axis][tuple(sl)] = prescribed
    bottom_liquid = phi[:, :, 0] > .01
    if circuit is not None:boundary[2][:, :, 0][bottom_liquid] = 0.
    epsilon_matrix, epsilon_bc, epsilon_conductance = whole.diffusion(relative, boundary)
    matrix = epsilon_matrix; rhs = epsilon_bc.copy()
    conductance = None; conductive_faces = None
    if advected_charge is not None:
        rhs += advected_charge.ravel()*grid.volume.ravel()/EPS0
        if dt is not None:
            conductance, conductive_bc, conductive_faces = whole.diffusion(np.maximum(conductivity, 1e-30)/EPS0, boundary)
            matrix = matrix+dt*conductance
            rhs += dt*conductive_bc
    factor = splu(matrix.tocsc())
    values = factor.solve(rhs)
    boundary_voltage = None
    if circuit is not None:
        if dt is None or conductive_faces is None:raise ValueError('A circuit requires implicit charge relaxation')
        ids = whole.ids[:, :, 0][bottom_liquid]
        ge = epsilon_conductance[2][:, :, 0][bottom_liquid]
        gs = conductive_faces[2][:, :, 0][bottom_liquid]
        gtotal = ge+dt*gs
        response_rhs = np.zeros(whole.n); np.add.at(response_rhs, ids, gtotal)
        response = factor.solve(response_rhs)
        current_weights = EPS0*gtotal/dt
        resistance = float(circuit['resistance_ohm'])
        previous_d = float(circuit['previous_displacement_flux_c'])
        denominator = 1+resistance*(np.sum(current_weights)-np.dot(current_weights, response[ids]))
        boundary_voltage = float((circuit['source_voltage_v']+resistance*(previous_d/dt+
            np.dot(current_weights, values[ids])))/denominator)
        values += response*boundary_voltage
        boundary[2][:, :, 0][bottom_liquid] = boundary_voltage
        np.add.at(epsilon_bc, ids, ge*boundary_voltage)
    potential = values.reshape(grid.shape)
    electric = -whole.gradient(potential, boundary)
    charge = (EPS0*(epsilon_matrix@potential.ravel()-epsilon_bc)/grid.volume.ravel()).reshape(grid.shape)
    displacement_flux = whole.diffusive_flux(potential, EPS0*relative, boundary)
    bottom_displacement = float(np.sum(displacement_flux[2][:, :, 0][bottom_liquid]))
    charge_diag = dict(max_absolute_charge_balance_error_c=0., net_boundary_current_a=0.,
                       bottom_liquid_current_a=0., charge_balance_scale_c=0.,
                       bottom_liquid_displacement_flux_c=bottom_displacement,
                       circuit_boundary_voltage_v=boundary_voltage,
                       circuit_total_supply_current_a=None, circuit_balance_error_a=None)
    if advected_charge is not None and dt is not None:
        current_flux = whole.diffusive_flux(potential, np.maximum(conductivity, 1e-30), boundary)
        charge_change = (charge-advected_charge)*grid.volume
        current_change = dt*grid.net_outflow(current_flux)
        error = charge_change+current_change
        charge_diag = dict(max_absolute_charge_balance_error_c=float(np.max(abs(error))),
            net_boundary_current_a=float(sum(np.sum(np.take(q, -1, axis=a))-np.sum(np.take(q, 0, axis=a))
                                            for a, q in enumerate(current_flux))),
            bottom_liquid_current_a=float(np.sum(current_flux[2][:, :, 0][bottom_liquid])),
            charge_balance_scale_c=float(np.sum(abs(charge_change))+np.sum(abs(current_change))),
            bottom_liquid_displacement_flux_c=bottom_displacement,
            circuit_boundary_voltage_v=boundary_voltage,
            circuit_total_supply_current_a=None, circuit_balance_error_a=None)
        if circuit is not None:
            total = charge_diag['bottom_liquid_current_a']+(bottom_displacement-circuit['previous_displacement_flux_c'])/dt
            supply_current = (circuit['source_voltage_v']-boundary_voltage)/circuit['resistance_ohm'] if circuit['resistance_ohm'] > 0 else total
            charge_diag['circuit_total_supply_current_a'] = float(supply_current)
            charge_diag['circuit_balance_error_a'] = float(total-supply_current)
    return potential, electric, EPS0*relative, charge, charge_diag


def solve_phase(volume: VolumeResult, config, settings=None, progress=lambda value: None):
    """Run a new short, non-axisymmetric 3-D transient without source mutation."""
    started = time.perf_counter(); config = deepcopy(config).validate()
    settings = deepcopy(settings or PhaseSettings()).validate(); g = config.geometry; f = config.fluid
    radius = min(settings.radius_mm*.001, volume.grid.edges[0][-1]*.98)
    # Retain the native fine Cartesian transverse grid and add exact external
    # boundaries; this is a true x,y mesh, never radial revolution.
    transverse = volume.grid.edges[0]
    xy = np.r_[-radius, transverse[(transverse > -radius)&(transverse < radius)], radius]
    start = -min(settings.upstream_mm*.001, abs(volume.grid.edges[2][0])*.9)
    stop = min(settings.downstream_mm*.001, g.ms_gap_mm*.001*.95)
    upstream = max(3, int(settings.axial_cells*abs(start)/(stop-start)))
    downstream = max(5, settings.axial_cells-upstream)
    z = np.r_[np.linspace(start, 0., upstream+1)[:-1], np.linspace(0., stop, downstream+1)]
    grid = CartesianGrid((xy, xy.copy(), z)); material = materials(grid, config)
    active = material != 2; domain = FiniteVolumeDomain(grid, active)
    xyz = np.stack(np.meshgrid(*grid.centers, indexing='ij'), axis=-1)
    phi = np.where(material == 1, 1., 0.)
    velocity = np.nan_to_num(volume.sample(xyz, 'velocity')); velocity[~active] = 0.
    pressure = np.zeros(grid.shape)
    concentration = np.where(active, np.nan_to_num(volume.sample(xyz, 'methanol')), 0.)
    concentration = np.where(phi > .5, concentration, 0.)
    salt = np.where(phi > .5, np.nan_to_num(volume.sample(xyz, 'salt_mol_m3')), 0.)
    phase_bc = grid.faces(0.)
    x, y = np.meshgrid(*grid.centers[:2], indexing='ij')
    phase_bc[2][:, :, 0] = (np.hypot(x, y) < g.c).astype(float)
    solvent_bc = grid.faces(0.); solvent_bc[2][:, :, 0] = np.nan_to_num(volume.sample(
        np.stack([x, y, np.full_like(x, start)], axis=-1), 'methanol'))
    salt_bc = grid.faces(0.); salt_bc[2][:, :, 0] = np.nan_to_num(volume.sample(
        np.stack([x, y, np.full_like(x, start)], axis=-1), 'salt_mol_m3'))
    pbc = grid.faces(np.nan); fixed = grid.faces(np.nan)
    ubc = [grid.faces(np.nan) for _ in range(3)]
    for axis in range(3):
        axes = list(grid.centers); axes[axis] = grid.edges[axis]
        face_xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        native = np.nan_to_num(volume.sample(face_xyz, 'velocity'))
        for side in (0, -1):
            sl = [slice(None)]*3; sl[axis] = side
            adjacent = np.take(active, 0 if side == 0 else -1, axis=axis)
            if axis == 2 and side == 0:
                fixed[axis][tuple(sl)][adjacent] = (native[..., axis]*grid.area(axis))[tuple(sl)][adjacent]
                for component in range(3):ubc[component][axis][tuple(sl)][adjacent] = native[..., component][tuple(sl)][adjacent]
            else:pbc[axis][tuple(sl)][adjacent] = 0.
    for axis, (left, right, _, _, _) in enumerate(domain.face_data):
        wall = ((left >= 0)^(right >= 0)) & ~np.isfinite(pbc[axis]) & ~np.isfinite(fixed[axis])
        for component in range(3):ubc[component][axis][wall] = 0.
    q = domain.predicted_flux(velocity, pbc, fixed)
    rho = f.gas_density+(f.sheath_density-f.gas_density)*phi
    q, correction, residual = domain.project(q, 1./rho, pbc)
    velocity = domain.cell_velocity(q); pressure += correction
    potential, electric, permittivity, charge, initial_electric_diag = _electric(volume, config, grid, material, phi)
    if settings.electrical_model == 'dielectric':charge[...] = 0.
    times = [0.]; phases = [phi.copy()]; velocities = [velocity.copy()]; pressures = [pressure.copy()]
    potentials = [potential.copy()]; electrics = [electric.copy()]; solvents = [concentration.copy()]
    salts = [salt.copy()]; charges = [charge.copy()]
    initial_volume = float(np.sum(phi*grid.volume)); boundary_total = 0.; clipping_total = 0.
    max_mass_error = 0.; max_courant = 0.; t = 0.; steps = 0; max_continuity = residual
    maxwell_peak = 0.; capillary_peak = 0.; retries = 0
    initial_charge = float(np.sum(charge*grid.volume)); boundary_charge_total = 0.
    max_charge_error = 0.; max_charge_cell_error = 0.; max_charge_relative = 0.
    current_samples = []
    circuit_currents = []; circuit_voltages = []; circuit_errors = []
    previous_displacement = initial_electric_diag['bottom_liquid_displacement_flux_c']
    source_voltage = config.electrical.voltage_kv*1000
    initial_lip_voltage = float(np.mean(volume.sample(np.stack([x, y, np.full_like(x, start)], axis=-1), 'potential')[phase_bc[2][:, :, 0] > .5]))
    specified_current = (config.electrical.spray_current_na or 0.)*1e-9*np.sign(config.electrical.voltage_kv)
    fallback_r = volume.diagnostics.get('electric', {}).get('remote_resistance_ohm', 0.)
    circuit_resistance = max((source_voltage-initial_lip_voltage)/specified_current, 0.) if specified_current != 0 else fallback_r
    tiny = min(float(np.min(w)) for w in grid.widths)
    duration = settings.duration_us*1e-6
    while t < duration-1e-16 and steps < settings.max_steps:
        rho = f.gas_density+(f.sheath_density-f.gas_density)*phi
        mu_liquid = np.exp((1-concentration)*np.log(f.sample_viscosity)+concentration*np.log(f.sheath_viscosity))
        mu = f.gas_viscosity+(mu_liquid-f.gas_viscosity)*phi
        frequency = float(np.max(outflow_frequency(domain, q)[active]))
        dt = min(duration-t, settings.cfl/max(frequency, 1e-20))
        if settings.surface_tension_enabled:
            interface = active & (phi > .02) & (phi < .98)
            interface_rho = float(np.min(rho[interface])) if interface.any() else .5*(f.sheath_density+f.gas_density)
            dt = min(dt, settings.capillary_safety*np.sqrt(interface_rho*tiny**3/max(f.surface_tension, 1e-30)))
        force = np.zeros(grid.shape+(3,))
        if settings.surface_tension_enabled:
            surface = capillary_force(phi, grid, active, f.surface_tension)
            force += surface; capillary_peak = max(capillary_peak, float(np.max(np.linalg.norm(surface, axis=-1))))
        if settings.electric_stress_enabled:
            stress = dielectric_force(electric, permittivity, grid, active)
            if settings.electrical_model == 'leaky_dielectric':stress += charge[..., None]*electric
            force += stress; maxwell_peak = max(maxwell_peak, float(np.max(np.linalg.norm(stress, axis=-1))))
        acceleration = float(np.max(np.linalg.norm(force[active], axis=-1)/rho[active]))
        dt = min(dt, np.sqrt(.1*tiny/max(acceleration, 1e-20)))
        for attempt in range(10):
            qnew, correction, residual = _momentum_step(domain, velocity, pressure, rho, mu,
                force, q, pbc, fixed, ubc, dt, settings.inertia)
            new_frequency = float(np.max(outflow_frequency(domain, qnew)[active]))
            if new_frequency*dt <= settings.cfl*1.5:break
            dt *= min(.5, settings.cfl/max(new_frequency*dt, 1e-30))
            retries += 1
        else:raise FloatingPointError('3-D phase momentum failed adaptive CFL after 10 retries')
        max_continuity = max(max_continuity, residual)
        # A strong electric transient can alter the CFL after momentum. Retry
        # smaller time steps next iteration rather than clipping phase mass.
        new_frequency = float(np.max(outflow_frequency(domain, qnew)[active]))
        new_phi, liquid_flux, accounting = advect_fraction(domain, phi, qnew, dt, phase_bc)
        # Advect liquid methanol inventory with the SAME liquid face flux. It
        # cannot be moved independently through gas or through quartz walls.
        methanol_flux = scalar_face_flux(domain, concentration, liquid_flux, solvent_bc)
        inventory = phi*concentration-dt*grid.net_outflow(methanol_flux)/grid.volume
        concentration = np.divide(inventory, new_phi, out=np.zeros_like(inventory), where=new_phi > 1e-12)
        if concentration[active].min() < -1e-6 or concentration[active].max() > 1+1e-6:
            raise FloatingPointError('Unbounded liquid methanol inventory in phase solver')
        concentration = np.clip(concentration, 0., 1.)
        salt_flux = scalar_face_flux(domain, salt, liquid_flux, salt_bc)
        salt_inventory = phi*salt-dt*grid.net_outflow(salt_flux)/grid.volume
        salt = np.divide(salt_inventory, new_phi, out=np.zeros_like(salt_inventory), where=new_phi > 1e-12)
        if np.min(salt[active]) < -1e-5:raise FloatingPointError('Negative salt inventory in native 3-D phase transport')
        salt = np.maximum(salt, 0.)
        if settings.electrical_model == 'leaky_dielectric':
            charge_flux = scalar_face_flux(domain, charge, qnew, grid.faces(0.))
            advected_charge = charge-dt*grid.net_outflow(charge_flux)/grid.volume
            advected_charge[~active] = charge[~active]
            advective_boundary = float(dt*np.sum(grid.net_outflow(charge_flux)))
            conductivity = new_phi*(config.electrical.sheath_conductivity_s_m+
                FARADAY**2*(D_PLUS+D_MINUS)*salt/(GAS_CONSTANT*f.temperature_k))
            conductivity[~active] = 0.
            circuit = dict(resistance_ohm=circuit_resistance, source_voltage_v=source_voltage,
                previous_displacement_flux_c=previous_displacement) if settings.remote_circuit else None
            potential, electric, permittivity, charge, charge_diag = _electric(volume, config, grid,
                material, new_phi, advected_charge=advected_charge, dt=dt, conductivity=conductivity, circuit=circuit)
            previous_displacement = charge_diag['bottom_liquid_displacement_flux_c']
            boundary_charge_total += advective_boundary+dt*charge_diag['net_boundary_current_a']
            charge_error = abs(float(np.sum(charge*grid.volume))-initial_charge+boundary_charge_total)
            max_charge_error = max(max_charge_error, charge_error)
            max_charge_cell_error = max(max_charge_cell_error, charge_diag['max_absolute_charge_balance_error_c'])
            scale = max(charge_diag['charge_balance_scale_c'], abs(initial_charge), 1e-24)
            max_charge_relative = max(max_charge_relative, charge_diag['max_absolute_charge_balance_error_c']/scale)
            current_samples.append(charge_diag['bottom_liquid_current_a'])
            if settings.remote_circuit:
                circuit_currents.append(charge_diag['circuit_total_supply_current_a'])
                circuit_voltages.append(charge_diag['circuit_boundary_voltage_v'])
                circuit_errors.append(abs(charge_diag['circuit_balance_error_a']))
        phi = new_phi; q = qnew; velocity = domain.cell_velocity(q); pressure = correction
        t += dt; steps += 1
        boundary_total += accounting['boundary_net_out_volume_m3']; clipping_total += accounting['clipping_volume_m3']
        mass_error = abs(float(np.sum(phi*grid.volume))-initial_volume+boundary_total-clipping_total)
        max_mass_error = max(max_mass_error, mass_error); max_courant = max(max_courant, accounting['courant'])
        if settings.electrical_model == 'dielectric' and steps % settings.electric_update_every == 0:
            potential, electric, permittivity, _, _ = _electric(volume, config, grid, material, phi)
        if steps % settings.save_every == 0 or t >= duration-1e-16 or steps == settings.max_steps:
            times.append(t); phases.append(phi.copy()); velocities.append(velocity.copy()); pressures.append(pressure.copy())
            potentials.append(potential.copy()); electrics.append(electric.copy()); solvents.append(concentration.copy())
            salts.append(salt.copy()); charges.append(charge.copy())
            progress(f'3-D {settings.electrical_model} phase: {t*1e6:.3g}/{settings.duration_us:g} us; step {steps}')
    charge_relaxation = EPS0*config.chemistry.liquid_relative_permittivity/config.electrical.sheath_conductivity_s_m
    diag = dict(method='Native 3-D incompressible two-phase projection, conservative VOF, CSF and Maxwell stress with implicit Ohmic charge relaxation',
        electrical_model=settings.electrical_model,
        dimensions=3, shape=list(grid.shape), completed=t >= duration-1e-16,
        actual_duration_s=t, requested_duration_s=duration, steps=steps, max_steps=settings.max_steps,
        momentum_cfl_retries=retries,
        initial_liquid_volume_m3=initial_volume, final_liquid_volume_m3=float(np.sum(phi*grid.volume)),
        boundary_net_out_liquid_volume_m3=boundary_total, roundoff_clipping_volume_m3=clipping_total,
        max_volume_balance_error_m3=max_mass_error,
        max_relative_volume_balance_error=max_mass_error/max(initial_volume, 1e-30),
        phase_range=[float(phi.min()), float(phi.max())], max_transport_courant=max_courant,
        max_continuity_relative_residual=max_continuity,
        max_maxwell_force_n_m3=maxwell_peak, max_capillary_force_n_m3=capillary_peak,
        initial_free_charge_c=initial_charge, final_free_charge_c=float(np.sum(charge*grid.volume)),
        boundary_net_out_free_charge_c=boundary_charge_total,
        max_free_charge_global_balance_error_c=max_charge_error,
        max_free_charge_cell_balance_error_c=max_charge_cell_error,
        max_free_charge_relative_balance_error=max_charge_relative,
        sheath_charge_relaxation_s=charge_relaxation,
        actual_duration_over_sheath_charge_relaxation=t/charge_relaxation,
        computed_bottom_liquid_current_a_range=[float(min(current_samples)), float(max(current_samples))] if current_samples else None,
        remote_circuit_enabled=settings.remote_circuit,
        effective_upstream_resistance_ohm=float(circuit_resistance),
        circuit_total_supply_current_a_range=[float(min(circuit_currents)), float(max(circuit_currents))] if circuit_currents else None,
        circuit_cut_voltage_v_range=[float(min(circuit_voltages)), float(max(circuit_voltages))] if circuit_voltages else None,
        max_circuit_current_balance_error_a=float(max(circuit_errors)) if circuit_errors else None,
        measured_or_scenario_spray_current_a=(config.electrical.spray_current_na or 0.)*1e-9,
        phase_current_boundary_scope='A lumped upstream resistor couples supply voltage to phase displacement plus Ohmic current; measured spray current is not independently imposed a second time',
        elapsed_s=time.perf_counter()-started,
        limits=[
            'Leaky-dielectric free charge is advected and relaxed implicitly through Ohmic current; ionic species-resolved Poisson-Nernst-Planck transport and electrode reactions remain absent.',
            'A lumped upstream resistance closes total-current feedback at the liquid cut plane. Outer far-field potentials remain inherited from the fixed 3-D volume; full distributed electrode/lead capacitance and chemistry remain absent.',
            'Initial liquid is a flat nozzle outlet, not an imposed Taylor cone. A steady cone or resolved breakup is not inferred from this short transient.',
            'Diffuse donor-cell interface has no compression or geometric reconstruction; mesh and time-step convergence remain required.',
            'Quartz no-slip and neutral contact-angle extrapolation are uncalibrated; wetting hysteresis and actual lip rounding are absent.',
            'Boundary velocity and outer electric potential come from the independently solved 3-D tip field; the liquid potential uses a lumped supply circuit, while whole-device pressure feedback is not closed.',
            'Liquid methanol and salt inventories are conservatively advected; sub-grid molecular mixing, evaporation, latent heat and Marangoni stresses are absent from this transient.',
            'No droplet children or cell-origin MS signals are invented. Disconnected phase components are geometry, not calibrated spray rates.',
        ])
    if settings.electrical_model == 'dielectric':
        diag['limits'].insert(0, 'This run explicitly selects dielectric polarization only; free-charge transport is disabled and times beyond dielectric relaxation are not a conducting ESI prediction.')
    return PhaseResult(grid, material, np.array(times), np.array(phases), np.array(velocities),
                       np.array(pressures), np.array(potentials), np.array(electrics), np.array(solvents),
                       np.array(salts), np.array(charges), diag, settings)
