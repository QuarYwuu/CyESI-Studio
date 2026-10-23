"""Segmented long-tube finite-sphere transport in a 3D cross-section.

The tube fluid is developed Poiseuille. Particle response is overdamped (the
Stokes relaxation time is microseconds), including buoyancy, anisotropic wall
mobility and slip-dependent Saffman lift. Smooth bends rotate the local frame;
Dean flow, bend impacts and a resolved deformable cell are not represented.
"""
from dataclasses import dataclass
import numpy as np
from .forces import lubrication_factors, saffman_lift, brownian_diffusivity
from ..fluid.supply import segments


@dataclass
class BulkResult:
    x: np.ndarray
    y: np.ndarray
    time: np.ndarray
    status: np.ndarray
    velocity_z: np.ndarray
    stress: dict
    history: dict
    distance: np.ndarray


def simulate(config, radius0, rng):
    if (config.numerics.particle_backend!='numpy' and not config.cell.use_brownian
            and config.supply.wall_capture_rate_s==0):
        try:from .compiled_bulk import simulate as compiled
        except ImportError:
            if config.numerics.particle_backend=='numba':raise RuntimeError('Numba unavailable; install cyesi-simulator[performance]')
        else:return compiled(config,radius0,rng)
    return _reference(config,radius0,rng)


def _reference(config, radius0, rng):
    g, f, c, n, settings = config.geometry, config.fluid, config.cell, config.numerics, config.supply
    count = len(radius0); R = c.diameter_um*.5e-6
    angle = rng.uniform(0, 2*np.pi, count)
    x, y = radius0*np.cos(angle), radius0*np.sin(angle)
    time = np.zeros(count); distance = np.zeros(count); state = np.zeros(count, np.int8)
    shear_dose = np.zeros(count); square_dose = np.zeros(count); peak = np.zeros(count)
    wall_time = np.zeros(count); vz = np.zeros(count)
    ids = np.linspace(0, count-1, min(n.saved_trajectories, count), dtype=int)
    history = {"ids": ids, "time_s": [time[ids].copy()], "distance_m": [distance[ids].copy()],
               "x_m": [x[ids].copy()], "y_m": [y[ids].copy()]}
    route = segments(config)
    boundaries = np.array([s["end_m"] for s in route])
    zeta = 6*np.pi*f.sample_viscosity*R
    buoyancy = (c.density-f.sample_density)*4/3*np.pi*R**3/zeta
    D = brownian_diffusivity(f.temperature_k, f.sample_viscosity, R)
    brownian = c.use_brownian
    h_min = settings.wall_capture_gap_um*1e-6
    if h_min >= g.a-R: raise ValueError("Capture/roughness gap exceeds available tube clearance")
    umean = f.qs/(np.pi*g.a**2)
    # A fully resolved short capillary has no bulk segment, but still hands
    # the developed inlet velocity to the finite-sphere tip integrator.
    _, initial_parallel = lubrication_factors(R, np.maximum(g.a-radius0-R, h_min))
    if not settings.lubrication_enabled: initial_parallel = np.ones(count)
    vz[:] = 2*umean*(1-(radius0/g.a)**2)+buoyancy*route[0]["gz"]/initial_parallel
    # Step is controlled by transverse displacement and axial segment crossing.
    # The wall approach is lubrication-limited, not an absorbing perfect wall.
    dt = min(n.bulk_dt_s, .1*R/max(abs(buoyancy)*9.80665, 1e-30))
    if brownian: dt = min(dt, R**2/(100*D))
    save_every = max(1, int(min(n.bulk_timeout_s, max(1., g.bulk_length/max(umean, 1e-20))*4)/dt/180))
    active = distance < g.bulk_length
    step = 0
    while np.any(active):
        idx = np.flatnonzero(active)
        r = np.hypot(x[idx], y[idx]); nx = x[idx]/np.maximum(r, 1e-30); ny = y[idx]/np.maximum(r, 1e-30)
        si = np.minimum(np.searchsorted(boundaries, distance[idx], side="right"), len(route)-1)
        gravity = np.array([[route[j][key] for j in si] for key in ("gx", "gy", "gz")])
        gap = np.maximum(g.a-r-R, h_min)
        normal, parallel = lubrication_factors(R, gap)
        if not settings.lubrication_enabled:
            normal = np.ones_like(normal); parallel = np.ones_like(parallel)
        u = 2*umean*(1-(r/g.a)**2)
        slip_z = buoyancy*gravity[2]/parallel
        shear = -4*umean*r/g.a**2
        lift = saffman_lift(f.sample_density, f.sample_viscosity, R, -slip_z, shear)/zeta
        lift *= np.clip(gap/(2*R), 0, 1)
        if not settings.saffman_enabled: lift *= 0.
        ex, ey = buoyancy*gravity[0], buoyancy*gravity[1]
        projection = ex*nx+ey*ny
        vx = ex/parallel+projection*nx*(1/normal-1/parallel)+lift*nx/normal
        vy = ey/parallel+projection*ny*(1/normal-1/parallel)+lift*ny/normal
        local_vz = u+slip_z
        vz[idx] = local_vz
        stalled = local_vz <= 0
        state[idx[stalled]] = 3; active[idx[stalled]] = False
        remaining = np.minimum(g.bulk_length, boundaries[si])-distance[idx]
        local_dt = np.minimum(dt, remaining/np.maximum(local_vz, 1e-30))
        local_dt = np.minimum(local_dt, np.maximum(0., n.bulk_timeout_s-time[idx]))
        # Exact force-free characteristic avoids needless integration of vertical
        # density-matched and zero-gravity validation cases.
        near = gap <= h_min*1.05
        exact = ((not brownian) & (np.hypot(vx, vy) < 1e-20) & (np.abs(slip_z) < 1e-20)
                 & ((settings.wall_capture_rate_s == 0) | ~near))
        local_dt[exact] = np.minimum(remaining[exact]/np.maximum(local_vz[exact], 1e-30),
                                    np.maximum(0., n.bulk_timeout_s-time[idx[exact]]))
        local_dt[stalled] = 0.
        stress = f.sample_viscosity*np.abs(shear)
        shear_dose[idx] += stress*local_dt; square_dose[idx] += stress**settings.shear_damage_exponent*local_dt
        peak[idx] = np.maximum(peak[idx], stress)
        wall_time[idx] += local_dt*near
        x[idx] += vx*local_dt; y[idx] += vy*local_dt
        distance[idx] += local_vz*local_dt
        at_boundary = np.abs(distance[idx]-boundaries[si]) < 1e-12
        distance[idx[at_boundary]] = boundaries[si[at_boundary]]
        if brownian:
            noise = rng.normal(size=(3, len(idx)))*np.sqrt(2*D*local_dt)
            x[idx] += noise[0]; y[idx] += noise[1]; distance[idx] += noise[2]
        # Reflect at the finite-size exclusion/roughness surface. Capture is a
        # separate calibrated hazard, not inevitable loss on near-wall approach.
        rr = np.hypot(x[idx], y[idx]); limit = g.a-R-h_min
        contact = rr > limit
        scale = np.minimum(1., limit/np.maximum(rr, 1e-30))
        x[idx] *= scale; y[idx] *= scale
        if settings.wall_capture_rate_s:
            captured = (near | contact) & (rng.random(len(idx)) < -np.expm1(-settings.wall_capture_rate_s*local_dt))
            state[idx[captured]] = 2; active[idx[captured]] = False
        time[idx] += local_dt
        timeout = (time[idx] >= n.bulk_timeout_s-1e-10) | (distance[idx] < -1e-9)
        state[idx[timeout & (state[idx] == 0)]] = 3
        done = distance[idx] >= g.bulk_length-1e-12
        active[idx[done | timeout]] = False
        if step % save_every == 0:
            for key, value in (("time_s", time), ("distance_m", distance), ("x_m", x), ("y_m", y)):
                history[key].append(value[ids].copy())
        step += 1
    for key, value in (("time_s", time), ("distance_m", distance), ("x_m", x), ("y_m", y)):
        history[key].append(value[ids].copy())
    history = {k: np.asarray(v) for k, v in history.items()}
    stress = {"shear_dose_pa_s": shear_dose, "damage_exposure": square_dose,
              "peak_equivalent_stress_pa": peak, "near_wall_time_s": wall_time,
              "bulk_time_s": time.copy(), "bulk_status": state.copy()}
    return BulkResult(x, y, time, state, vz, stress, history, distance)
