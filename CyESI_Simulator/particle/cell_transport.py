from dataclasses import dataclass
import numpy as np
from .forces import (drag_coefficient, saffman_lift, lubrication_factors,
    brownian_diffusivity, electrophoretic_mobility, dielectrophoretic_force, sphere_linear_step)
from ..geometry.coaxial_geometry import wall_distance
from ..electro.electrostatics import EPS0


@dataclass
class ParticleResult:
    initial_radius: np.ndarray
    residence_time: np.ndarray
    tip_residence_time: np.ndarray
    outlet_radius: np.ndarray
    outlet_methanol: np.ndarray
    methanol_dose: np.ndarray
    status: np.ndarray  # 1 = nozzle exit, 2 = wall contact, 3 = censored / reverse flow
    saved_ids: np.ndarray
    trajectory_time: np.ndarray
    trajectory_r: np.ndarray
    trajectory_z: np.ndarray
    trajectory_methanol: np.ndarray
    diagnostics: dict
    stress: dict | None = None
    bulk_history: dict | None = None

    @property
    def exited(self): return self.status == 1


def _initial_radius(config, rng):
    # Flux-weighted Poiseuille injection, conditioned on finite-sphere clearance.
    a = config.geometry.a
    rmax = min(a-config.cell.diameter_um*.5e-6*(1+1e-3),
               a-config.cell.diameter_um*.5e-6-config.supply.wall_capture_gap_um*1e-6)
    if rmax <= 0: raise ValueError("Wall exclusion gap leaves no accessible cell inlet")
    smax = (rmax/a)**2
    return a*np.sqrt(1-np.sqrt(1-rng.random(config.cell.count)*(2*smax-smax*smax)))


def _bulk(config, radius0, rng):
    """Shared developed-flow transport, including legacy axial gravity.

    Finite-size wall exclusion is nonadhesive unless a capture hazard is set.
    The exact force-free characteristic remains subject to the configured
    observation timeout, just like the integrated and Brownian paths.
    """
    from .long_tube import simulate as long_tube
    result = long_tube(config, radius0, rng)
    return result.x, result.y, result.time, result.status, result.velocity_z


def simulate(config, solvent, electric, voltage_scale=1., progress=lambda x: None, *, save_every_steps=None):
    """Track finite spheres; optional recording cadence leaves integration unchanged."""
    if save_every_steps is not None and (isinstance(save_every_steps,bool) or
            not isinstance(save_every_steps,int) or save_every_steps<1):
        raise ValueError('save_every_steps must be a positive integer or None')
    g, f, c, n = config.geometry, config.fluid, config.cell, config.numerics
    rng = np.random.default_rng(n.seed)
    N = c.count; R = c.diameter_um*.5e-6
    initial = _initial_radius(config, rng)
    from .long_tube import simulate as long_tube
    bulk = long_tube(config, initial, rng)
    x, y, bulk_time, bulk_status, vb = bulk.x, bulk.y, bulk.time, bulk.status, bulk.velocity_z
    stress = bulk.stress; bulk_history = bulk.history
    stress["extension_dose_pa_s"] = np.zeros(N)
    r = np.hypot(x, y)
    z = np.full(N, g.liquid_start)
    if bulk_history is not None:
        stopped=bulk_status!=0
        # Unresolved/captured cells remain at their actual long-tube location.
        z[stopped]=g.inner_tip-g.sample_length_mm*1e-3+bulk.distance[stopped]
    vr = np.zeros(N); vz = vb.copy()
    time = np.zeros(N); dose = np.zeros(N); exposure = np.zeros(N)
    status = bulk_status.copy()
    outlet_radius = np.full(N, np.nan); residence = np.full(N, np.nan)
    outlet_c = np.full(N, np.nan); exit_tip = np.full(N, np.nan)
    k = min(n.saved_trajectories, N)
    ids = np.linspace(0, N-1, k, dtype=int)
    # First two frames connect the exact developed segment to the resolved region.
    tr = [initial[ids].copy(), r[ids].copy()]
    tz = [np.full(k, g.inner_tip-g.sample_length_mm*1e-3), z[ids].copy()]
    tt = [np.zeros(k), bulk_time[ids].copy()]
    tc = [np.zeros(k), np.zeros(k)]
    # Fill solid cells before differentiating so the lip does not introduce arbitrary zeros.
    from ..chemistry.solvent_transport import filled
    solvent.shear = np.gradient(filled(solvent.velocity_z, solvent.active), solvent.r, axis=0)
    from ..fluid.stress import axisymmetric_stress
    composition = np.clip(solvent.concentration, 0, 1)
    mu_grid = np.exp((1-composition)*np.log(f.sample_viscosity)+composition*np.log(f.sheath_viscosity))
    full_stress = axisymmetric_stress(solvent.r, solvent.z,
        filled(solvent.velocity_r, solvent.active), filled(solvent.velocity_z, solvent.active), mu_grid)
    solvent.equivalent_stress_pa = full_stress["equivalent_stress_pa"]
    solvent.axial_extension_stress_pa = full_stress["axial_extension_stress_pa"]
    from ..fluid.supply import segments
    gravity_z = segments(config)[-1]["gz"]
    dt = n.particle_dt_s
    max_steps = int(np.ceil(n.particle_tip_timeout_s/dt))
    save_interval = max(1, max_steps//400) if save_every_steps is None else save_every_steps
    mass = 4/3*np.pi*R**3*c.density
    eps = EPS0*config.chemistry.liquid_relative_permittivity
    max_slip_re = 0.
    backend='numpy';particle_threads=1
    if n.particle_backend!='numpy' and not c.use_brownian:
        try:
            from .compiled_tip import integrate as compiled_integrate
        except ImportError:
            if n.particle_backend=='numba':raise RuntimeError('Numba is unavailable. Install cyesi-studio[performance].')
        else:
            progress(f'Tracking {N} cells with compiled finite-sphere transport')
            time,dose,exposure,outlet_radius,outlet_c,exit_tip,history,max_slip_re=compiled_integrate(
                config,solvent,electric,r,z,vz,status,stress,ids,voltage_scale,save_every_steps)
            ok=status==1;residence[ok]=bulk_time[ok]+exit_tip[ok]
            for frame in history:
                tr.append(frame[:,0].copy());tz.append(frame[:,1].copy())
                tt.append(bulk_time[ids]+frame[:,2]);tc.append(frame[:,3].copy())
            backend='numba'
            from numba import get_num_threads
            particle_threads=get_num_threads()
    for step in range(0 if backend=='numba' else max_steps):
        idx = np.flatnonzero(status == 0)
        if not len(idx): break
        rr, zz = r[idx], z[idx]
        C = np.clip(solvent.sample(rr, zz), 0, 1)
        mu = np.exp((1-C)*np.log(f.sample_viscosity)+C*np.log(f.sheath_viscosity))
        rho = f.sample_density*(1-C)+f.sheath_density*C
        ur, uz = solvent.sample(rr, zz, "velocity_r"), solvent.sample(rr, zz, "velocity_z")
        shear = solvent.sample(rr, zz, "shear")
        distance, normal_r, normal_z = wall_distance(rr, zz, g)
        normal_drag, parallel_drag = lubrication_factors(R, distance-R)
        if not config.supply.lubrication_enabled:
            normal_drag = np.ones_like(normal_drag); parallel_drag = np.ones_like(parallel_drag)
        zeta = drag_coefficient(mu, R)
        lift_ratio = saffman_lift(rho, mu, R, np.ones(len(idx)), shear)/zeta
        # Saffman law is invalid close to a wall; smoothly suppress there.
        lift_ratio *= np.clip((distance-R)/(2*R), 0, 1)
        if not config.supply.saffman_enabled: lift_ratio *= 0.
        er = electric.sample(rr, zz, "field_r")*voltage_scale
        ez = electric.sample(rr, zz, "field_z")*voltage_scale
        ep = electrophoretic_mobility(eps, c.zeta_mv*1e-3, mu)
        dep_r = dielectrophoretic_force(eps, R, c.clausius_mossotti,
                  electric.sample(rr, zz, "grad_e2_r"))*voltage_scale**2
        dep_z = dielectrophoretic_force(eps, R, c.clausius_mossotti,
                  electric.sample(rr, zz, "grad_e2_z"))*voltage_scale**2
        extra_r = ep*er+dep_r/zeta
        extra_z = ep*ez+(dep_z+(c.density-rho)*4/3*np.pi*R**3*gravity_z)/zeta
        dr,dz,vr[idx],vz[idx] = sphere_linear_step(vr[idx],vz[idx],ur,uz,extra_r,extra_z,
            normal_r,normal_z,normal_drag,parallel_drag,lift_ratio,mass/zeta,dt)
        old_z = z[idx].copy()
        if c.use_brownian:
            D = brownian_diffusivity(f.temperature_k, mu, R)
            theta_x = x[idx]/np.maximum(rr, 1e-30)
            theta_y = y[idx]/np.maximum(rr, 1e-30)
            noise = rng.normal(size=(3, len(idx)))*np.sqrt(2*D*dt)
            x[idx] += dr*theta_x+noise[0]; y[idx] += dr*theta_y+noise[1]
            r[idx] = np.hypot(x[idx], y[idx]); dz += noise[2]
        else:
            raw_r = rr+dr
            reflected = raw_r < 0
            r[idx] = np.abs(raw_r)
            vr[idx[reflected]] *= -1
        z[idx] += dz
        time[idx] += dt
        dose[idx] += C*dt
        tau = np.maximum(0., solvent.sample(rr, zz, "equivalent_stress_pa"))
        ext = np.abs(solvent.sample(rr, zz, "axial_extension_stress_pa"))
        stress["shear_dose_pa_s"][idx] += tau*dt
        stress["damage_exposure"][idx] += tau**config.supply.shear_damage_exponent*dt
        stress["extension_dose_pa_s"][idx] += ext*dt
        stress["peak_equivalent_stress_pa"][idx] = np.maximum(stress["peak_equivalent_stress_pa"][idx], tau)
        exposure[idx] = C
        newdist, _, _ = wall_distance(r[idx], z[idx], g)
        hit = newdist < R*(1-1e-6)
        reverse = z[idx] < g.liquid_start-1e-9
        done = (z[idx] >= 0) & ~hit & ~reverse
        status[idx[hit]] = 2; status[idx[reverse]] = 3
        exited_idx = idx[done]
        if len(exited_idx):
            fraction = np.clip((0-old_z[done])/np.maximum(dz[done], 1e-30), 0, 1)
            time[exited_idx] -= dt*(1-fraction)
            dose[exited_idx] -= C[done]*dt*(1-fraction)
            stress["shear_dose_pa_s"][exited_idx] -= tau[done]*dt*(1-fraction)
            stress["damage_exposure"][exited_idx] -= tau[done]**config.supply.shear_damage_exponent*dt*(1-fraction)
            stress["extension_dose_pa_s"][exited_idx] -= ext[done]*dt*(1-fraction)
            z[exited_idx] = 0.
            outlet_radius[exited_idx] = r[exited_idx]
            exit_tip[exited_idx] = time[exited_idx]
            residence[exited_idx] = bulk_time[exited_idx]+time[exited_idx]
            outlet_c[exited_idx] = solvent.sample(r[exited_idx], np.zeros(len(exited_idx)))
            status[exited_idx] = 1
        max_slip_re = max(max_slip_re, float(np.max(rho*2*R*np.hypot(ur-vr[idx], uz-vz[idx])/mu)))
        if step % save_interval == 0:
            tr.append(r[ids].copy()); tz.append(z[ids].copy());
            tt.append((bulk_time[ids]+time[ids]).copy()); tc.append(exposure[ids].copy())
        if step % 1000 == 0:
            progress(f"Cells: resolved-tip time {step*dt:.2f} s; active {len(idx)} / {N}")
    status[status == 0] = 3
    tr.append(r[ids].copy()); tz.append(z[ids].copy());
    tt.append((bulk_time[ids]+time[ids]).copy()); tc.append(exposure[ids].copy())
    ok = status == 1
    diagnostics = {"count": N, "exited": int(ok.sum()), "wall_contacts": int(np.sum(status == 2)),
        "censored_or_reverse": int(np.sum(status == 3)),
        "transport_fraction": float(ok.mean()),
        "focusing_fraction_of_exits": float(np.mean(outlet_radius[ok] <= c.focusing_radius_um*1e-6)) if ok.any() else None,
        "outlet_r95_um": float(np.quantile(outlet_radius[ok], .95)*1e6) if ok.any() else None,
        "mean_residence_s": float(np.mean(residence[ok])) if ok.any() else None,
        "mean_tip_residence_s": float(np.mean(exit_tip[ok])) if ok.any() else None,
        "mean_outlet_methanol_fraction": float(np.mean(outlet_c[ok])) if ok.any() else None,
        "bulk_method": "analytic developed Poiseuille + finite-sphere Saffman/lubrication ODE; exact far-wall characteristics" if not c.use_brownian else "developed Poiseuille + Brownian drift-diffusion",
        "bulk_length_mm": g.bulk_length*1e3, "max_slip_reynolds": max_slip_re,
        "sample_volume_over_flow_s": float(np.pi*g.a*g.a*g.sample_length_mm*1e-3/f.qs),
        "electrical_scale": voltage_scale, "post_nozzle_time_s": None,
        "post_nozzle_note": "MS arrival requires gas/charged-droplet/capture coupling; not inferred from distance"}
    diagnostics.update({"tube_orientation": config.supply.orientation,
        "brownian_enabled":c.use_brownian,
        "brownian_control":"size_cutoff" if c.brownian_enabled is None else "manual",
        "particle_backend":backend,
        "particle_threads":particle_threads,
        "sample_control": config.supply.sample_control,
        "actual_sample_flow_ul_min": f.sample_flow_ul_min,
        "wall_capture_rate_s": config.supply.wall_capture_rate_s,
        "wall_model": "nonadhesive exclusion surface unless a conditional capture hazard is supplied",
        "mean_shear_dose_pa_s": float(np.mean(stress["shear_dose_pa_s"][ok])) if ok.any() else None,
        "mean_near_wall_time_s": float(np.mean(stress["near_wall_time_s"]))})
    if config.supply.shear_damage_coefficient is not None:
        stress["conditional_mechanical_survival"] = np.exp(-config.supply.shear_damage_coefficient*stress["damage_exposure"])
        diagnostics["conditional_mechanically_retained_fraction"] = float(np.mean(ok*stress["conditional_mechanical_survival"]))
    else:
        diagnostics["conditional_mechanically_retained_fraction"] = None
    diagnostics["bulk_method"] = "3D cross-section buoyancy / slip-dependent Saffman / wall mobility along orientation segments; overdamped sphere"
    diagnostics["bulk_wall_boundary"] = "finite-size exclusion; conditional capture only when wall_capture_rate_s > 0"
    diagnostics["bulk_time_censoring"] = "all characteristics, including exact force-free paths, obey bulk_timeout_s"
    return ParticleResult(initial, residence, exit_tip, outlet_radius, outlet_c, dose, status, ids,
                          np.array(tt), np.array(tr), np.array(tz), np.array(tc), diagnostics, stress, bulk_history)
