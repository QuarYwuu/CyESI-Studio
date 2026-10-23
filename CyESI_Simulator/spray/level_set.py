"""Experimental axisymmetric two-phase Navier-Stokes / conservative level set.

MAC pressure projection, explicit variable-viscosity stress and convection,
CSF surface tension including azimuthal curvature, Maxwell tensor traction
with a moving dielectric field. The default electrical extension conserves
advected free charge, implicitly relaxes Ohmic current and closes a remote RC
supply. The original prescribed-potential approximation remains selectable.
Annular nitrogen inlet uses an assumed developed hydraulic supply length.
This is a transient research backend. Cone/jet/breakup are outcomes, never
imposed. Mesh/time/contact-line sensitivity must precede quantitative use.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse.linalg import spsolve
from ..numerics import metric, elliptic, dirichlet_solve
from ..fluid.navier_stokes import annular_gas, annular_profile
from ..electro.electrostatics import EPS0
from .droplet import identify


@dataclass
class SprayResult:
    radial_edges: np.ndarray
    axial_edges: np.ndarray
    times: np.ndarray
    phase: np.ndarray
    velocity_r: np.ndarray
    velocity_z: np.ndarray
    pressure: np.ndarray
    active: np.ndarray
    diagnostics: dict
    droplets: list
    velocity_r_history: np.ndarray | None = None
    velocity_z_history: np.ndarray | None = None
    pressure_history: np.ndarray | None = None
    charge_density_history: np.ndarray | None = None
    potential_history: np.ndarray | None = None
    field_r_history: np.ndarray | None = None
    field_z_history: np.ndarray | None = None
    supply_current_history: np.ndarray | None = None
    inlet_voltage_history: np.ndarray | None = None
    methanol_history: np.ndarray | None = None
    salt_history: np.ndarray | None = None


def electrical_arrays(result):
    names={'charge_density_c_m3_history':'charge_density_history','potential_v_history':'potential_history',
        'field_r_v_m_history':'field_r_history','field_z_v_m_history':'field_z_history',
        'supply_current_a_history':'supply_current_history','inlet_voltage_v_history':'inlet_voltage_history',
        'methanol_fraction_history':'methanol_history','salt_mol_m3_history':'salt_history'}
    return {key:getattr(result,name) for key,name in names.items() if getattr(result,name,None) is not None}


def divergence(u, w, re, ze):
    _, _, aa, ra, volume = metric(re, ze)
    return (np.diff(ra*u, axis=0)+aa[:, None]*np.diff(w, axis=1))/volume


def project(u, w, rho, active, re, ze, dt):
    """Face velocities and harmonic 1/rho share the exact pressure operator."""
    rc, zc, aa, ra, volumes = metric(re, ze)
    beta = 1/rho
    A, rhs = elliptic(re, ze, beta, active, {"r1": 0., "z1": 0.})
    rhs -= divergence(u, w, re, ze).ravel()*volumes.ravel()/dt
    rhs[~active.ravel()] = 0.
    pressure = spsolve(A, rhs).reshape(active.shape)
    br = 2*beta[:-1]*beta[1:]/(beta[:-1]+beta[1:])
    bz = 2*beta[:, :-1]*beta[:, 1:]/(beta[:, :-1]+beta[:, 1:])
    u[1:-1] -= dt*br*np.diff(pressure, axis=0)/np.diff(rc)[:, None]
    w[:, 1:-1] -= dt*bz*np.diff(pressure, axis=1)/np.diff(zc)[None, :]
    u[-1] -= dt*beta[-1]*(0-pressure[-1])/(re[-1]-rc[-1])
    w[:, -1] -= dt*beta[:, -1]*(0-pressure[:, -1])/(ze[-1]-zc[-1])
    u[1:-1] *= active[:-1] & active[1:]
    w[:, 1:-1] *= active[:, :-1] & active[:, 1:]
    u[-1] *= active[-1]; w[:, -1] *= active[:, -1]
    u[0] = 0.
    return u, w, pressure


def _phase_flux(phi, u, w, inlet_phi, aa, ra):
    ar = np.zeros_like(u); az = np.zeros_like(w)
    ar[1:-1] = np.where(u[1:-1] >= 0, phi[:-1], phi[1:])
    ar[-1] = np.where(u[-1] >= 0, phi[-1], 0.)
    az[:, 1:-1] = np.where(w[:, 1:-1] >= 0, phi[:, :-1], phi[:, 1:])
    az[:, 0] = np.where(w[:, 0] >= 0, inlet_phi, phi[:, 0])
    az[:, -1] = np.where(w[:, -1] >= 0, phi[:, -1], 0.)
    return ra*u*ar, aa[:, None]*w*az


def bounded_reinitialization(phi, nr, nz, active, re, ze, dt, gamma, epsilon,return_flux=False):
    """Conservative compression-diffusion with local mass/capacity flux limits."""
    rc, zc, aa, ra, volume = metric(re, ze)
    fr = np.zeros((len(rc)+1, len(zc))); fz = np.zeros((len(rc), len(zc)+1))
    avgr = (phi[:-1]+phi[1:])/2; avgz = (phi[:, :-1]+phi[:, 1:])/2
    fr[1:-1] = gamma*ra[1:-1]*(avgr*(1-avgr)*(nr[:-1]+nr[1:])/2-epsilon*np.diff(phi, axis=0)/np.diff(rc)[:, None])
    fz[:, 1:-1] = gamma*aa[:, None]*(avgz*(1-avgz)*(nz[:, :-1]+nz[:, 1:])/2-epsilon*np.diff(phi, axis=1)/np.diff(zc)[None, :])
    fr[1:-1] *= active[:-1] & active[1:]
    fz[:, 1:-1] *= active[:, :-1] & active[:, 1:]
    outgoing = np.maximum(fr[1:], 0)+np.maximum(-fr[:-1], 0)+np.maximum(fz[:, 1:], 0)+np.maximum(-fz[:, :-1], 0)
    incoming = np.maximum(-fr[1:], 0)+np.maximum(fr[:-1], 0)+np.maximum(-fz[:, 1:], 0)+np.maximum(fz[:, :-1], 0)
    tout = np.minimum(1, np.maximum(phi, 0)*volume/(dt*outgoing+1e-300))
    tin = np.minimum(1, np.maximum(1-phi, 0)*volume/(dt*incoming+1e-300))
    fr[1:-1] *= np.where(fr[1:-1] >= 0, np.minimum(tout[:-1], tin[1:]), np.minimum(tout[1:], tin[:-1]))
    fz[:, 1:-1] *= np.where(fz[:, 1:-1] >= 0, np.minimum(tout[:, :-1], tin[:, 1:]), np.minimum(tout[:, 1:], tin[:, :-1]))
    new=phi-dt*(np.diff(fr, axis=0)+np.diff(fz, axis=1))/volume
    return (new,fr,fz) if return_flux else new


def _upwind_derivative(value, velocity, spacing, axis):
    pad = [(0, 0), (0, 0)]; pad[axis] = (1, 1)
    p = np.pad(value, pad, mode="edge")
    back = np.diff(p, axis=axis)
    lo = back[:-1] if axis == 0 else back[:, :-1]
    hi = back[1:] if axis == 0 else back[:, 1:]
    coordinates=np.asarray(spacing)
    if coordinates.ndim==1:
        distances=np.diff(coordinates)
        back=np.r_[distances[0],distances]; forward=np.r_[distances,distances[-1]]
        if axis==0:back=back[:,None];forward=forward[:,None]
        return np.where(velocity>=0,lo/back,hi/forward)
    return np.where(velocity >= 0, lo, hi)/spacing


def _interface(phi, r, z, active):
    # Extend the liquid indicator into the solid for differentiation only.
    nearest = distance_transform_edt(~active, return_distances=False, return_indices=True)
    extended = phi[tuple(nearest)]
    gr, gz = np.gradient(extended, r, z, edge_order=2)
    norm = np.hypot(gr, gz)
    nr, nz = gr/np.maximum(norm, 1e-12), gz/np.maximum(norm, 1e-12)
    curvature = -(np.gradient(nr, r, axis=0)+np.gradient(nz, z, axis=1)+nr/r[:, None])
    return gr, gz, nr, nz, curvature


def _electric_pressure(config, electric, phi, active, re, ze, nr, nz):
    r, z, *_ = metric(re, ze)
    R, Z = np.meshgrid(r, z, indexing="ij")
    boundary = {"r1": electric.sample(np.full_like(z, re[-1]), z, "potential"),
                "z1": electric.sample(r, np.full_like(r, ze[-1]), "potential"),
                "z0": electric.sample(r, np.full_like(r, ze[0]), "potential")}
    eps = np.where(active, 1., config.electrical.quartz_relative_permittivity)
    A, rhs = elliptic(re, ze, eps, boundary=boundary)
    conductor = (phi >= .5) & active
    # Quartz never becomes an HV conductor. Local liquid potentials include the
    # remote ohmic drop; external connected meniscus uses the calculated lip V.
    ionic_diag = electric.diagnostics.get("ionic")
    lip_voltage = ionic_diag["lip_potential_v"] if ionic_diag else config.electrical.voltage_kv*1000
    values = np.where(Z <= 0, electric.sample(R, Z, "potential"), lip_voltage)
    V, _ = dirichlet_solve(A, rhs, conductor, values.ravel())
    er, ez = [-x for x in np.gradient(V.reshape(phi.shape), r, z)]
    air = (phi < .5) & active
    nearest = distance_transform_edt(~air, return_distances=False, return_indices=True)
    er, ez = er[tuple(nearest)], ez[tuple(nearest)]
    # Jump of the COMPLETE Maxwell tensor: normal traction and tangential
    # electric shear. The liquid field vanishes only in ideal-conductor mode.
    lr = np.zeros_like(er); lz = np.zeros_like(ez)
    if config.electrical.liquid_model == "resistive":
        lr = np.where(Z<=0, electric.sample(R, Z, "field_r"), 0.)
        lz = np.where(Z<=0, electric.sample(R, Z, "field_z"), 0.)
    epsl = EPS0*config.chemistry.liquid_relative_permittivity
    trr = .5*EPS0*(er*er-ez*ez)-.5*epsl*(lr*lr-lz*lz)
    tzz = -trr
    trz = EPS0*er*ez-epsl*lr*lz
    return trr, trz, tzz


def simulate(config, solvent, electric, progress=lambda x: None):
    g, f, s = config.geometry, config.fluid, config.spray
    # Include the full gas recess, so nitrogen is released 1 mm behind the lip.
    zmin = max(g.liquid_start, -max(s.upstream_length_mm, g.inner_recess_mm+.25, g.gas_recess_mm+.25)*1e-3)
    zend = (g.ms_gap_mm if s.extend_to_ms else s.length_mm)*1e-3
    # Conform to every quartz radius, rather than losing the 25 um sheath gap
    # to a uniform grid. At least three cells span each liquid/gas passage.
    radii = np.array([0.,g.a,g.b,g.c,g.d,g.gas_id_um*.5e-6,g.gas_od_um*.5e-6,s.radius_mm*1e-3])
    radii = np.unique(radii[(radii>=0)&(radii<=s.radius_mm*1e-3)])
    counts = np.maximum(3,np.rint(s.radial_cells*np.diff(radii)/(s.radius_mm*1e-3)).astype(int))
    re = np.unique(np.concatenate([np.linspace(a,b,k+1) for a,b,k in zip(radii[:-1],radii[1:],counts)]))
    # Cluster in a 0.25 mm band around each lip. All three ends lie on faces.
    lips = np.unique([g.inner_tip,-g.gas_recess_mm*1e-3,0.])
    base = np.linspace(zmin,zend,s.axial_cells+1)
    refined = [np.linspace(max(zmin,lip-.125e-3),min(zend,lip+.125e-3),9) for lip in lips]
    ze = np.unique(np.round(np.r_[base,*refined,lips],14))
    # Merge nearly coincident faces to avoid arbitrarily small explicit steps.
    selected=[ze[0]]
    for edge in ze[1:]:
        if edge-selected[-1]>=5e-6 or np.any(np.isclose(edge,lips,atol=1e-13,rtol=0)):
            if edge-selected[-1]<5e-6:selected[-1]=edge
            else:selected.append(edge)
    ze=np.array(selected)
    r, z, aa, ra, volume = metric(re, ze)
    R, Z = np.meshgrid(r, z, indexing="ij")
    active = ~((R >= g.c) & (R <= g.d) & (Z < 0))
    active &= ~((R >= g.a) & (R <= g.b) & (Z < g.inner_tip))
    active &= ~((R >= g.gas_id_um*.5e-6) & (R <= g.gas_od_um*.5e-6) & (Z < -g.gas_recess_mm*1e-3))
    h = min(np.min(np.diff(re)), np.min(np.diff(ze)))
    epsilon = s.interface_cells*h
    signed = np.where(Z < 0, R-g.c, np.hypot(R, Z)-g.c)
    phi = .5*(1-np.tanh(signed/(2*epsilon)))
    phi[(Z < 0) & (R < g.c)] = 1.
    phi[~active] = 0.
    inlet_phi = ((r < g.c)&active[:,0]).astype(float)
    # Liquid at zmin is sampled from the solved liquid field.
    liquid_in = np.where((r < g.c) & active[:, 0], np.maximum(solvent.sample(np.minimum(r, g.c*.999), np.full_like(r, zmin), "velocity_z"), 0), 0)
    # Preserve each supplied flow separately before their recessed junction.
    for mask, target in [(r<g.a,f.qs),(r>g.b,f.qh)]:
        mask=mask&(r<g.c)&active[:,0]
        liquid_in[mask] *= target/np.sum(liquid_in[mask]*aa[mask])
    gas = annular_gas(config)
    gas_in = np.zeros_like(r)
    ingas = (r > g.d) & (r < g.gas_id_um*.5e-6)
    gas_in[ingas] = annular_profile(r[ingas], g.d, g.gas_id_um*.5e-6, gas["flow_m3_s"])
    if gas_in.sum() > 0: gas_in *= gas["flow_m3_s"]/np.sum(gas_in*aa)
    # If outer tube is less recessed than zmin, its wall and inlet are unresolved.
    inlet = liquid_in+gas_in
    u = np.zeros((len(r)+1, len(z))); w = np.zeros((len(r), len(z)+1))
    w[:, 0] = inlet
    pressure = np.zeros_like(phi)
    times, snapshots = [0.], [phi.copy()]
    ur_history=[np.zeros_like(phi)]; uz_history=[np.zeros_like(phi)]; p_history=[pressure.copy()]
    charge_model=None;charge_state=None
    q_history=[];v_history=[];er_history=[];ez_history=[];i_history=[];node_history=[]
    methanol_history=[];salt_history=[];species_boundary=np.zeros(2);species_error=np.zeros(2);species_roundoff=np.zeros(2)
    if s.free_charge_enabled:
        from .charge import AxisymmetricCharge
        from .species import transport_inventory,recover_concentration
        concentration=np.clip(solvent.sample(np.minimum(R,g.c*.999),np.minimum(Z,0.)),0,1)
        ionic=getattr(solvent,'ionic',None)
        salt=(ionic.sample(R,np.minimum(Z,0.),'salt_mol_m3') if ionic is not None else
              config.chemistry.ammonium_formate_mm*(1-concentration))
        concentration[phi<1e-14]=0.;salt[phi<1e-14]=0.
        inventories=[phi*concentration,phi*np.maximum(salt,0.)]
        initial_species=np.array([np.sum(a*volume) for a in inventories])
        inlet_species=[concentration[:,0].copy(),salt[:,0].copy()]
        methanol_history.append(concentration.copy());salt_history.append(salt.copy())
        charge_model=AxisymmetricCharge(config,electric,re,ze,active,phi)
        q_history.append(charge_model.q.copy());v_history.append(charge_model.initial_potential)
        er_history.append(charge_model.initial_er);ez_history.append(charge_model.initial_ez);i_history.append(np.nan)
        node_history.append(float(np.mean(charge_model.boundary['z0'][charge_model.inlet])))
    t = 0.; expected = float(np.sum(phi*volume)); initial_volume = expected
    max_div = 0.; max_mass_error = 0.; max_electric_force = 0.
    electric_stress = (np.zeros_like(phi),)*3
    liquid_in_volume=0.; liquid_out_volume=0.; gas_in_volume=0.; gas_out_volume=0.
    duration = s.duration_ms*1e-3
    for step in range(s.max_steps):
        rho_liquid=f.sample_density*(1-concentration)+f.sheath_density*concentration if charge_model is not None else f.sheath_density
        mu_liquid=np.exp((1-concentration)*np.log(f.sample_viscosity)+concentration*np.log(f.sheath_viscosity)) if charge_model is not None else f.sheath_viscosity
        rho = f.gas_density+(rho_liquid-f.gas_density)*phi
        mu = f.gas_viscosity+(mu_liquid-f.gas_viscosity)*phi
        uc, wc = (u[1:]+u[:-1])/2, (w[:, 1:]+w[:, :-1])/2
        uc[~active] = 0; wc[~active] = 0
        cfl = .18/(np.max(np.abs(uc))/np.min(np.diff(re))+np.max(np.abs(wc))/np.min(np.diff(ze))+1e-30)
        dt = min(cfl, .10*h*h/np.max(mu/rho), .15*np.sqrt(f.gas_density*h**3/f.surface_tension), duration-t)
        if dt <= 1e-13: raise RuntimeError("Spray time step collapsed; refine mesh / check parameters")
        gr, gz, normal_r, normal_z, curvature = _interface(phi, r, z, active)
        surface = (Z >= -h) & active
        if charge_model is not None:
            # Re-evaluate rejected trial steps from the same prior charge.
            # Electrical acceleration must not outrun the momentum step.
            for trial in range(8):
                charge_state=charge_model.predict(phi,u,w,dt,salt)
                electric_r=charge_state['force_r'];electric_z=charge_state['force_z']
                limit=.2*np.sqrt(h/max(float(np.max(np.hypot(electric_r,electric_z)/rho)),1e-30))
                if limit>=dt*(1-1e-8):break
                dt=min(dt,limit)
            else:raise RuntimeError('Charged spray force step did not stabilize; refine grid or reduce duration')
            if dt<=1e-13:raise RuntimeError('Charged spray step collapsed; check electrical assumptions')
            charge_model.commit(charge_state)
        else:
            if step % s.electric_update_every == 0:
                electric_stress = _electric_pressure(config, electric, phi, active, re, ze, normal_r, normal_z)
            trr_e,trz_e,tzz_e=electric_stress
            electric_r=-(trr_e*gr+trz_e*gz)*surface
            electric_z=-(trz_e*gr+tzz_e*gz)*surface
        max_electric_force=max(max_electric_force,float(np.max(np.hypot(electric_r,electric_z))))
        force_r = f.surface_tension*curvature*gr*surface+electric_r
        force_z = f.surface_tension*curvature*gz*surface+electric_z
        dur, duz = np.gradient(uc, r, z)
        dwr, dwz = np.gradient(wc, r, z)
        trr, tzz, trz, ttheta = 2*mu*dur, 2*mu*dwz, mu*(duz+dwr), 2*mu*uc/R
        visc_r = np.gradient(trr, r, axis=0)+np.gradient(trz, z, axis=1)+(trr-ttheta)/R
        visc_z = np.gradient(trz, r, axis=0)+np.gradient(tzz, z, axis=1)+trz/R
        adv_r = uc*_upwind_derivative(uc, uc, r, 0)+wc*_upwind_derivative(uc, wc, z, 1)
        adv_z = uc*_upwind_derivative(wc, uc, r, 0)+wc*_upwind_derivative(wc, wc, z, 1)
        acceleration_r = (visc_r+force_r)/rho-adv_r
        acceleration_z = (visc_z+force_z)/rho-adv_z
        acceleration_r[~active] = 0; acceleration_z[~active] = 0
        # Update the existing MAC velocities. Replacing them by interpolated cell
        # velocities would filter momentum every step even as dt tends to zero.
        u[1:-1] += dt*(acceleration_r[:-1]+acceleration_r[1:])/2
        u[0] = 0.; u[-1] = u[-2]
        w[:, 1:-1] += dt*(acceleration_z[:, :-1]+acceleration_z[:, 1:])/2
        w[:, 0] = inlet; w[:, -1] = w[:, -2]
        u[1:-1] *= active[:-1] & active[1:]
        w[:, 1:-1] *= active[:, :-1] & active[:, 1:]
        u, w, pressure = project(u, w, rho, active, re, ze, dt)
        div = divergence(u, w, re, ze)
        max_div = max(max_div, float(np.max(np.abs(div[active]))))
        fr, fz = _phase_flux(phi, u, w, inlet_phi, aa, ra)
        liquid_in_volume+=dt*fz[:,0].sum()
        liquid_out_volume+=dt*(fr[-1].sum()+fz[:,-1].sum())
        gas_in_volume+=dt*np.sum(aa*w[:,0]-fz[:,0])
        gas_out_volume+=dt*(np.sum(ra[-1]*u[-1]-fr[-1])+np.sum(aa*w[:,-1]-fz[:,-1]))
        expected -= dt*(fr[-1].sum()-fr[0].sum()+fz[:, -1].sum()-fz[:, 0].sum())
        phi -= dt*(np.diff(fr, axis=0)+np.diff(fz, axis=1))/volume
        if charge_model is not None:
            adv_phi=phi.copy();adv_concentrations=[]
            for j,cfield in enumerate((concentration,salt)):
                inventories[j],boundary=transport_inventory(inventories[j],cfield,fr,fz,inlet_species[j],volume,dt)
                species_boundary[j]+=boundary;adv_concentrations.append(recover_concentration(inventories[j],adv_phi))
            phi,refr,refz=bounded_reinitialization(phi,normal_r,normal_z,active,re,ze,dt,.05,epsilon,return_flux=True)
            for j,cfield in enumerate(adv_concentrations):
                inventories[j],boundary=transport_inventory(inventories[j],cfield,refr,refz,np.zeros_like(r),volume,dt)
                species_boundary[j]+=boundary
            concentration=recover_concentration(inventories[0],phi);salt=recover_concentration(inventories[1],phi)
            resolved_liquid=phi>1e-10
            if (np.any(((concentration<-1e-6)|(concentration>1+1e-6))&resolved_liquid) or np.any((salt<-1e-5)&resolved_liquid)):
                bad=np.unravel_index(np.argmax(concentration),concentration.shape)
                raise FloatingPointError(f'Unbounded charged-spray liquid inventory: C={concentration.min():.6g}..{concentration.max():.6g}, salt_min={salt.min():.6g}, phi_at_max={phi[bad]:.6g}; reduce the transport step')
            concentration=np.clip(concentration,0,1);salt=np.maximum(salt,0)
            for j,cfield in enumerate((concentration,salt)):
                bounded=phi*cfield
                species_roundoff[j]+=float(np.sum((bounded-inventories[j])*volume))
                inventories[j]=bounded
            species_error=np.maximum(species_error,np.abs(np.array([np.sum(a*volume) for a in inventories])-initial_species+species_boundary))
        else:phi=bounded_reinitialization(phi, normal_r, normal_z, active, re, ze, dt, .05, epsilon)
        phi[~active] = 0.
        if phi.min() < -1e-6 or phi.max() > 1+1e-6 or not np.isfinite(phi).all():
            raise RuntimeError("Spray phase lost boundedness; reduce CFL / refine mesh")
        phi = np.clip(phi, 0, 1)  # Roundoff only; mass diagnostics include this change.
        t += dt
        error = abs(float(np.sum(phi*volume))-expected)/max(initial_volume, 1e-30)
        max_mass_error = max(max_mass_error, error)
        if step % s.save_every == 0 or t >= duration:
            times.append(t); snapshots.append(phi.copy())
            ur_history.append((u[1:]+u[:-1])/2); uz_history.append((w[:,1:]+w[:,:-1])/2); p_history.append(pressure.copy())
            if charge_model is not None:
                q_history.append(charge_model.q.copy());v_history.append(charge_state['potential'])
                er_history.append(charge_state['er']);ez_history.append(charge_state['ez'])
                i_history.append(charge_state['supply_current_a']);node_history.append(charge_state['node_voltage_v'])
                methanol_history.append(concentration.copy());salt_history.append(salt.copy())
        if step % 250 == 0:
            progress(f"Two-phase: {t*1e3:.4f} / {s.duration_ms:g} ms; mass error {error:.2e}")
        if t >= duration*(1-1e-10): break
    if times[-1] != t:
        times.append(t); snapshots.append(phi.copy())
        ur_history.append((u[1:]+u[:-1])/2); uz_history.append((w[:,1:]+w[:,:-1])/2); p_history.append(pressure.copy())
        if charge_model is not None:
            q_history.append(charge_model.q.copy());v_history.append(charge_state['potential'])
            er_history.append(charge_state['er']);ez_history.append(charge_state['ez'])
            i_history.append(charge_state['supply_current_a']);node_history.append(charge_state['node_voltage_v'])
            methanol_history.append(concentration.copy());salt_history.append(salt.copy())
    droplets = identify(phi, volume, r, z)
    diag = {"method": "experimental variable-density two-phase NS + conservative level set + Maxwell tensor traction",
        "elapsed_s": t, "requested_s": duration, "completed": bool(t >= duration*(1-1e-10)),
        "steps": step+1, "max_relative_liquid_volume_error": max_mass_error,
        "max_divergence_s_inv": max_div, "minimum_cell_um": h*1e6,
        "detached_components": len(droplets), "gas": gas,
        "max_electric_force_density_n_m3": max_electric_force,
        "domain_z_m": [zmin,zend], "wall_material": "quartz dielectric; three resolved tube ends",
        "liquid_in_volume_m3": liquid_in_volume, "liquid_out_volume_m3": liquid_out_volume,
        "gas_in_volume_m3": gas_in_volume, "gas_out_volume_m3": gas_out_volume,
        "gas_field": "velocity and pressure solved in both nitrogen and external air; saved at every phase frame",
        "initial_condition": "hemispherical meniscus; initially quiescent exterior; pinned middle-tube lip",
        "limitations": ["research backend: no validated steady cone-jet prediction", "no spray surface-charge transport, evaporation or MS capture",
                        "axisymmetry excludes whipping and asymmetric breakup", "coarse grids cannot resolve sub-grid jets/droplets",
                        "gas supply hydraulics use assumed length; not a measured mass flow",
                        "liquid and gas feed continuously at rear plane; quartz ends inside calculation domain",
                        "sharp quartz edges; contact angle / edge rounding unmeasured; no wetting/contact-line calibration",
                        "detached-drop electric charge is not conserved by prescribed liquid potential approximation",
                        "MS plane is a truncation/open pressure boundary, not a resolved inlet aperture or vacuum flow"]}
    if charge_model is not None:
        diag['electric_coupling']=charge_model.diagnostics()
        diag['species_coupling']=dict(methanol_inventory_balance_error_m3=float(species_error[0]),salt_inventory_balance_error_mol=float(species_error[1]),
            numerical_methanol_inventory_adjustment_m3=float(species_roundoff[0]),numerical_salt_inventory_adjustment_mol=float(species_roundoff[1]),
            method='phase-flux consistent conservative liquid methanol/salt advection, including phase compression; salt updates Ohmic conductivity',
            scope='neutral methanol advection; initial/inlet mixing from solved diffusion field; no additional two-phase molecular diffusion, evaporation or full PNP')
        diag['method']+='; conservative free-charge advection, implicit Ohmic relaxation and remote RC closure'
        diag['limitations']=[x for x in diag['limitations'] if 'no spray surface-charge' not in x and 'charge is not conserved' not in x]
        diag['limitations'].extend(['No full ion PNP, electrode reactions, evaporation or calibrated MS capture.',
            'Effective support conductivity is unmeasured; composition and temperature dependence are approximate.',
            'Electrical field/charge are a first-order split substep before the saved interface update; grid/time convergence is required.',
            'Outer electrical boundaries are inherited from the solved background, not a full chamber circuit.'])
    return SprayResult(re, ze, np.array(times), np.array(snapshots), (u[1:]+u[:-1])/2,
                       (w[:, 1:]+w[:, :-1])/2, pressure, active, diag, droplets,
                       np.array(ur_history),np.array(uz_history),np.array(p_history),
                       *([np.array(x) for x in (q_history,v_history,er_history,ez_history,i_history,node_history,methanol_history,salt_history)]
                         if charge_model is not None else [None]*8))
