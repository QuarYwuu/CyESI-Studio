"""Finite-radius cell transport through Cartesian tip fields.

A segmented Poiseuille model supplies upstream transport. The tip uses all
three velocity and electric-field components. Cells are dilute, rigid and
overdamped; membrane deformation, pair interactions and Dean flow are excluded.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from ..particle.cell_transport import ParticleResult, _initial_radius
from ..particle.long_tube import simulate as bulk_transport
from ..particle.forces import (brownian_diffusivity, lubrication_factors,
    electrophoretic_mobility, dielectrophoretic_force)
from ..geometry.coaxial_geometry import wall_distance
from ..electro.electrostatics import EPS0


@dataclass
class NativeParticleResult(ParticleResult):
    trajectory_x: np.ndarray | None = None
    trajectory_y: np.ndarray | None = None
    transport_config: object = None


def simulate_native(config,volume,path=None,progress=lambda value:None):
    """Return a ParticleResult-compatible population with genuine tip x/y/z.

    ``config.fluid.sample_flow_ul_min`` must already be the pressure-network
    operating point. The supplied configuration and volume fields are never
    edited. ``transport_config`` records any route-gravity substitution.
    """
    cfg=deepcopy(config)
    requested_resolved_mm=cfg.geometry.tip_resolved_length_mm
    native_entry=float(volume.grid.edges[2][0])
    physical_entry=cfg.geometry.inner_tip-cfg.geometry.sample_length_mm*.001
    if native_entry<physical_entry-1e-10:
        raise ValueError('The native liquid inlet lies upstream of the physical sample tube entrance')
    # The gas-supply extent may move the Cartesian inlet upstream of the
    # Match the upstream extension to the Cartesian inlet position.
    actual_resolved=cfg.geometry.inner_tip-native_entry
    if actual_resolved<=0:raise ValueError('Native liquid inlet must precede the inner capillary lip')
    cfg.geometry.tip_resolved_length_mm=actual_resolved*1000
    if path is not None:
        cfg.supply.orientation='custom';cfg.supply.segments=path.segments_for_supply(64)
    cfg.validate();g,f,c,n=cfg.geometry,cfg.fluid,cfg.cell,cfg.numerics
    rng=np.random.default_rng(n.seed);N=c.count;R=c.diameter_um*.5e-6
    initial=_initial_radius(cfg,rng)
    progress(f'3-D transport: developed upstream route, {N} finite spheres')
    bulk=bulk_transport(cfg,initial,rng)
    ids=np.asarray(bulk.history['ids'],int)
    status=bulk.status.copy();xyz=np.c_[bulk.x,bulk.y,np.full(N,g.liquid_start)]
    stopped=status!=0
    xyz[stopped,2]=g.inner_tip-g.sample_length_mm*.001+bulk.distance[stopped]
    elapsed=np.zeros(N);dose=np.zeros(N);exposure=np.zeros(N)
    residence=np.full(N,np.nan);exit_tip=residence.copy();outlet_r=residence.copy();outlet_c=residence.copy()
    stress=deepcopy(bulk.stress);stress['extension_dose_pa_s']=np.zeros(N)
    bh=bulk.history
    tx=[bh['x_m'][0].copy(),xyz[ids,0].copy()]
    ty=[bh['y_m'][0].copy(),xyz[ids,1].copy()]
    tz=[np.full(len(ids),g.inner_tip-g.sample_length_mm*.001),xyz[ids,2].copy()]
    tt=[np.zeros(len(ids)),bulk.time[ids].copy()];tc=[np.zeros(len(ids)),np.zeros(len(ids))]
    axes=volume.grid.centers;lower=np.array([a[0] for a in axes]);upper=np.array([a[-1] for a in axes])
    liquid=volume.material==1
    # Gradients use nearest liquid values in solid/gas cells. Differentiating
    # Restrict the energy gradient to fluid cells to avoid spurious wall forces.
    nearest=distance_transform_edt(~liquid,return_distances=False,return_indices=True)
    def fill(values):return np.asarray(values)[tuple(nearest)]
    electric=fill(volume.electric)
    energy=np.sum(electric**2,axis=-1)
    grad_e2=np.stack(np.gradient(energy,*axes,edge_order=1),axis=-1)
    velocity=fill(volume.velocity)
    gradients=np.stack([np.stack(np.gradient(velocity[...,j],*axes,edge_order=1),axis=-1)
                        for j in range(3)],axis=-2)
    omega=np.stack([gradients[...,2,1]-gradients[...,1,2],
                    gradients[...,0,2]-gradients[...,2,0],
                    gradients[...,1,0]-gradients[...,0,1]],axis=-1)
    composition=np.clip(fill(volume.methanol),0,1)
    viscosity=np.exp((1-composition)*np.log(f.sample_viscosity)+composition*np.log(f.sheath_viscosity))
    symmetric=(gradients+np.swapaxes(gradients,-1,-2))*.5
    deviator=symmetric-np.trace(symmetric,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3
    tau=2*viscosity[...,None,None]*deviator
    equivalent=np.sqrt(1.5*np.sum(tau*tau,axis=(-2,-1)))
    # One interpolation evaluates the same trilinear weights for all fifteen
    # scalar channels. Seven separate interpolators repeated these weights at
    # every ODE step and dominated runtime for small response populations.
    channels=np.concatenate([composition[...,None],velocity,electric,grad_e2,omega,
                             equivalent[...,None],np.abs(tau[...,2,2,None])],axis=-1)
    joint=RegularGridInterpolator(axes,channels,bounds_error=False,fill_value=np.nan)
    eps=EPS0*cfg.chemistry.liquid_relative_permittivity
    if path is None:
        from ..fluid.supply import segments
        gravity=np.array([segments(cfg)[-1][key] for key in ('gx','gy','gz')])
    else:
        _,T,Bn,Bb=path.sample(path.length_m)
        global_g=np.array([0.,-9.80665,0.]) if cfg.supply.gravity_enabled else np.zeros(3)
        gravity=np.array([global_g@Bn,global_g@Bb,global_g@T])
    min_grid=min(float(np.diff(edge).min()) for edge in volume.grid.edges)
    max_speed=float(np.nanmax(np.linalg.norm(volume.velocity[liquid],axis=-1)))
    # The fastest sheath-grid cell is not the speed of every sample cell.
    # A global sheath-based CFL forced microsecond steps through seconds of
    # slow capillary transit. Enforce the mesh-displacement bound per particle
    # after its actual terminal velocity has been computed instead.
    dt=n.particle_dt_s
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid native particle integration step')
    nominal_steps=int(np.ceil(n.particle_tip_timeout_s/dt))
    max_steps=min(2000000,nominal_steps*20);save_every=max(1,nominal_steps//400)
    max_slip_re=0.;max_step_relaxation=0.
    for step in range(max_steps):
        idx=np.flatnonzero(status==0)
        if not len(idx):break
        p=xyz[idx];r=np.hypot(p[:,0],p[:,1]);sampled=joint(np.clip(p,lower,upper))
        C=np.clip(sampled[:,0],0,1)
        mu=np.exp((1-C)*np.log(f.sample_viscosity)+C*np.log(f.sheath_viscosity))
        rho=f.sample_density*(1-C)+f.sheath_density*C
        uf=sampled[:,1:4];ef=sampled[:,4:7];ge2=sampled[:,7:10];w=sampled[:,10:13]
        distance,nr,nz=wall_distance(r,p[:,2],g)
        radial=np.c_[p[:,0]/np.maximum(r,1e-30),p[:,1]/np.maximum(r,1e-30)]
        normal=np.c_[radial*nr[:,None],nz]
        # The radial direction is immaterial on axis; keep a valid local basis.
        normal[r<1e-20,0]=nr[r<1e-20]
        normal_drag,parallel=lubrication_factors(R,distance-R)
        if not cfg.supply.lubrication_enabled:normal_drag=np.ones(len(idx));parallel=normal_drag.copy()
        drag=6*np.pi*mu*R
        mobility=electrophoretic_mobility(eps,c.zeta_mv*.001,mu)
        extra=mobility[:,None]*ef+dielectrophoretic_force(eps,R,c.clausius_mossotti,ge2)/drag[:,None]
        extra+=(c.density-rho)[:,None]*(4/3*np.pi*R**3)*gravity/drag[:,None]
        A=parallel[:,None,None]*np.eye(3)+(normal_drag-parallel)[:,None,None]*normal[:,:,None]*normal[:,None,:]
        rhs=parallel[:,None]*uf+(1-parallel)[:,None]*np.sum(uf*normal,axis=1)[:,None]*normal+extra
        if cfg.supply.saffman_enabled:
            # 3-D low-slip Saffman closure: slip cross vorticity. It vanishes
            # near a wall and cannot manufacture lift from a timestep lag.
            wn=np.linalg.norm(w,axis=1);factor=6.46*rho*R**2*np.sqrt(mu/rho)/drag
            factor*=np.clip((distance-R)/(2*R),0,1)/np.sqrt(np.maximum(wn,1e-30))
            B=np.zeros_like(A)
            B[:,0,1]=w[:,2];B[:,0,2]=-w[:,1];B[:,1,0]=-w[:,2]
            B[:,1,2]=w[:,0];B[:,2,0]=w[:,1];B[:,2,1]=-w[:,0]
            B*=factor[:,None,None];A+=B;rhs+=np.einsum('nij,nj->ni',B,uf)
        # Explicit singleton RHS works with both NumPy 1.x and 2.x stacked
        # solve semantics (NumPy 2 treats every 2-D RHS as a matrix).
        target=np.linalg.solve(A,rhs[...,None])[...,0]
        valid=np.isfinite(target).all(axis=1)
        status[idx[~valid]]=3
        local_dt=np.minimum(dt,.5*min_grid/np.maximum(np.linalg.norm(target,axis=1),1e-20))
        local_dt=np.minimum(local_dt,np.maximum(0.,n.particle_tip_timeout_s-elapsed[idx]))
        local_dt[~valid]=0.
        displacement=target*local_dt[:,None]
        if c.use_brownian:
            D=brownian_diffusivity(f.temperature_k,mu,R)
            displacement+=rng.normal(size=displacement.shape)*np.sqrt(2*D*local_dt)[:,None]
        old=xyz[idx].copy();xyz[idx]+=np.where(valid[:,None],displacement,0.)
        elapsed[idx]+=local_dt;dose[idx]+=C*local_dt;exposure[idx]=C
        eq=sampled[:,13];extension=sampled[:,14]
        stress['shear_dose_pa_s'][idx]+=eq*local_dt
        stress['damage_exposure'][idx]+=eq**cfg.supply.shear_damage_exponent*local_dt
        stress['extension_dose_pa_s'][idx]+=extension*local_dt
        stress['peak_equivalent_stress_pa'][idx]=np.maximum(stress['peak_equivalent_stress_pa'][idx],eq)
        newr=np.hypot(xyz[idx,0],xyz[idx,1]);newdist,_,_=wall_distance(newr,xyz[idx,2],g)
        hit=newdist<R*(1-1e-6);reverse=xyz[idx,2]<g.liquid_start-1e-9
        done=(xyz[idx,2]>=0)&~hit&~reverse&valid
        status[idx[hit]]=2;status[idx[reverse]]=3
        exited=idx[done]
        if len(exited):
            frac=np.clip(-old[done,2]/np.maximum(displacement[done,2],1e-30),0,1)
            xyz[exited]=old[done]+frac[:,None]*displacement[done];xyz[exited,2]=0.
            elapsed[exited]-=local_dt[done]*(1-frac)
            dose[exited]-=C[done]*local_dt[done]*(1-frac)
            stress['shear_dose_pa_s'][exited]-=eq[done]*local_dt[done]*(1-frac)
            stress['damage_exposure'][exited]-=eq[done]**cfg.supply.shear_damage_exponent*local_dt[done]*(1-frac)
            stress['extension_dose_pa_s'][exited]-=extension[done]*local_dt[done]*(1-frac)
            outlet_r[exited]=np.hypot(xyz[exited,0],xyz[exited,1]);exit_tip[exited]=elapsed[exited]
            residence[exited]=bulk.time[exited]+elapsed[exited]
            outlet_c[exited]=joint(np.clip(xyz[exited],lower,upper))[:,0];status[exited]=1
        max_slip_re=max(max_slip_re,float(np.max(rho*2*R*np.linalg.norm(uf-target,axis=1)/mu)))
        relaxation=c.density*(4/3*np.pi*R**3)/drag
        max_step_relaxation=max(max_step_relaxation,float(np.max(relaxation/np.maximum(local_dt,1e-30))))
        timed=(elapsed>=n.particle_tip_timeout_s-1e-12)&(status==0);status[timed]=3
        if step%save_every==0:
            tx.append(xyz[ids,0].copy());ty.append(xyz[ids,1].copy());tz.append(xyz[ids,2].copy())
            tt.append((bulk.time[ids]+elapsed[ids]).copy());tc.append(exposure[ids].copy())
        if step%1000==0:progress(f'Native 3-D tip transport: {float(elapsed.max()):.3f} s, active {np.sum(status==0)} / {N}')
    step_budget_censored=int(np.sum(status==0));status[status==0]=3
    tx.append(xyz[ids,0].copy());ty.append(xyz[ids,1].copy());tz.append(xyz[ids,2].copy())
    tt.append((bulk.time[ids]+elapsed[ids]).copy());tc.append(exposure[ids].copy())
    ok=status==1
    diagnostics=dict(count=N,exited=int(ok.sum()),wall_contacts=int(np.sum(status==2)),
        censored_or_reverse=int(np.sum(status==3)),transport_fraction=float(ok.mean()),
        focusing_fraction_of_exits=float(np.mean(outlet_r[ok]<=c.focusing_radius_um*1e-6)) if ok.any() else None,
        outlet_r95_um=float(np.quantile(outlet_r[ok],.95)*1e6) if ok.any() else None,
        mean_residence_s=float(np.mean(residence[ok])) if ok.any() else None,
        mean_tip_residence_s=float(np.mean(exit_tip[ok])) if ok.any() else None,
        mean_outlet_methanol_fraction=float(np.mean(outlet_c[ok])) if ok.any() else None,
        particle_backend='native_cartesian_overdamped',brownian_enabled=c.use_brownian,
        brownian_control='size_cutoff' if c.brownian_enabled is None else 'manual',
        tip_step_s=dt,max_slip_reynolds=max_slip_re,max_relaxation_over_step=max_step_relaxation,
        requested_resolved_length_mm=requested_resolved_mm,
        actual_native_resolved_length_mm=actual_resolved*1000,native_entry_z_m=native_entry,
        adaptive_step='per-cell displacement <= half the smallest native grid width; configured dt is the maximum',
        numerical_step_budget=max_steps,steps_executed=step+1 if max_steps else 0,
        numerical_step_budget_censored=step_budget_censored,
        bulk_length_mm=g.bulk_length*1000,tube_orientation=cfg.supply.orientation,
        actual_sample_flow_ul_min=f.sample_flow_ul_min,sample_control=cfg.supply.sample_control,
        bulk_method='developed segmented flow with route-projected gravity; no Dean secondary flow',
        resolved_method='native 3-D velocity/E/composition with finite-sphere wall mobility, EP, DEP, low-slip Saffman and optional Brownian diffusion',
        inertia_scope='overdamped terminal velocity; cell inertia not integrated',
        wall_model='bulk exclusion/conditional capture; tip finite-radius wall crossing stops the saved response',
        post_nozzle_time_s=None,conditional_mechanically_retained_fraction=None,
        notes=['3-D tip motion is not constrained to a meridian or assigned a fixed azimuth.',
               'Long-pipe hydrodynamics remain a developed-flow extension, not full curved-tube 3-D CFD.',
               'Rigid dilute cells do not deform, collide with one another, or feed back on the liquid.'])
    if cfg.supply.shear_damage_coefficient is not None:
        stress['conditional_mechanical_survival']=np.exp(-cfg.supply.shear_damage_coefficient*stress['damage_exposure'])
        diagnostics['conditional_mechanically_retained_fraction']=float(np.mean(ok*stress['conditional_mechanical_survival']))
    return NativeParticleResult(initial,residence,exit_tip,outlet_r,outlet_c,dose,status,ids,
        np.asarray(tt),np.hypot(np.asarray(tx),np.asarray(ty)),np.asarray(tz),np.asarray(tc),diagnostics,
        stress,bulk.history,np.asarray(tx),np.asarray(ty),cfg)
