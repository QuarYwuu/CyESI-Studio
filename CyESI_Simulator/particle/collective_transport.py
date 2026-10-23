"""Continuous long-capillary populations on a shared physical clock.

Events enter at their recorded source times, including prehistory. The domain
ends at the resolved-tip entrance. Changes to downstream signal timing require
new tip transport from the resulting entrance positions.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from ..fluid.supply import segments
from .forces import brownian_diffusivity,saffman_lift
from .confined_mobility import TubeKernel,self_mobility,solve_group,membrane_step,REFERENCES


@dataclass
class CollectiveParameters:
    start_s: float=0.
    duration_s: float=30.
    dt_s: float=.02
    sample_interval_s: float=.1
    prehistory_interval_s: float=2.
    hydrodynamics: bool=True
    wall_reflections: bool=True
    pair_lubrication: bool=True
    exclusion: bool=True
    inlet_queue: bool=True
    faxen: bool=True
    membrane: bool=False
    membrane_shear_modulus_n_m: float=.001
    max_shape_strain: float=.15
    inertia: bool=False
    compare_independent: bool=True
    roughness_um: float=.1
    neighbor_diameters: float=3.
    wall_theta: int=12
    wall_rings: int=7
    wall_half_length_radii: float=8.
    max_events: int=10000
    max_steps: int=300000
    seed: int|None=None

    def validate(self):
        numbers=['start_s','duration_s','dt_s','sample_interval_s','prehistory_interval_s',
            'membrane_shear_modulus_n_m','max_shape_strain','roughness_um','neighbor_diameters','wall_half_length_radii']
        if any(isinstance(getattr(self,k),(bool,str)) for k in numbers) or not np.isfinite([getattr(self,k) for k in numbers]).all():raise ValueError("Population parameters must be finite numbers.")
        if self.start_s<0 or not 0<self.duration_s<=60:raise ValueError("Observation start must be nonnegative and duration must be between 0 and 60 s.")
        if not 1e-5<=self.dt_s<=.1 or not 1e-4<=self.sample_interval_s<=2 or not .1<=self.prehistory_interval_s<=30:raise ValueError("Invalid integration step or output interval.")
        if not .001<=self.roughness_um<=2 or not 1<=self.neighbor_diameters<=10:raise ValueError("Invalid contact gap or neighbour cutoff.")
        if self.membrane_shear_modulus_n_m<=0 or not .01<=self.max_shape_strain<=.3:raise ValueError("Membrane shear modulus must be positive and the small-strain limit must be between 0.01 and 0.3.")
        for key in ('hydrodynamics','wall_reflections','pair_lubrication','exclusion','inlet_queue','faxen','membrane','inertia','compare_independent'):
            if not isinstance(getattr(self,key),bool):raise ValueError(key+'  Must be a Boolean ')
        for key,low,high in [('wall_theta',8,32),('wall_rings',5,21),('max_events',1,30000),('max_steps',1,2000000)]:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or not low<=v<=high:raise ValueError(key+'  Outside the allowed integer range ')
        if not 4<=self.wall_half_length_radii<=30:raise ValueError("Wall-patch half-length must be between 4 and 30 capillary radii.")
        if self.seed is not None and (isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0):raise ValueError("Random seed must be a nonnegative integer.")
        return self

    def to_dict(self):return asdict(self)


def interval_statistics(times,ids=None):
    times=np.asarray(times,float);valid=np.isfinite(times);order=np.argsort(times[valid]);t=times[valid][order]
    gaps=np.diff(t);mean=float(gaps.mean()) if len(gaps) else None
    return dict(count=len(t),gap_count=len(gaps),mean_interval_s=mean,
        cv=None if not len(gaps) or mean<=0 else float(gaps.std()/mean),
        q10_s=None if not len(gaps) else float(np.quantile(gaps,.1)),
        q50_s=None if not len(gaps) else float(np.median(gaps)),
        close_fraction_30ms=None if not len(gaps) else float(np.mean(gaps<.03)),
        intervals_s=gaps.tolist(),times_s=t.tolist(),
        event_ids=[] if ids is None else np.asarray(ids)[valid][order].astype(int).tolist(),
        scope='All physical crossings in the declared observation window; not a selected survivor cohort')


def pair_statistics(time,positions,states,start,end,diameter):
    """Observed same-lane neighbour gap/relative-speed relation, no force fit."""
    bounds=np.array([1.,1.25,1.5,2.,3.,5.,10.,20.,50.,100.,np.inf])
    samples=[[] for _ in range(len(bounds)-1)];axial_gaps=[]
    for k in np.flatnonzero((time>=start)&(time<=end)):
        live=np.flatnonzero(states[k]==1)
        if len(live)<2:continue
        order=live[np.argsort(positions[k,live,2])];back,front=order[:-1],order[1:]
        delta=positions[k,front]-positions[k,back]
        axial_gaps.extend((delta[:,2]/diameter).tolist())
        if k==0:continue
        valid=(states[k-1,back]==1)&(states[k-1,front]==1)&(np.linalg.norm(delta[:,:2],axis=1)<diameter/2)
        valid &=np.isfinite(positions[k-1,back]).all(axis=1)&np.isfinite(positions[k-1,front]).all(axis=1)
        dist=np.linalg.norm(delta[valid],axis=1)/diameter
        previous=positions[k-1,front[valid],2]-positions[k-1,back[valid],2]
        rel=(delta[valid,2]-previous)/(time[k]-time[k-1])*1e6
        for gap,v in zip(dist,rel):
            j=np.searchsorted(bounds,gap,side='right')-1
            if 0<=j<len(samples):samples[j].append(float(v))
    return dict(axial_gaps_in_diameters=axial_gaps,
        bins=[dict(gap_low_d=float(bounds[i]),gap_high_d=None if not np.isfinite(bounds[i+1]) else float(bounds[i+1]),
                   count=len(values),mean_relative_speed_um_s=None if not values else float(np.mean(values)),
                   median_relative_speed_um_s=None if not values else float(np.median(values))) for i,values in enumerate(samples)],
        notes=[' Positive relative velocity indicates separation ,  Negative indicates approach ;  No fitted target spacing .',
               ' Velocity from common-time finite differences ;  Pairs with transverse separation below half a diameter .',
               ' Repeated observations of the same pair ;  Temporally correlated samples . Zero-crossing means require further pair-dynamics analysis .'])


def _components(xyz,cutoff):
    n=len(xyz);parent=np.arange(n)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for i,j in cKDTree(xyz).query_pairs(cutoff,output_type='ndarray'):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    groups={}
    for i in range(n):groups.setdefault(find(i),[]).append(i)
    return [np.array(v,int) for v in groups.values()]


def initial_positions(config,ids,seed):
    # Fixed per-event independent random stream: repeat/ablation uses exactly
    # the same inlet points, independent of dt, neighbour discovery and noise.
    a=config.geometry.a;R=config.cell.diameter_um*.5e-6
    limit=a-R-max(config.supply.wall_capture_gap_um, .001)*1e-6
    if limit<=0:raise ValueError("The cell radius and gap exceed the capillary radius.")
    smax=(limit/a)**2;points=[]
    for i in ids:
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(i),1701]))
        r=a*np.sqrt(1-np.sqrt(1-rng.random()*(2*smax-smax*smax)));angle=rng.uniform(0,2*np.pi)
        points.append([r*np.cos(angle),r*np.sin(angle),0.])
    return np.asarray(points,float).reshape(-1,3)


@dataclass
class CollectiveResult:
    parameters: CollectiveParameters
    source_id: str
    event_ids: np.ndarray
    injection_s: np.ndarray
    time_s: np.ndarray
    positions_m: np.ndarray
    states: np.ndarray
    shape_strain: np.ndarray
    crossings_s: np.ndarray
    baseline_positions_m: np.ndarray|None
    baseline_states: np.ndarray|None
    baseline_crossings_s: np.ndarray|None
    diagnostics: dict
    analysis: dict

    def frame(self,time_s,baseline=False):
        pos=self.baseline_positions_m if baseline else self.positions_m
        states=self.baseline_states if baseline else self.states
        if pos is None or not self.time_s[0]<=time_s<=self.time_s[-1]:return []
        k=int(np.clip(np.searchsorted(self.time_s,time_s,side='right')-1,0,len(self.time_s)-1));q=min(k+1,len(self.time_s)-1)
        f=0 if q==k else (time_s-self.time_s[k])/(self.time_s[q]-self.time_s[k])
        cells=[]
        for j,i in enumerate(self.event_ids):
            # Hold a valid frame at admission/exit; never interpolate through
            # an unborn cell or jump a captured cell into a mobile path.
            state=int(states[k,j]);point=pos[k,j]
            if state not in (1,3):continue
            if states[q,j]==state and np.isfinite(pos[q,j]).all():point=point+(pos[q,j]-point)*f
            cells.append(dict(event_id=int(i),local_xyz_m=point.tolist(),state='mobile' if state==1 else 'wall_bound',
                shape_strain=None if baseline else float(self.shape_strain[k,j]),age_s=float(time_s-self.injection_s[j])))
        return cells

    def history(self,event_id):
        found=np.flatnonzero(self.event_ids==int(event_id))
        if not len(found):raise ValueError("This event ID is absent from the population study.")
        j=int(found[0]);valid=np.isfinite(self.positions_m[:,j]).all(axis=1)
        return dict(event_id=int(event_id),injection_s=float(self.injection_s[j]),time_s=self.time_s[valid].tolist(),
            positions_m=self.positions_m[valid,j].tolist(),states=self.states[valid,j].tolist(),
            shape_strain=self.shape_strain[valid,j].tolist(),crossings_s=self.crossings_s[j].tolist())

    def save(self,directory):
        from ..scene3d.data import _clean
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        arrays={key:getattr(self,key) for key in ('event_ids','injection_s','time_s','positions_m','states','shape_strain','crossings_s')}
        for key in ('baseline_positions_m','baseline_states','baseline_crossings_s'):
            if getattr(self,key) is not None:arrays[key]=getattr(self,key)
        np.savez_compressed(root/'collective.npz',**arrays)
        metadata=dict(schema_version=1,source_id=self.source_id,parameters=self.parameters.to_dict(),diagnostics=self.diagnostics,analysis=self.analysis)
        (root/'collective.json').write_text(json.dumps(_clean(metadata),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        return root

    @classmethod
    def load(cls,directory):
        """Read arrays without pickle; the caller verifies record provenance."""
        root=Path(directory)
        metadata=json.loads((root/'collective.json').read_text(encoding='utf-8'))
        if metadata.get('schema_version')!=1:raise ValueError("Unsupported population-record format.")
        p=CollectiveParameters(**metadata['parameters']).validate()
        names=('event_ids','injection_s','time_s','positions_m','states','shape_strain','crossings_s',
               'baseline_positions_m','baseline_states','baseline_crossings_s')
        with np.load(root/'collective.npz',allow_pickle=False) as data:
            arrays={name:data[name] if name in data else None for name in names}
        n=len(arrays['event_ids']);nt=len(arrays['time_s'])
        if n>p.max_events or nt*max(n,1)>6000000:raise ValueError("This record exceeds the history-storage limit.")
        shapes={'event_ids':(n,),'injection_s':(n,),'time_s':(nt,),
                'positions_m':(nt,n,3),'states':(nt,n),'shape_strain':(nt,n),'crossings_s':(n,5),
                'baseline_positions_m':(nt,n,3),'baseline_states':(nt,n),'baseline_crossings_s':(n,5)}
        for name,shape in shapes.items():
            if arrays[name] is None:
                if not name.startswith('baseline_'):raise ValueError(' Population record missing  '+name)
            elif arrays[name].shape!=shape:raise ValueError(' Inconsistent population array shape : '+name)
        t=arrays['time_s']
        if nt<1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError("Invalid population clock.")
        return cls(p,metadata['source_id'],**arrays,diagnostics=metadata['diagnostics'],analysis=metadata['analysis'])


def _advance(config,ids,birth,initial,p,progress,coupling):
    g=config.geometry;f=config.fluid;R=config.cell.diameter_um*.5e-6;a=g.a;mu=f.sample_viscosity
    length=g.bulk_length;start_z=g.inner_tip-g.sample_length_mm*.001;rough=p.roughness_um*1e-6
    if length<=0:raise ValueError("The resolved tip occupies the full sample capillary, leaving no upstream tube domain.")
    if a-R-rough<=0:raise ValueError("The cell and roughness gap do not fit inside the capillary.")
    mean=f.qs/(np.pi*a*a);re=f.sample_density*mean*2*a/mu
    if p.inertia and re>.5:raise ValueError("First-order inertia requires Re <= 0.5. Use a finite-Re solver above this range.")
    seed=config.numerics.seed+1703 if p.seed is None else p.seed
    N=len(ids);xyz=np.full((N,3),np.nan);state=np.zeros(N,np.uint8);Q=np.zeros((N,3,3))
    planes=np.linspace(0,length,5);cross=np.full((N,5),np.nan)
    route=segments(config);boundaries=np.array([row['end_m'] for row in route]);gravity=np.array([[row[k] for k in ('gx','gy','gz')] for row in route])
    zeta=6*np.pi*mu*R;weight=(config.cell.density-f.sample_density)*(4/3*np.pi*R**3)
    kernel=TubeKernel(a,R,mu,p.wall_theta,p.wall_rings,p.wall_half_length_radii)
    t=float(min(0.,birth[0])) if N else 0.;end=p.start_s+p.duration_s
    thermal=None;thermal_values=np.full((N,3),np.nan)
    if config.cell.use_brownian:
        from .thermal_path import ThermalPath
        thermal=ThermalPath(t,end,seed)
    history_t=[];history_x=[];history_state=[];history_shape=[];next_save=t;next_in=0;steps=0
    counters=dict(pair_evaluations=0,pair_contacts=0,wall_contacts=0,captured=0,
        projection_max_um=0.,max_overlap_um=0.,max_shape_strain=0.,max_induced_um_s=0.,
        max_lubrication_dissipation_w=0.,min_step_s=p.dt_s,max_cluster_cells=1)
    contact_pairs=set();rng=np.random.default_rng(seed)
    def save():
        if (len(history_t)+1)*max(N,1)>6000000:raise ValueError("History-storage limit exceeded. Increase the output interval or shorten the observation window.")
        if history_t and abs(history_t[-1]-t)<1e-12:
            history_x[-1]=xyz.copy();history_state[-1]=state.copy();history_shape[-1]=np.linalg.norm(Q,axis=(1,2));return
        history_t.append(t);history_x.append(xyz.copy());history_state.append(state.copy());history_shape.append(np.linalg.norm(Q,axis=(1,2)))
    while t<end-1e-10:
        if steps>=p.max_steps:raise ValueError("Population step limit reached. Adjust the time step or observation duration.")
        while next_in<N and birth[next_in]<=t+1e-10:
            state[next_in]=5;next_in+=1
        # Prescribed Poisson times are source attempts. If a finite sphere's
        # assigned inlet slot is occupied, retain it in a geometric waiting
        # state instead of overlapping cells or enforcing a temporal gap.
        # Different lateral slots may admit simultaneously; this is NOT a
        # reservoir-flow or universal single-file queue model.
        for j in np.flatnonzero(state==5):
            resident=xyz[(state==1)|(state==3)]
            occupied=(p.inlet_queue and len(resident) and np.any(np.linalg.norm(resident-initial[j],axis=1)<2*R+1e-12))
            if not occupied:
                xyz[j]=initial[j];state[j]=1;cross[j,0]=t
                if thermal is not None:thermal_values[j]=thermal.value(ids[j:j+1],t)[0]
        live=np.flatnonzero((state==1)|(state==3))
        if len(live):
            fixed=state[live]==3
            x=xyz[live];r=np.linalg.norm(x[:,:2],axis=1);si=np.minimum(np.searchsorted(boundaries,x[:,2],side='right'),len(route)-1)
            grad=np.zeros((len(x),3,3));grad[:,2,0]=-4*mean*x[:,0]/a**2;grad[:,2,1]=-4*mean*x[:,1]/a**2
            E=.5*(grad+grad.swapaxes(-1,-2))
            ambient=np.zeros_like(x);ambient[:,2]=2*mean*(1-r*r/a**2)
            if p.faxen:ambient[:,2]-=4*mean*R**2/(3*a*a)
            force=weight*gravity[si]
            blocks=self_mobility(x,R,a,mu,rough)
            velocity=ambient+np.einsum('nij,nj->ni',blocks,force)
            if config.supply.saffman_enabled:
                shear=-4*mean*r/a**2;slip=velocity[:,2]-ambient[:,2]
                lift=saffman_lift(f.sample_density,mu,R,-slip,shear)
                normal=np.c_[x[:,:2]/np.maximum(r[:,None],1e-30),np.zeros(len(x))]
                force+=normal*lift[:,None]*np.clip((a-r-R)/(2*R),0,1)[:,None]
                velocity=ambient+np.einsum('nij,nj->ni',blocks,force)
            if coupling or p.membrane or p.inertia:
                groups=_components(x,p.neighbor_diameters*2*R) if coupling else [np.array([i]) for i in range(len(x))]
                for group in groups:
                    if len(group)==1 and not p.membrane and not p.inertia:continue
                    if len(group)>40:raise ValueError("Neighbour cluster exceeds 40 cells, outside the dilute confined approximation.")
                    velocity[group],detail=solve_group(x[group],ambient[group],E[group],force[group],kernel,rough,
                        coupling=coupling and p.hydrodynamics,wall=p.wall_reflections,
                        lubrication=coupling and p.pair_lubrication,
                        deformation=Q[live[group]] if p.membrane else None,
                        inertia=p.inertia,umean=mean,rho=f.sample_density,fixed=fixed[group])
                    counters['pair_evaluations']+=int(len(group)>1)
                    counters['max_cluster_cells']=max(counters['max_cluster_cells'],len(group))
                    counters['max_induced_um_s']=max(counters['max_induced_um_s'],detail['max_induced_m_s']*1e6)
                    counters['max_lubrication_dissipation_w']=max(counters['max_lubrication_dissipation_w'],detail['lubrication_dissipation_w'])
            dt=min(p.dt_s,end-t)
            if t<p.start_s<t+dt:dt=p.start_s-t
            if np.any(state==5):dt=min(dt,.1*R/max(2*mean,1e-30))
            if next_in<N:dt=min(dt,max(birth[next_in]-t,1e-10))
            # Relative displacement, rather than fast common advection,
            # determines collision accuracy. Check axial neighbours even when
            # outside the hydrodynamic cutoff to avoid skipping an encounter.
            order=np.argsort(x[:,2]);ia,ib=order[:-1],order[1:]
            delta=x[ib]-x[ia];d=np.linalg.norm(delta,axis=1)
            closing=-np.einsum('ij,ij->i',velocity[ib]-velocity[ia],delta)/np.maximum(d,1e-30)
            approach=(closing>1e-15)&(np.linalg.norm(delta[:,:2],axis=1)<2*R+rough)
            if approach.any():dt=min(dt,float(np.min(.2*np.maximum(d[approach]-2*R,rough)/closing[approach])))
            radial_speed=np.linalg.norm(velocity[:,:2],axis=1)
            if radial_speed.max()>0:dt=min(dt,.2*R/radial_speed.max())
            if config.cell.use_brownian:
                D=float(brownian_diffusivity(f.temperature_k,mu,R));dt=min(dt,R*R/(40*D))
            if dt<1e-9:raise ValueError("Population time step is too small. Check the near-field parameters.")
            old=x.copy();new=x+velocity*dt
            new[fixed]=old[fixed]
            if config.cell.use_brownian:
                moving=live[~fixed];next_thermal=thermal.value(ids[moving],t+dt)
                new[~fixed]+=(next_thermal-thermal_values[moving])*np.sqrt(2*D)
                thermal_values[moving]=next_thermal
            if p.membrane:
                Q[live]=membrane_step(Q[live],grad,dt,mu*R/p.membrane_shear_modulus_n_m)
                strain=np.linalg.norm(Q[live],axis=(1,2));counters['max_shape_strain']=max(counters['max_shape_strain'],float(strain.max()))
                if strain.max()>p.max_shape_strain:raise ValueError("Deformation exceeds the small-strain range. Increase membrane modulus or use a resolved membrane model.")
            # Finite exclusion is distinct from adhesion and from a prescribed
            # axial minimum gap. Side-by-side passage is allowed by geometry.
            if coupling and p.exclusion:
                for iteration in range(12):
                    pairs=cKDTree(new).query_pairs(2*R,output_type='ndarray')
                    if not len(pairs):break
                    residual=0.
                    for i,j in pairs:
                        delta=new[j]-new[i];distance=np.linalg.norm(delta);overlap=2*R-distance
                        if overlap<=0:continue
                        direction=delta/max(distance,1e-30)
                        if distance<1e-25:direction=np.array([0.,0.,1.])
                        if fixed[i] and fixed[j]:raise ValueError("Bound cells overlap. Check the contact parameters.")
                        correction=(overlap+1e-12)*direction
                        if fixed[i]:new[j]+=correction
                        elif fixed[j]:new[i]-=correction
                        else:new[i]-=.5*correction;new[j]+=.5*correction
                        counters['projection_max_um']=max(counters['projection_max_um'],float(overlap*1e6))
                        key=tuple(sorted((int(ids[live[i]]),int(ids[live[j]]))))
                        if key not in contact_pairs:contact_pairs.add(key);counters['pair_contacts']+=1
                        residual=max(residual,overlap)
                    radius=np.linalg.norm(new[:,:2],axis=1);new[:,:2]*=np.minimum(1.,(a-R-rough)/np.maximum(radius,1e-30))[:,None]
                    if residual<1e-11:break
                for i,j in cKDTree(new).query_pairs(2*R,output_type='ndarray'):
                    counters['max_overlap_um']=max(counters['max_overlap_um'],float(max(0.,2*R-np.linalg.norm(new[i]-new[j]))*1e6))
                if counters['max_overlap_um']>.01:raise ValueError("Cell-contact constraints did not converge. Reduce the time step.")
            radius=np.linalg.norm(new[:,:2],axis=1);touch=radius>a-R-rough
            counters['wall_contacts']+=int(touch.sum())
            new[:,:2]*=np.minimum(1.,(a-R-rough)/np.maximum(radius,1e-30))[:,None]
            if config.supply.wall_capture_rate_s:
                capture=~fixed&(radius>=a-R-rough*1.05)&(rng.random(len(x))<-np.expm1(-config.supply.wall_capture_rate_s*dt))
                state[live[capture]]=3;counters['captured']+=int(capture.sum())
            for k,plane in enumerate(planes[1:],1):
                hit=~fixed&(old[:,2]<plane)&(new[:,2]>=plane)&np.isnan(cross[live,k])
                if hit.any():cross[live[hit],k]=t+dt*(plane-old[hit,2])/np.maximum(new[hit,2]-old[hit,2],1e-30)
            done=~fixed&(new[:,2]>=length)
            # Retain actual interface coordinates and time; the downstream
            # path is explicitly awaiting a new tip solution.
            new[done,2]=length;state[live[done]]=2
            reverse=new[:,2]<-1e-9;state[live[reverse]]=4
            xyz[live]=new;t+=dt
            counters['min_step_s']=min(counters['min_step_s'],dt)
        else:
            t=min(end,birth[next_in] if next_in<N else end)
        if t>=next_save-1e-10 or t>=end-1e-10 or abs(t-p.start_s)<1e-10:
            save();next_save=t+(p.prehistory_interval_s if t<p.start_s else p.sample_interval_s)
        if steps%1000==0:progress((' Coupled ' if coupling else ' Independent reference ')+f' · t={t:.3f} s / {end:g} s · {len(live)}  Cells in capillary ')
        steps+=1
    save()
    history_x=np.asarray(history_x);history_x[:,:,2]+=start_z
    counters.update(steps=steps,wall_collocation_residual_m_s=kernel.max_wall_residual,
        wall_kernel_evaluations=kernel.evaluations,created=next_in,
        queued=int(np.sum(state==5)),admitted=int(np.sum(np.isfinite(cross[:,0]))),
        mobile=int(np.sum(state==1)),captured=int(np.sum(state==3)),crossed_bulk_end=int(np.sum(state==2)),reverse_censored=int(np.sum(state==4)))
    counters['brownian_path_resolution_s']=None if thermal is None else thermal.resolution_s
    counters['wall_contact_statistic']='wall_contacts counts cell-step wall clamps, not unique cells or independent collision events'
    counters['mass_balance_error']=next_in-sum(counters[k] for k in ('mobile','captured','crossed_bulk_end','reverse_censored','queued'))
    return np.asarray(history_t),history_x,np.asarray(history_state),np.asarray(history_shape),cross,counters


def simulate(scene,parameters=None,progress=lambda s:None):
    from ..scene3d.data import _clean
    p=deepcopy(parameters or CollectiveParameters()).validate();cfg=scene.data.config
    if p.inertia and cfg.supply.wall_capture_rate_s:
        raise ValueError("First-order inertia excludes convection of fixed-body reaction forces. Set wall capture to zero or disable the inertial correction.")
    if p.start_s+p.duration_s>scene.model.parameters.duration_s:raise ValueError("The study window exceeds the source duration.")
    r=scene.model.records;keep=np.asarray(r['injection_s'])<p.start_s+p.duration_s
    ids=np.asarray(r['event_id'])[keep];birth=np.asarray(r['injection_s'])[keep]
    if len(ids)>p.max_events:raise ValueError(f' Source events including warm-up  {len(ids)},  Exceeds limit  {p.max_events};  Adjust concentration / Study settings ,  All cell events retained ')
    seed=cfg.numerics.seed+1703 if p.seed is None else p.seed
    initial=initial_positions(cfg,ids,seed)
    if len(initial):
        limit=cfg.geometry.a-cfg.cell.diameter_um*.5e-6-p.roughness_um*1e-6
        rr=np.linalg.norm(initial[:,:2],axis=1);initial[:,:2]*=np.minimum(1.,limit/np.maximum(rr,1e-30))[:,None]
    time,x,state,shape,cross,diag=_advance(cfg,ids,birth,initial,p,progress,True)
    baseline_x=baseline_state=baseline_cross=None;baseline_diag=None
    if p.compare_independent:
        bt,bx,bs,_,baseline_cross,baseline_diag=_advance(cfg,ids,birth,initial,p,progress,False)
        # Sample adaptive histories on a common clock, respecting admission and exit states.
        baseline_x=np.full_like(x,np.nan);baseline_state=np.zeros_like(state)
        for j in range(len(ids)):
            baseline_state[:,j]=bs[np.clip(np.searchsorted(bt,time,side='right')-1,0,len(bt)-1),j]
            good=np.isfinite(bx[:,j]).all(axis=1)
            if good.any():
                for axis in range(3):baseline_x[:,j,axis]=np.interp(time,bt[good],bx[good,j,axis],left=np.nan,right=bx[good,j,axis][-1])
    plane_rows=[]
    for k,fraction in enumerate(np.linspace(0,1,5)):
        actual=cross[:,k];mask=(actual>=p.start_s)&(actual<p.start_s+p.duration_s)
        row=dict(fraction=float(fraction),position_mm=float(fraction*cfg.geometry.bulk_length*1e3),coupled=interval_statistics(actual[mask],ids[mask]))
        if baseline_cross is not None:
            b=baseline_cross[:,k];bm=(b>=p.start_s)&(b<p.start_s+p.duration_s)
            row['independent']=interval_statistics(b[bm],ids[bm])
        plane_rows.append(row)
    a=cfg.geometry.a;R=cfg.cell.diameter_um*.5e-6;mu=cfg.fluid.sample_viscosity;u=cfg.fluid.qs/(np.pi*a*a)
    scopes=[
        dict(process=' Common-clock capillary population ',status='computed',detail=' All events advanced from the earliest saved source time ,  Includes warm-up ;  Finite warm-up ,  Assess steady-state convergence by extending warm-up .'),
        dict(process=' Finite-size inlet occupancy ',status='conditional' if p.inlet_queue else 'off',detail=' Occupied inlet positions queue until clear ;  Different inlet positions can admit simultaneously . Poisson times denote source attempts ,  Actual admission times recorded separately ;  Reservoir flow excluded .'),
        dict(process=' Cell — Cell — Wall hydrodynamics ',status='reduced' if p.hydrodynamics else 'off',detail=' Finite cylindrical-wall regularized Stokes Collocation ＋ Self-mobility correction ;  Reduced particle surface / Curved-capillary CFD.'),
        dict(process=' Pair lubrication and finite-size exclusion ',status='computed' if p.pair_lubrication and p.exclusion else 'partial',detail=' Normal squeeze resistance ＋ Geometric non-overlap ;  Sphere rotation omitted / Tangential lubrication .'),
        dict(process=' Membrane deformation ',status='conditional' if p.membrane else 'off',detail=' Optional small-strain Kelvin Shape tensor and stresslet feedback ;  Specified effective membrane modulus ,  Reduced Skalak/Helfrich Resolved membrane .'),
        dict(process=' Inertial correction ',status='conditional' if p.inertia else 'off',detail=' First-order linearized disturbance convection Born Volume integral ,  Restricted to Re≤0.5;  Reduced finite Re Wake and inertial-train effects .'),
        dict(process=' Curved capillary Dean Flow / Resolved membrane mechanics / Correlated thermal noise ',status='unresolved',detail=' Piecewise locally straight capillary ;  No prescribed ordering force .'),
        dict(process=' Outlet / MS continuation ',status='unresolved',detail=' Population domain ends at the resolved-tip entrance ;  Tip continuation requires the updated inlet states ,  Recompute continuation to assess MS Efficiency changes .')]
    diag.update(source_id=scene.source_id,source_rate_cells_s=scene.model.source_rate,
        prehistory_s=max(0.,-float(birth[0])) if len(birth) else 0.,
        reynolds=cfg.fluid.sample_density*u*2*a/mu,particle_reynolds=cfg.fluid.sample_density*u*2*a/mu*(R/a)**2,
        bulk_length_m=cfg.geometry.bulk_length,model='common_clock_reduced_confined_long_tube',
        baseline=baseline_diag,mechanisms=scopes,references=REFERENCES,
        notes=[' Computed axial spacing ,  Arrival intervals and single-source rates ;  Finite-size exclusion enforces 3D non-overlap .',
            ' Matched source IDs / Attempt times / Inlet positions ;  Inlet occupancy can delay admission ,  Input Poisson Output statistics computed from crossings .',
            ' Dilute background flow from the supply solution ;  Population-induced global pressure-flow feedback omitted .',
            ' Warm-up and observation-length convergence / Time step / Wall collocation / Cutoff convergence ;  Sparse events CV Insufficient evidence for axial ordering .',
            ' Independent free-space thermal noise ;  Correlated mobility noise and thermal drift omitted ,  Reduced confined FDT Solver .'])
    pairs=dict(coupled=pair_statistics(time,x,state,p.start_s,p.start_s+p.duration_s,2*R))
    if baseline_x is not None:pairs['independent']=pair_statistics(time,baseline_x,baseline_state,p.start_s,p.start_s+p.duration_s,2*R)
    analysis=dict(planes=plane_rows,pair_dynamics=pairs,actual_ms_detection_rate_s=None,
        interpretation=' Radial focusing ,  Spatial non-overlap ,  Stable pairing ,  Periodic trains and single-source acquisition are distinct metrics .CV Changes can reflect selection / Finite sampling ;  Assess pair dynamics across repeated windows .')
    return CollectiveResult(p,scene.source_id,ids,birth,time,x,state,shape,cross,baseline_x,baseline_state,baseline_cross,_clean(diag),_clean(analysis))
