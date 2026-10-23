"""Shared continuous finite-sphere physics over solved local tip fields.

Cartesian and axisymmetric field adapters use the same interaction, contact
and bond laws. An axisymmetric field never makes each cell a particle ring.

This optional study admits the existing Poisson event IDs at their actual tip
arrival times. It never treats independently timed response templates as one
simultaneous suspension. Background fields remain one-way coupled. Pair flow
uses a cutoff RPY external-force mobility plus the leading rigid-sphere strain
stresslet; this is a dilute, unbounded-fluid approximation with self wall
mobility, not a confined many-body Stokes solver. Finite exclusion/contact is
separate from unmeasured, explicitly conditional adhesion kinetics.

References: Wajnryb et al., JFM 731 (2013), R3,
https://arxiv.org/abs/1307.7312 ; Bell, Science 200 (1978), 618,
https://doi.org/10.1126/science.347575 . Bell-type force acceleration below is
a phenomenological aggregate bond, not a measured quartz/lymphocyte constant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from ..electro.electrostatics import EPS0
from ..particle.forces import (brownian_diffusivity, lubrication_factors,
    electrophoretic_mobility, dielectrophoretic_force)


STATE_LABELS = {0: 'not_in_window', 1: 'mobile', 2: 'cell_bound',
    3: 'wall_bound', 4: 'outlet_bound', 5: 'exited', 6: 'censored'}
LIVE_STATES = (1, 2, 3, 4)


@dataclass
class InteractionParameters:
    start_s: float = 0.
    duration_s: float = 1.
    dt_s: float = .001
    sample_interval_s: float = .005
    contact_gap_um: float = .1
    neighbor_cutoff_radii: float = 8.
    hydrodynamics: bool = True
    cell_adhesion_on_rate_s: float = 0.
    wall_adhesion_on_rate_s: float = 0.
    outlet_adhesion_on_rate_s: float = 0.
    adhesion_off_rate_s: float = 0.
    adhesion_force_scale_pn: float = 100.
    max_events: int = 2000
    max_steps: int = 200000
    seed: int | None = None

    def validate(self):
        values=[self.start_s,self.duration_s,self.dt_s,self.sample_interval_s,
            self.contact_gap_um,self.neighbor_cutoff_radii,self.cell_adhesion_on_rate_s,
            self.wall_adhesion_on_rate_s,self.outlet_adhesion_on_rate_s,
            self.adhesion_off_rate_s,self.adhesion_force_scale_pn]
        if not np.isfinite(values).all():raise ValueError('Interaction parameters must be finite')
        if not isinstance(self.hydrodynamics,(bool,np.bool_)):raise ValueError('hydrodynamics must be boolean')
        for name in ('max_events','max_steps'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
                raise ValueError(f'{name} must be an integer')
        if self.start_s<0 or not 0<self.duration_s<=60:raise ValueError('Interaction window must start at >=0 s and last 0--60 s')
        if not 1e-7<=self.dt_s<=.05:raise ValueError('Interaction dt must be 1e-7--0.05 s')
        if not 1e-5<=self.sample_interval_s<=1:raise ValueError('Interaction sample interval must be 1e-5--1 s')
        if self.contact_gap_um<0 or self.neighbor_cutoff_radii<2:raise ValueError('Invalid contact gap or neighbour cutoff')
        if min(values[6:10])<0 or self.adhesion_force_scale_pn<=0:raise ValueError('Adhesion rates must be nonnegative and force scale positive')
        if not 1<=self.max_events<=10000 or not 1<=self.max_steps<=2000000:raise ValueError('Invalid interaction resource limits')
        if self.seed is not None and (isinstance(self.seed,(bool,np.bool_)) or not isinstance(self.seed,(int,np.integer)) or self.seed<0):raise ValueError('Interaction seed must be a nonnegative integer')
        if (int(np.ceil(self.duration_s/self.sample_interval_s))+2)*self.max_events>8000000:
            raise ValueError('Interaction history is too large; increase sample interval or reduce max events')
        return self

    def to_dict(self):return asdict(self)


def adhesion_probability(rate_s, dt_s, force_n=0., force_scale_n=1.):
    """Exact frozen-rate Poisson hazard, optionally Bell slip-bond accelerated."""
    if rate_s==0:return np.zeros(np.broadcast_shapes(np.shape(dt_s),np.shape(force_n)))
    exponent=np.minimum(np.maximum(np.asarray(force_n),0.)/force_scale_n,700.)
    with np.errstate(over='ignore'):
        hazard=rate_s*np.exp(exponent)*np.asarray(dt_s)
    return -np.expm1(-hazard)


def _annulus_distance(xyz, r0, r1, lip):
    """Signed distance and fluid normal to a finite-radius quartz lip/side."""
    xyz=np.asarray(xyz,float);r=np.hypot(xyz[:,0],xyz[:,1]);z=xyz[:,2]
    dr=r-np.clip(r,r0,r1);dz=np.maximum(z-lip,0.)
    d=np.hypot(dr,dz);nr=dr/np.maximum(d,1e-30);nz=dz/np.maximum(d,1e-30)
    inside=(r>=r0)&(r<=r1)&(z<=lip)
    choices=np.stack([r-r0,r1-r,lip-z],axis=1)
    side=np.argmin(choices,axis=1)
    d=np.where(inside,-np.min(choices,axis=1),d)
    nr=np.where(inside,np.where(side==0,-1.,np.where(side==1,1.,0.)),nr)
    nz=np.where(inside,(side==2).astype(float),nz)
    radial=np.c_[xyz[:,0]/np.maximum(r,1e-30),xyz[:,1]/np.maximum(r,1e-30)]
    radial[r<1e-25]=[1.,0.]
    return d,np.c_[radial*nr[:,None],nz]


def wall_geometry(xyz, geometry, radius_m, contact_gap_m=0.):
    """Quartz distance, fluid normal and outlet classification, including corners.

    The inner and sheath quartz tubes are annular solids ending at their actual
    lips. A cell within one radius of either annular end is an outlet contact;
    the open centre is never treated as a solid flat cap.
    """
    xyz=np.asarray(xyz,float);g=geometry
    di,ni=_annulus_distance(xyz,g.a,g.b,g.inner_tip)
    do,no=_annulus_distance(xyz,g.c,g.d,0.)
    choose=do<di;distance=np.where(choose,do,di)
    normal=np.where(choose[:,None],no,ni)
    lip=np.where(choose,0.,g.inner_tip)
    outlet=np.abs(xyz[:,2]-lip)<=radius_m+contact_gap_m
    return distance,normal,outlet


def pair_disturbance(xyz, external_force_n, strain_rate_s, viscosity_pa_s,
                     radius_m, cutoff_m):
    """Dilute RPY force mobility + leading force-free strain stresslet.

    Electrophoretic drift is not a net external force and must not be passed as
    a Stokeslet source. Near-contact and confining wall reflections are omitted;
    hard exclusion is applied separately by the integrator.
    """
    xyz=np.asarray(xyz,float);answer=np.zeros_like(xyz)
    if len(xyz)<2:return answer,np.empty((0,2),int)
    pairs=cKDTree(xyz).query_pairs(cutoff_m,output_type='ndarray')
    if not len(pairs):return answer,pairs
    i,j=pairs.T;rr=xyz[i]-xyz[j];d=np.linalg.norm(rr,axis=1)
    n=rr/np.maximum(d[:,None],1e-30)
    coincident=d<1e-25;n[coincident]=[1.,0.,0.]
    d=np.maximum(d,2*radius_m)
    mu=.5*(np.asarray(viscosity_pa_s)[i]+np.asarray(viscosity_pa_s)[j])
    a=1+2*radius_m**2/(3*d**2);b=1-2*radius_m**2/d**2
    prefactor=1/(8*np.pi*mu*d)
    def rpy(force):return prefactor[:,None]*(a[:,None]*force+b[:,None]*np.sum(force*n,axis=1)[:,None]*n)
    # u'=-(5/2)a^3 (n.E.n)n/r^2, with opposite n at the second sphere.
    ej=np.einsum('ni,nij,nj->n',n,strain_rate_s[j],n)
    ei=np.einsum('ni,nij,nj->n',n,strain_rate_s[i],n)
    stress=2.5*radius_m**3/d**2
    np.add.at(answer,i,rpy(external_force_n[j])-stress[:,None]*ej[:,None]*n)
    np.add.at(answer,j,rpy(external_force_n[i])+stress[:,None]*ei[:,None]*n)
    return answer,pairs


@dataclass
class InteractionResult:
    parameters: InteractionParameters
    diagnostics: dict
    event_ids: np.ndarray
    time_s: np.ndarray
    positions_m: np.ndarray
    states: np.ndarray
    cluster_ids: np.ndarray
    contact_codes: np.ndarray
    methanol_fraction: np.ndarray
    admitted_s: np.ndarray
    exit_s: np.ndarray
    events: list

    def frame(self,time_s):
        if not self.time_s[0]-1e-12<=time_s<=self.time_s[-1]+1e-12:return []
        k=int(np.clip(np.searchsorted(self.time_s,time_s,side='right')-1,0,len(self.time_s)-1))
        q=min(k+1,len(self.time_s)-1);dt=self.time_s[q]-self.time_s[k]
        alpha=0. if dt==0 else float(np.clip((time_s-self.time_s[k])/dt,0,1))
        xyz=self.positions_m[k].copy()
        both=np.isfinite(self.positions_m[q]).all(axis=1)&np.isin(self.states[q],LIVE_STATES)
        xyz[both]=(1-alpha)*xyz[both]+alpha*self.positions_m[q,both]
        return [dict(event_id=int(self.event_ids[j]),local_xyz_m=xyz[j].tolist(),
            status=STATE_LABELS[int(self.states[k,j])],state=int(self.states[k,j]),
            cluster_id=int(self.cluster_ids[k,j]),contact_code=int(self.contact_codes[k,j]),
            methanol_fraction=float(self.methanol_fraction[k,j]))
            for j in np.flatnonzero(np.isin(self.states[k],LIVE_STATES))]

    def event_history(self,event_id):
        found=np.flatnonzero(self.event_ids==int(event_id))
        if not len(found):return None
        j=int(found[0]);keep=self.states[:,j]!=0
        return dict(event_id=int(event_id),time_s=self.time_s[keep].tolist(),
            local_xyz_m=self.positions_m[keep,j].tolist(),states=self.states[keep,j].tolist(),
            statuses=[STATE_LABELS[int(s)] for s in self.states[keep,j]],
            cluster_ids=self.cluster_ids[keep,j].tolist(),contact_codes=self.contact_codes[keep,j].tolist(),
            methanol_fraction=self.methanol_fraction[keep,j].tolist(),
            admitted_s=float(self.admitted_s[j]),exit_s=float(self.exit_s[j]) if np.isfinite(self.exit_s[j]) else None,
            events=[e for e in self.events if e['event_id']==int(event_id) or e.get('other_event_id')==int(event_id)])

    def to_storage(self):
        arrays={key:np.asarray(getattr(self,key)) for key in ('event_ids','time_s','positions_m',
            'states','cluster_ids','contact_codes','methanol_fraction','admitted_s','exit_s')}
        return dict(parameters=self.parameters.to_dict(),diagnostics=deepcopy(self.diagnostics),
                    events=deepcopy(self.events),state_labels=STATE_LABELS),arrays

    @classmethod
    def from_storage(cls,metadata,arrays):
        return cls(InteractionParameters(**metadata['parameters']).validate(),
            deepcopy(metadata['diagnostics']),**{k:np.asarray(arrays[k]).copy() for k in
            ('event_ids','time_s','positions_m','states','cluster_ids','contact_codes',
             'methanol_fraction','admitted_s','exit_s')},events=deepcopy(metadata.get('events',[])))


class _TipField:
    def __init__(self,scene):
        v=scene.volume;cfg=scene.data.config;self.cfg=cfg
        self.axes=v.grid.centers;self.lower=np.array([a[0] for a in self.axes]);self.upper=np.array([a[-1] for a in self.axes])
        self.inlet=float(v.grid.edges[2][0]);self.min_grid=min(float(np.diff(e).min()) for e in v.grid.edges)
        liquid=v.material==1
        if not liquid.any():raise ValueError('No liquid cells in native volume')
        nearest=distance_transform_edt(~liquid,return_distances=False,return_indices=True)
        def fill(a):return np.asarray(a)[tuple(nearest)]
        u=fill(v.velocity);E=fill(v.electric);C=np.clip(fill(v.methanol),0,1)
        grad=np.stack([np.stack(np.gradient(u[...,k],*self.axes,edge_order=1),axis=-1) for k in range(3)],axis=-2)
        strain=.5*(grad+np.swapaxes(grad,-1,-2))
        strain-=np.trace(strain,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)/3
        ge=np.stack(np.gradient(np.sum(E*E,axis=-1),*self.axes,edge_order=1),axis=-1)
        omega=np.stack([grad[...,2,1]-grad[...,1,2],grad[...,0,2]-grad[...,2,0],grad[...,1,0]-grad[...,0,1]],axis=-1)
        channels=np.concatenate([u,E,C[...,None],ge,strain.reshape(*strain.shape[:3],9),omega],axis=-1)
        self.joint=RegularGridInterpolator(self.axes,channels,bounds_error=False,fill_value=np.nan)
        self.gravity=np.zeros(3)
        if cfg.supply.gravity_enabled:
            if scene.path is not None:
                _,T,Bn,Bb=scene.path.sample(scene.path.length_m);gg=np.array([0.,-9.80665,0.])
                self.gravity=np.array([gg@Bn,gg@Bb,gg@T])
            else:
                from ..fluid.supply import segments
                self.gravity=np.array([segments(cfg)[-1][key] for key in ('gx','gy','gz')])

    def sample(self,xyz,radius):
        cfg=self.cfg;f,c=cfg.fluid,cfg.cell;a=self.joint(np.clip(xyz,self.lower,self.upper))
        C=np.clip(a[:,6],0,1);mu=np.exp((1-C)*np.log(f.sample_viscosity)+C*np.log(f.sheath_viscosity))
        rho=f.sample_density*(1-C)+f.sheath_density*C;drag=6*np.pi*mu*radius
        external=(c.density-rho)[:,None]*(4/3*np.pi*radius**3)*self.gravity
        eps=EPS0*cfg.chemistry.liquid_relative_permittivity
        external+=dielectrophoretic_force(eps,radius,c.clausius_mossotti,a[:,7:10])
        extra=external/drag[:,None]+electrophoretic_mobility(eps,c.zeta_mv*.001,mu)[:,None]*a[:,3:6]
        distance,normal,outlet=wall_geometry(xyz,cfg.geometry,radius)
        normal_drag,parallel=lubrication_factors(radius,distance-radius)
        if not cfg.supply.lubrication_enabled:normal_drag=np.ones(len(xyz));parallel=normal_drag.copy()
        A=parallel[:,None,None]*np.eye(3)+(normal_drag-parallel)[:,None,None]*normal[:,:,None]*normal[:,None,:]
        rhs=parallel[:,None]*a[:,:3]+(1-parallel)[:,None]*np.sum(a[:,:3]*normal,axis=1)[:,None]*normal+extra
        if cfg.supply.saffman_enabled:
            w=a[:,19:22];wn=np.linalg.norm(w,axis=1)
            factor=6.46*rho*radius**2*np.sqrt(mu/rho)/drag*np.clip((distance-radius)/(2*radius),0,1)/np.sqrt(np.maximum(wn,1e-30))
            B=np.zeros_like(A);B[:,0,1]=w[:,2];B[:,0,2]=-w[:,1];B[:,1,0]=-w[:,2]
            B[:,1,2]=w[:,0];B[:,2,0]=w[:,1];B[:,2,1]=-w[:,0]
            B*=factor[:,None,None];A+=B;rhs+=np.einsum('nij,nj->ni',B,a[:,:3])
        velocity=np.linalg.solve(A,rhs[...,None])[...,0]
        return dict(velocity=velocity,external=external,strain=a[:,10:19].reshape(-1,3,3),
            mu=mu,drag=drag,C=C,distance=distance,normal=normal,outlet=outlet,
            normal_drag=normal_drag,parallel_drag=parallel)


def _population(scene,inlet,start,end,max_events):
    """Use actual stream records, including residents supplied before t=0."""
    r=scene.model.records;templates=[]
    for k in range(len(scene.model.tracks)):
        age,xyz,_,_=scene._track_xyz(k);age=np.asarray(age,float);xyz=np.asarray(xyz,float)
        keep=np.isfinite(age)&np.isfinite(xyz).all(axis=1);age,xyz=age[keep],xyz[keep]
        if not len(age) or xyz[-1,2]<inlet-1e-12:templates.append(None);continue
        hit=np.flatnonzero(xyz[:,2]>=inlet-1e-12)
        if not len(hit):templates.append(None);continue
        j=int(hit[0])
        ta=float(age[j] if j==0 else np.interp(inlet,xyz[j-1:j+1,2],age[j-1:j+1]))
        templates.append((age,xyz,ta,float(age[-1])))
    event_ids=[];entry=[];position=[]
    for j,event_id in enumerate(r['event_id']):
        info=templates[int(r['template_id'][j])]
        if info is None:continue
        age,xyz,ta,tend=info;injection=float(r['injection_s'][j]);arrival=injection+ta
        if arrival>end+1e-12 or injection+tend<start-1e-12:continue
        at=max(start,arrival)-injection
        p=np.array([np.interp(at,age,xyz[:,d]) for d in range(3)])
        if p[2]>=-1e-12:continue
        if len(event_ids)>=max_events:raise ValueError(f'Continuous interaction window has more than {max_events} events; shorten the window or raise max_events. No cells were subsampled.')
        event_ids.append(int(event_id));entry.append(arrival);position.append(p)
    return np.asarray(event_ids,int),np.asarray(entry,float),np.asarray(position,float).reshape(-1,3)


def _clusters(n,bonds,event_ids):
    parent=np.arange(n)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for i,j in bonds:
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    return np.array([event_ids[find(j)] for j in range(n)],int)


def simulate_interactions(scene,parameters=None,progress=lambda value:None,field_provider=None):
    """Observe an unbounded continuous feed in a finite local tip time window.

    Initial residents use saved prehistory; new entrants retain their real event
    IDs/times. All interacting particles subsequently share the same clock.
    Upstream transport and global theoretical single-cell rates are unchanged.
    """
    p=deepcopy(parameters or InteractionParameters()).validate()
    if field_provider is None and not hasattr(scene,'volume'):raise ValueError('Interactions require a resolved background field provider')
    if p.start_s+p.duration_s>scene.model.parameters.duration_s+1e-10:raise ValueError('Interaction window exceeds the continuous source observation duration')
    field=_TipField(scene) if field_provider is None else field_provider
    cfg=scene.data.config;R=cfg.cell.diameter_um*.5e-6
    gap=p.contact_gap_um*1e-6
    if gap>=cfg.geometry.a-R:raise ValueError('Contact gap exceeds available sample-tube clearance')
    ids,entry,initial=_population(scene,field.inlet,p.start_s,p.start_s+p.duration_s,p.max_events)
    N=len(ids);rng=np.random.default_rng(cfg.numerics.seed+703 if p.seed is None else p.seed)
    xyz=np.full((N,3),np.nan);state=np.zeros(N,np.int8);exit_s=np.full(N,np.nan)
    anchor=np.zeros((N,3));wall_kind=np.zeros(N,np.int8);contacts=np.zeros(N,np.uint8)
    C=np.full(N,np.nan);bonds={};events=[]
    hist_t=[];hist_xyz=[];hist_state=[];hist_clusters=[];hist_contacts=[];hist_C=[]
    counters=dict(pair_contacts=0,wall_contacts=0,outlet_contacts=0,pair_bonds_formed=0,
        wall_bonds_formed=0,outlet_bonds_formed=0,bonds_released=0,initial_overlap_pairs=0)
    pair_seen=set();wall_seen=set();outlet_seen=set();max_disturbance=0.;max_overlap=0.;max_penetration=0.
    t=p.start_s;end=t+p.duration_s;next_sample=t;steps=0
    def event(kind,i,j=None,**extra):
        row=dict(time_s=float(t),kind=kind,event_id=int(ids[i]),**extra)
        if j is not None:row['other_event_id']=int(ids[j])
        events.append(row)
    def save():
        # Admissions and bond transitions add frames beyond the regular sample
        # interval. Bound the actual history, rather than its nominal estimate.
        if (len(hist_t)+1)*max(N,1)>8000000:
            raise ValueError('Actual interaction history exceeds the memory budget; shorten the window or reduce its event population. No physical cells were subsampled.')
        # One composition sample per saved frame, rather than re-evaluating all
        # 22 interpolated force channels twice on every integration step.
        present=np.flatnonzero(state!=0)
        if len(present):C[present]=field.sample(xyz[present],R)['C']
        if hist_t and abs(hist_t[-1]-t)<1e-12:
            hist_xyz[-1]=xyz.copy();hist_state[-1]=state.copy();hist_contacts[-1]=contacts.copy();hist_C[-1]=C.copy();hist_clusters[-1]=_clusters(N,bonds,ids)
        else:
            hist_t.append(float(t));hist_xyz.append(xyz.copy());hist_state.append(state.copy())
            hist_contacts.append(contacts.copy());hist_C.append(C.copy());hist_clusters.append(_clusters(N,bonds,ids))
    def admit():
        new=np.flatnonzero((state==0)&(entry<=t+1e-12))
        if len(new):
            xyz[new]=initial[new];state[new]=1
            # Response templates are independent, so their superposition can
            # begin in overlap. Admit with explicit excluded-volume adjustment;
            # this is recorded rather than mistaken for predicted aggregation.
            live=np.flatnonzero(np.isin(state,LIVE_STATES))
            for _ in range(8):
                pairs=cKDTree(xyz[live]).query_pairs(2*R,output_type='ndarray')
                for aa,bb in pairs:
                    i,j=int(live[aa]),int(live[bb]);delta=xyz[i]-xyz[j];d=np.linalg.norm(delta)
                    if d>=2*R:continue
                    key=(min(i,j),max(i,j))
                    if key not in pair_seen:
                        pair_seen.add(key);counters['pair_contacts']+=1;counters['initial_overlap_pairs']+=1
                    n=delta/max(d,1e-30)
                    if d<1e-25:n=np.array([1.,0.,0.])
                    wi=float(wall_kind[i]==0);wj=float(wall_kind[j]==0)
                    if wi+wj:xyz[i]+=wi/(wi+wj)*(2*R-d)*n;xyz[j]-=wj/(wi+wj)*(2*R-d)*n
                dist,norm,_=wall_geometry(xyz[live],cfg.geometry,R,gap)
                penetration=(dist<R)&(wall_kind[live]==0)
                xyz[live[penetration]]+=(R-dist[penetration])[:,None]*norm[penetration]
            C[live]=field.sample(xyz[live],R)['C']
        return len(new)>0
    admit()
    if N:
        live=np.flatnonzero(np.isin(state,LIVE_STATES))
        if len(live):C[live]=field.sample(xyz[live],R)['C']
    save();next_sample=min(end,t+p.sample_interval_s)
    while t<end-1e-12 and steps<p.max_steps:
        new=admit()
        if new:save()
        live=np.flatnonzero(np.isin(state,LIVE_STATES));future=entry[state==0]
        stop=min(end,next_sample,float(future.min()) if len(future) else end)
        if not len(live):
            t=stop
            if t>=next_sample-1e-12:save();next_sample=min(end,next_sample+p.sample_interval_s)
            steps+=1;continue
        sample=field.sample(xyz[live],R);velocity=sample['velocity'].copy();C[live]=sample['C']
        valid=np.isfinite(velocity).all(axis=1)
        if not valid.all():state[live[~valid]]=6;live=live[valid]
        if not len(live):save();continue
        if not valid.all():sample={k:v[valid] for k,v in sample.items()};velocity=velocity[valid]
        local=np.full(N,-1,int);local[live]=np.arange(len(live))
        fixed=wall_kind[live]>0
        pair_force=sample['external'].copy()
        pair_force[fixed]-=sample['drag'][fixed,None]*velocity[fixed]
        if p.hydrodynamics:
            disturbance,pairs=pair_disturbance(xyz[live],pair_force,sample['strain'],sample['mu'],R,max(2*R+gap,p.neighbor_cutoff_radii*R))
            normal=sample['normal'];vn=np.sum(disturbance*normal,axis=1)
            disturbance=disturbance/sample['parallel_drag'][:,None]+vn[:,None]*normal*(1/sample['normal_drag']-1/sample['parallel_drag'])[:,None]
            velocity+=disturbance
            max_disturbance=max(max_disturbance,float(np.linalg.norm(disturbance,axis=1).max(initial=0)))
        else:pairs=cKDTree(xyz[live]).query_pairs(max(2*R+gap,p.neighbor_cutoff_radii*R),output_type='ndarray')
        speed=np.linalg.norm(velocity,axis=1).max(initial=0.)
        # Absolute advection sees native grid, relative approach sees sphere size.
        dt=min(p.dt_s,stop-t,.4*field.min_grid/max(speed,1e-30))
        # Land on the start of a contact shell before integrating its hazard.
        # Otherwise an annular outlet touch shorter than configured dt could be
        # projected and immediately exit without any modeled contact dwell.
        wall_approach=-np.sum(velocity*sample['normal'],axis=1)
        remaining_gap=sample['distance']-R-gap
        approaching=(remaining_gap>1e-12)&(wall_approach>1e-20)&~fixed
        if approaching.any():dt=min(dt,float(np.min(remaining_gap[approaching]/wall_approach[approaching])))
        if len(pairs):
            rv=velocity[pairs[:,0]]-velocity[pairs[:,1]]
            dt=min(dt,.2*R/max(np.linalg.norm(rv,axis=1).max(initial=0.),1e-30))
            rr=xyz[live[pairs[:,0]]]-xyz[live[pairs[:,1]]]
            aa=np.sum(rv*rv,axis=1);bb=2*np.sum(rr*rv,axis=1)
            cc=np.sum(rr*rr,axis=1)-(2*R+gap)**2;disc=bb*bb-4*aa*cc
            crosses=(cc>(2*R+gap)*2e-12)&(bb<0)&(disc>=0)&(aa>1e-30)
            if crosses.any():
                hit=2*cc[crosses]/(-bb[crosses]+np.sqrt(disc[crosses]))
                dt=min(dt,float(hit.min()))
        if cfg.cell.use_brownian:
            D=brownian_diffusivity(cfg.fluid.temperature_k,sample['mu'],R)
            dt=min(dt,R*R/(20*D.max(initial=1e-30)))
        if dt<=1e-14:raise RuntimeError('Interaction step collapsed; refine geometry/contact settings')
        # Break existing aggregate tethers using their estimated opposing drag.
        changed=False
        for (i,j),rest in list(bonds.items()):
            li,lj=local[i],local[j]
            if li<0 or lj<0:del bonds[i,j];changed=True;continue
            n=xyz[i]-xyz[j];n/=max(np.linalg.norm(n),1e-30)
            tension=.5*(sample['drag'][li]+sample['drag'][lj])*.5*abs(np.dot(velocity[li]-velocity[lj],n))
            if rng.random()<adhesion_probability(p.adhesion_off_rate_s,dt,tension,p.adhesion_force_scale_pn*1e-12):
                del bonds[i,j];counters['bonds_released']+=1;event('cell_release',i,j);changed=True
        for li in np.flatnonzero(fixed):
            i=int(live[li]);load=sample['drag'][li]*np.linalg.norm(velocity[li])
            if rng.random()<adhesion_probability(p.adhesion_off_rate_s,dt,load,p.adhesion_force_scale_pn*1e-12):
                wall_kind[i]=0;counters['bonds_released']+=1;event('surface_release',i);changed=True
        # Cell contact is geometric. It becomes a bond only under an on-rate.
        contacts[:]=0
        for aa,bb in pairs:
            i,j=int(live[aa]),int(live[bb]);distance=float(np.linalg.norm(xyz[i]-xyz[j]))
            if distance<=2*R+gap+1e-12:
                contacts[[i,j]]|=1;key=(min(i,j),max(i,j))
                if key not in pair_seen:pair_seen.add(key);counters['pair_contacts']+=1
                if key not in bonds and rng.random()<adhesion_probability(p.cell_adhesion_on_rate_s,dt):
                    bonds[key]=max(2*R,distance);counters['pair_bonds_formed']+=1;event('cell_bind',i,j);changed=True
        near=sample['distance']<=R+gap+1e-12
        for li in np.flatnonzero(near):
            i=int(live[li]);outlet=bool(sample['outlet'][li]);contacts[i]|=4 if outlet else 2
            seen=outlet_seen if outlet else wall_seen
            if i not in seen:seen.add(i);counters['outlet_contacts' if outlet else 'wall_contacts']+=1
            rate=p.outlet_adhesion_on_rate_s if outlet else p.wall_adhesion_on_rate_s
            if wall_kind[i]==0 and rng.random()<adhesion_probability(rate,dt):
                wall_kind[i]=4 if outlet else 3;anchor[i]=xyz[i]
                counters['outlet_bonds_formed' if outlet else 'wall_bonds_formed']+=1
                event('outlet_bind' if outlet else 'wall_bind',i);changed=True
        # A distance-constrained aggregate can translate and rotate; a cell-cell
        # bond does not freeze the cluster. Surface-bound cells have fixed anchor.
        velocity[wall_kind[live]>0]=0.
        for _ in range(4):
            for i,j in bonds:
                li,lj=local[i],local[j]
                if li<0 or lj<0:continue
                n=xyz[i]-xyz[j];n/=max(np.linalg.norm(n),1e-30)
                rel=np.dot(velocity[li]-velocity[lj],n)
                wi=float(wall_kind[i]==0);wj=float(wall_kind[j]==0)
                if wi+wj:velocity[li]-=wi/(wi+wj)*rel*n;velocity[lj]+=wj/(wi+wj)*rel*n
        old=xyz[live].copy();xyz[live]+=velocity*dt
        if cfg.cell.use_brownian:
            noise=rng.normal(size=(len(live),3))*np.sqrt(2*D*dt)[:,None]
            noise[wall_kind[live]>0]=0.;xyz[live]+=noise
        # Nonpenetration and tether distances are projected together, with a
        # finite roughness/contact gap. Residual overlaps are reported explicitly.
        for _ in range(8):
            for i,j in bonds:
                dvec=xyz[i]-xyz[j];d=np.linalg.norm(dvec);n=dvec/max(d,1e-30)
                wi=float(wall_kind[i]==0);wj=float(wall_kind[j]==0)
                if wi+wj:correction=(d-bonds[i,j])*n;xyz[i]-=wi/(wi+wj)*correction;xyz[j]+=wj/(wi+wj)*correction
            near_pairs=cKDTree(xyz[live]).query_pairs(2*R,output_type='ndarray')
            for aa,bb in near_pairs:
                i,j=int(live[aa]),int(live[bb]);delta=xyz[i]-xyz[j];d=np.linalg.norm(delta)
                if d>=2*R:continue
                n=delta/max(d,1e-30)
                if d<1e-25:n=np.array([1.,0.,0.])
                wi=float(wall_kind[i]==0);wj=float(wall_kind[j]==0)
                if wi+wj:xyz[i]+=wi/(wi+wj)*(2*R-d)*n;xyz[j]-=wj/(wi+wj)*(2*R-d)*n
                contacts[[i,j]]|=1;key=(min(i,j),max(i,j))
                if key not in pair_seen:
                    pair_seen.add(key);counters['pair_contacts']+=1
                    if steps==0:counters['initial_overlap_pairs']+=1
            dist,norm,outlet=wall_geometry(xyz[live],cfg.geometry,R,gap)
            penetrated=(dist<R)&(wall_kind[live]==0)
            if penetrated.any():xyz[live[penetrated]]+=(R-dist[penetrated])[:,None]*norm[penetrated]
            xyz[live[wall_kind[live]>0]]=anchor[live[wall_kind[live]>0]]
        # Alternating projections can leave an infeasible tightly packed
        # configuration. Report the actual residual rather than claiming exact
        # hard-sphere/contact resolution from a fixed iteration count.
        check_pairs=cKDTree(xyz[live]).query_pairs(2*R,output_type='ndarray')
        if len(check_pairs):
            distances=np.linalg.norm(xyz[live[check_pairs[:,0]]]-xyz[live[check_pairs[:,1]]],axis=1)
            max_overlap=max(max_overlap,float(np.max(2*R-distances,initial=0.)))
        check_distance,_,_=wall_geometry(xyz[live],cfg.geometry,R,gap)
        max_penetration=max(max_penetration,float(np.max(R-check_distance,initial=0.)))
        t+=dt;steps+=1
        # A centre crossing is the defined middle-lip plane event. The carrier
        # and externally resolved droplets are outside this local rigid study.
        done=(xyz[live,2]>=0)&(wall_kind[live]==0)
        for li in np.flatnonzero(done):
            i=int(live[li]);fraction=np.clip(-old[li,2]/max(xyz[i,2]-old[li,2],1e-30),0,1)
            exit_s[i]=t-dt+fraction*dt;xyz[i]=old[li]+fraction*(xyz[i]-old[li]);xyz[i,2]=0.;state[i]=5
            for key in list(bonds):
                if i in key:del bonds[key];event('domain_exit_tether_truncated',i)
            changed=True
        reverse=(xyz[live,2]<field.inlet-1e-9)&~done
        state[live[reverse]]=6
        for i in live[reverse]:
            for key in list(bonds):
                if int(i) in key:del bonds[key]
        remaining=np.flatnonzero(np.isin(state,LIVE_STATES));state[remaining]=1
        for i,j in bonds:
            if state[i] in LIVE_STATES:state[i]=2
            if state[j] in LIVE_STATES:state[j]=2
        bound=np.flatnonzero((wall_kind>0)&np.isin(state,LIVE_STATES));state[bound]=wall_kind[bound]
        if changed or new or t>=next_sample-1e-12 or t>=end-1e-12:save()
        if t>=next_sample-1e-12:next_sample=min(end,next_sample+p.sample_interval_s)
        if steps%1000==0:progress(f'Continuous tip interactions: {t-p.start_s:.3f}/{p.duration_s:.3f} s, active {len(remaining)}, bonds {len(bonds)}')
    budget=steps>=p.max_steps and t<end-1e-12
    if budget:
        # These are unfinished numerical observations, never adhesive loss.
        state[np.isin(state,LIVE_STATES)]=6;save()
    else:admit();save()
    counts={label:int(np.sum(state==code)) for code,label in STATE_LABELS.items()}
    live_final=sum(counts[STATE_LABELS[s]] for s in LIVE_STATES)
    diag=dict(model='continuous_common_clock_local_tip',scope='native resolved tip only; initial resident prehistory and upstream independent transport are reused',
        start_s=p.start_s,requested_end_s=end,end_s=float(t),steps=steps,numerical_step_budget_exhausted=budget,
        event_count=N,initial_residents=int(np.sum(entry<p.start_s)),window_entries=int(np.sum(entry>=p.start_s)),
        admitted_count=int(np.sum(state!=0)),exited_count=counts['exited'],retained_at_end=live_final,
        state_counts=counts,mass_balance_residual=int(np.sum(state!=0)-counts['exited']-counts['censored']-live_final),
        max_pair_disturbance_m_s=max_disturbance,**counters,
        max_residual_pair_overlap_um=max_overlap*1e6,max_residual_wall_penetration_um=max_penetration*1e6,
        exclusion_residual_resolved=bool(max(max_overlap,max_penetration)<=R*1e-3),
        adhesion_calibration='conditional unmeasured aggregate on/off rates; default zero on-rates imply contact without adhesion',
        event_sampling='all actual Poisson source IDs in the window; no population subsampling',
        notes=['All interacting cells share one absolute integration clock; source arrivals remain continuous Poisson.',
            'Initial residents have independent mobile prehistory; adhesive deposits accumulated before this observation window are unknown, not initialized as a steady fouling layer.',
            'Unbounded-fluid cutoff RPY force mobility plus leading strain stresslet; no exact confined many-body flow or pair lubrication resistance.',
            'Electrophoretic drift is force-free; it is not inserted as a net-force Stokeslet. Buoyancy, DEP and fixed-anchor reaction generate force disturbances.',
            'Finite hard-sphere exclusion and tether constraints are numerical projections; deformability, rolling receptor dynamics and membrane fusion are unresolved.',
            'Independent prehistory can place initial residents or new entrants in overlap; explicit admission exclusion adjustments are counted, not interpreted as measured aggregation.',
            'Wall/outlet contact is separate from adhesion. Cells retained at the observation end are not counted as irreversible loss.',
            'Surface on/off rates and force scale require quartz/medium/cell calibration; zero off-rate is a conditional irreversible-bond scenario.',
            'Optional Brownian noise is independent dilute noise, not a correlated fluctuating-RPY Brownian solver.',
            'This local interaction realization does not modify global theoretical single-cell detection rates or imply full-pipe many-body validation.',
            'Tethers intersecting the local outlet boundary are truncated and recorded; downstream aggregate droplet flight is unresolved.'])
    concentration_m3=cfg.cell.concentration_per_ml*1e6
    volume_fraction=concentration_m3*(4/3*np.pi*R**3)
    diag['input_volume_fraction']=float(volume_fraction)
    diag['dilute_suspension_assumption']=bool(volume_fraction<.01)
    diag['mean_sample_axial_spacing_m']=None if concentration_m3==0 else float(1/(concentration_m3*np.pi*cfg.geometry.a**2))
    if volume_fraction>=.01:
        diag['notes'].append('Input volume fraction is at least one percent; unbounded dilute pair closure and one-way field coupling require scrutiny, not a validated dense-suspension prediction.')
    if field_provider is not None:
        diag['model']=getattr(field,'model_name','continuous_common_clock_adapted_tip')
        diag['scope']=getattr(field,'scope','Local field-adapted continuous study')
        diag['notes'].append(diag['scope'])
    return InteractionResult(p,diag,ids,np.asarray(hist_t),np.asarray(hist_xyz),np.asarray(hist_state),
        np.asarray(hist_clusters),np.asarray(hist_contacts),np.asarray(hist_C),entry,exit_s,events)
