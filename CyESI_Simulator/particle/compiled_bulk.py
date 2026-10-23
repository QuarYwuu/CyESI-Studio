"""Compiled deterministic segmented-tube paths, sharing the reference equations."""
import math
import os
import numpy as np
from numba import njit, prange, set_num_threads, config as numba_config


@njit(cache=True,parallel=True)
def _integrate(initial,angle,ids,boundaries,gravity,values,interval,slots,lubrication,saffman):
    R,a,mean,buoyancy,mu,rho,dt,timeout,length,gap_min,power=values
    count=len(initial);x=initial*np.cos(angle);y=initial*np.sin(angle)
    time=np.zeros(count);distance=np.zeros(count);state=np.zeros(count,np.int8);vz=np.zeros(count)
    dose=np.zeros(count);damage=np.zeros(count);peak=np.zeros(count);wall=np.zeros(count);steps=np.zeros(count,np.int64)
    lookup=np.full(count,-1,np.int64)
    for j in range(len(ids)):lookup[ids[j]]=j
    history=np.zeros((slots,len(ids),4));zeta=6*math.pi*mu*R
    for i in prange(count):
        xx=x[i];yy=y[i];ss=0.;tt=0.;k=0;slot=1;j=lookup[i]
        initial_gap=max(a-initial[i]-R,gap_min)
        initial_parallel=1.
        if lubrication:initial_parallel=1+8/15*math.log(max(2*R/max(initial_gap,R*1e-4),1.))
        vz[i]=2*mean*(1-(initial[i]/a)**2)+buoyancy*gravity[0,2]/initial_parallel
        if j>=0:history[0,j,2]=xx;history[0,j,3]=yy
        while ss<length-1e-12 and state[i]==0:
            r=math.hypot(xx,yy);nx=xx/max(r,1e-30);ny=yy/max(r,1e-30)
            segment=min(np.searchsorted(boundaries,ss,side='right'),len(boundaries)-1)
            gx,gy,gz=gravity[segment]
            gap=max(a-r-R,gap_min);normal=1.;parallel=1.
            if lubrication:
                h=max(gap,R*1e-4);normal=1+max(R/h-.5,0.)
                parallel=1+8/15*math.log(max(2*R/h,1.))
            u=2*mean*(1-(r/a)**2);slip=buoyancy*gz/parallel;shear=-4*mean*r/a**2
            lift=0.
            if saffman:
                sign=1. if shear>0 else -1. if shear<0 else 0.
                lift=6.46*rho*R**2*math.sqrt(mu/rho*abs(shear))*(-slip)*sign/zeta*min(max(gap/(2*R),0.),1.)
            ex=buoyancy*gx;ey=buoyancy*gy;projection=ex*nx+ey*ny
            vx=ex/parallel+projection*nx*(1/normal-1/parallel)+lift*nx/normal
            vy=ey/parallel+projection*ny*(1/normal-1/parallel)+lift*ny/normal
            speed=u+slip;vz[i]=speed
            remaining=min(length,boundaries[segment])-ss
            local=min(dt,remaining/max(speed,1e-30),max(0.,timeout-tt))
            if math.hypot(vx,vy)<1e-20 and abs(slip)<1e-20:
                local=min(remaining/max(speed,1e-30),max(0.,timeout-tt))
            if speed<=0:state[i]=3;local=0.
            stress=mu*abs(shear);dose[i]+=stress*local;damage[i]+=stress**power*local;peak[i]=max(peak[i],stress)
            if gap<=gap_min*1.05:wall[i]+=local
            xx+=vx*local;yy+=vy*local;ss+=speed*local
            if abs(ss-boundaries[segment])<1e-12:ss=boundaries[segment]
            scale=min(1.,(a-R-gap_min)/max(math.hypot(xx,yy),1e-30));xx*=scale;yy*=scale;tt+=local
            if tt>=timeout-1e-10 or ss < -1e-9:state[i]=3
            if k%interval==0 and j>=0:
                history[slot,j,0]=tt;history[slot,j,1]=ss;history[slot,j,2]=xx;history[slot,j,3]=yy;slot+=1
            k+=1
        x[i]=xx;y[i]=yy;time[i]=tt;distance[i]=ss;steps[i]=k
        if j>=0:
            for z in range(slot,slots):history[z,j,0]=tt;history[z,j,1]=ss;history[z,j,2]=xx;history[z,j,3]=yy
    used=1+((np.max(steps)-1)//interval+1 if np.max(steps)>0 else 0)
    return x,y,time,distance,state,vz,dose,damage,peak,wall,history[:used]


def simulate(config,radius0,rng):
    from .long_tube import BulkResult
    from ..fluid.supply import segments
    g,f,c,n,s=config.geometry,config.fluid,config.cell,config.numerics,config.supply
    R=c.diameter_um*.5e-6;mean=f.qs/(np.pi*g.a**2)
    h=s.wall_capture_gap_um*1e-6
    if h>=g.a-R:raise ValueError('Capture/roughness gap exceeds available tube clearance')
    buoyancy=(c.density-f.sample_density)*4/3*np.pi*R**3/(6*np.pi*f.sample_viscosity*R)
    dt=min(n.bulk_dt_s,.1*R/max(abs(buoyancy)*9.80665,1e-30))
    interval=max(1,int(min(n.bulk_timeout_s,max(1.,g.bulk_length/max(mean,1e-20))*4)/dt/180))
    route=segments(config);boundaries=np.array([x['end_m'] for x in route]);gravity=np.array([[x[k] for k in ('gx','gy','gz')] for x in route])
    ids=np.linspace(0,len(radius0)-1,min(n.saved_trajectories,len(radius0)),dtype=int)
    values=np.array([R,g.a,mean,buoyancy,f.sample_viscosity,f.sample_density,dt,n.bulk_timeout_s,g.bulk_length,h,s.shear_damage_exponent])
    max_radius=max(g.a-R-h,float(np.max(radius0)))
    min_speed=2*mean*(1-(max_radius/g.a)**2)-abs(buoyancy)*9.80665
    bound=min(n.bulk_timeout_s,g.bulk_length/min_speed+2*dt) if min_speed>0 else n.bulk_timeout_s
    slots=3+(int(np.ceil(bound/dt))+len(route)+3)//interval
    threads=n.workers or min(8,max(1,(os.cpu_count() or 1)//2-1))
    set_num_threads(min(threads,numba_config.NUMBA_NUM_THREADS,len(radius0)))
    x,y,time,distance,status,vz,dose,damage,peak,wall,history=_integrate(radius0,rng.uniform(0,2*np.pi,len(radius0)),ids,
        boundaries,gravity,values,interval,slots,s.lubrication_enabled,s.saffman_enabled)
    # Reference appends a terminal frame even when it duplicates a saved step.
    terminal=np.stack((time[ids],distance[ids],x[ids],y[ids]),axis=-1)[None,...]
    history=np.concatenate((history,terminal),axis=0)
    tracks={'ids':ids,'time_s':history[:,:,0],'distance_m':history[:,:,1],'x_m':history[:,:,2],'y_m':history[:,:,3]}
    stress={'shear_dose_pa_s':dose,'damage_exposure':damage,'peak_equivalent_stress_pa':peak,
            'near_wall_time_s':wall,'bulk_time_s':time.copy(),'bulk_status':status.copy()}
    return BulkResult(x,y,time,status,vz,stress,tracks,distance)
