"""Compiled deterministic sphere tracking with the reference force model.

Cells run in parallel at full precision. Brownian paths use the NumPy
integrator and its random stream.
"""
import math
import os
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from numba import config as numba_config


@njit(cache=True,inline='always')
def _locate(rr,zz,r,z):
    r=min(max(r,rr[0]),rr[-1]);z=min(max(z,zz[0]),zz[-1])
    i=min(max(np.searchsorted(rr,r,side='right')-1,0),len(rr)-2)
    j=min(max(np.searchsorted(zz,z,side='right')-1,0),len(zz)-2)
    x=(r-rr[i])/(rr[i+1]-rr[i]);y=(z-zz[j])/(zz[j+1]-zz[j])
    return i,j,x,y


@njit(cache=True,inline='always')
def _value(grid,i,j,x,y,channel):
    return (grid[i,j,channel]*(1-x)*(1-y)+grid[i,j+1,channel]*(1-x)*y
            +grid[i+1,j,channel]*x*(1-y)+grid[i+1,j+1,channel]*x*y)


@njit(cache=True)
def _sample(grid,rr,zz,r,z,channel):
    i,j,x,y=_locate(rr,zz,r,z)
    return _value(grid,i,j,x,y,channel)


@njit(cache=True)
def _wall(r,z,a,b,outer,lip):
    dr=r-min(max(r,a),b);dz=max(z-lip,0.)
    inner=math.hypot(dr,dz)
    if a<=r<=b and z<lip:inner=-min(r-a,b-r)
    norm=max(math.hypot(dr,dz),1e-30)
    if outer-r<inner:return outer-r,-1.,0.
    return inner,dr/norm,dz/norm


@njit(cache=True)
def _step(vr,vz,ur,uz,er,ez,nr,nz,normal,parallel,lift,tau,dt):
    difference=normal-parallel
    a=parallel+difference*nr*nr;c=difference*nr*nz;b=c+lift;d=parallel+difference*nz*nz
    un=ur*nr+uz*nz
    br=parallel*ur+(1-parallel)*un*nr+er+lift*uz
    bz=parallel*uz+(1-parallel)*un*nz+ez
    determinant=a*d-b*c
    target_r=(d*br-b*bz)/determinant;target_z=(a*bz-c*br)/determinant
    delta_r=vr-target_r;delta_z=vz-target_z
    trace=(a+d)/2;discriminant=((a-d)/2)**2+b*c;h=dt/tau
    if discriminant<0:
        omega=math.sqrt(max(-discriminant,1e-300));decay=math.exp(-trace*h)
        c0=decay*math.cos(omega*h);c1=-decay*math.sin(omega*h)/omega
    else:
        delta=math.sqrt(discriminant)
        plus=math.exp(-(trace+delta)*h);minus=math.exp(-(trace-delta)*h)
        c0=(plus+minus)/2
        c1=(plus-minus)/(2*delta) if delta>1e-10 else -h*math.exp(-trace*h)
    evr=c0*delta_r+c1*((a-trace)*delta_r+b*delta_z)
    evz=c0*delta_z+c1*(c*delta_r+(d-trace)*delta_z)
    dr=target_r*dt+tau*(d*(delta_r-evr)-b*(delta_z-evz))/determinant
    dz=target_z*dt+tau*(-c*(delta_r-evr)+a*(delta_z-evz))/determinant
    return dr,dz,target_r+evr,target_z+evz


@njit(cache=True,parallel=True)
def _integrate(r,z,vz,status,stress,ids,liquid,lr,lz,electric,er,ez,physical,max_steps,interval,lubrication,saffman):
    R,rho_c,eps,zeta_v,cm,a,b,outer,lip,start,gravity,dt,power,mu0,mu1,rho0,rho1,voltage=physical
    count=len(r);time=np.zeros(count);dose=np.zeros(count);exposure=np.zeros(count)
    outlet_r=np.full(count,np.nan);outlet_c=np.full(count,np.nan);residence_tip=np.full(count,np.nan)
    steps=np.zeros(count,np.int64);slip_re=np.zeros(count)
    nsaved=len(ids);slots=(max_steps-1)//interval+1
    history=np.zeros((slots,nsaved,4));lookup=np.full(count,-1,np.int64)
    for j in range(nsaved):lookup[ids[j]]=j
    volume=4/3*math.pi*R**3;mass=volume*rho_c
    for i in prange(count):
        rr=r[i];zz=z[i];vr=0.;vv=vz[i];elapsed=0.;integral=0.;C=0.;k=0;slot=0;hidx=lookup[i]
        while k<max_steps and status[i]==0:
            li,lj,lx,ly=_locate(lr,lz,rr,zz)
            C=min(max(_value(liquid,li,lj,lx,ly,0),0.),1.)
            mu=math.exp((1-C)*math.log(mu0)+C*math.log(mu1));rho=rho0*(1-C)+rho1*C
            ur=_value(liquid,li,lj,lx,ly,1);uz=_value(liquid,li,lj,lx,ly,2)
            shear=_value(liquid,li,lj,lx,ly,3)
            tau=max(0.,_value(liquid,li,lj,lx,ly,4));ext=abs(_value(liquid,li,lj,lx,ly,5))
            distance,nr,nz=_wall(rr,zz,a,b,outer,lip)
            normal=1.;parallel=1.
            if lubrication:
                gap=max(distance-R,R*1e-4)
                normal=1+max(R/gap-.5,0.);parallel=1+8/15*math.log(max(2*R/gap,1.))
            zeta=6*math.pi*mu*R
            lift=0.
            if saffman:
                sign=1. if shear>0 else -1. if shear<0 else 0.
                lift=6.46*rho*R**2*math.sqrt(mu/rho*abs(shear))*sign/zeta*min(max((distance-R)/(2*R),0.),1.)
            ei,ej,exx,eyy=_locate(er,ez,rr,zz)
            Er=_value(electric,ei,ej,exx,eyy,0)*voltage;Ez=_value(electric,ei,ej,exx,eyy,1)*voltage
            dep_r=2*math.pi*eps*R**3*cm*_value(electric,ei,ej,exx,eyy,2)*voltage**2
            dep_z=2*math.pi*eps*R**3*cm*_value(electric,ei,ej,exx,eyy,3)*voltage**2
            extra_r=eps*zeta_v/mu*Er+dep_r/zeta
            extra_z=eps*zeta_v/mu*Ez+(dep_z+(rho_c-rho)*volume*gravity)/zeta
            dr,dz,vr,vv=_step(vr,vv,ur,uz,extra_r,extra_z,nr,nz,normal,parallel,lift,mass/zeta,dt)
            old_z=zz;raw_r=rr+dr;rr=abs(raw_r)
            if raw_r<0:vr=-vr
            zz+=dz;elapsed+=dt;integral+=C*dt
            stress[i,0]+=tau*dt;stress[i,1]+=tau**power*dt;stress[i,2]=max(stress[i,2],tau);stress[i,3]+=ext*dt
            distance,_,_=_wall(rr,zz,a,b,outer,lip)
            hit=distance<R*(1-1e-6);reverse=zz<start-1e-9
            if hit:status[i]=2
            if reverse:status[i]=3  # matches reference assignment order
            if zz>=0 and not hit and not reverse:
                fraction=min(max(-old_z/max(dz,1e-30),0.),1.)
                elapsed-=dt*(1-fraction);integral-=C*dt*(1-fraction)
                stress[i,0]-=tau*dt*(1-fraction);stress[i,1]-=tau**power*dt*(1-fraction);stress[i,3]-=ext*dt*(1-fraction)
                zz=0.;outlet_r[i]=rr;residence_tip[i]=elapsed
                outlet_c[i]=_sample(liquid,lr,lz,rr,0.,0);status[i]=1
            slip_re[i]=max(slip_re[i],rho*2*R*math.hypot(ur-vr,uz-vv)/mu)
            if k%interval==0 and hidx>=0:
                history[slot,hidx,0]=rr;history[slot,hidx,1]=zz
                history[slot,hidx,2]=elapsed;history[slot,hidx,3]=C;slot+=1
            k+=1
        r[i]=rr;z[i]=zz;vz[i]=vv;time[i]=elapsed;dose[i]=integral;exposure[i]=C;steps[i]=k
        if status[i]==0:status[i]=3
        if hidx>=0:
            for j in range(slot,slots):
                history[j,hidx,0]=rr;history[j,hidx,1]=zz;history[j,hidx,2]=elapsed;history[j,hidx,3]=C
    used=(np.max(steps)-1)//interval+1 if np.max(steps)>0 else 0
    return time,dose,exposure,outlet_r,outlet_c,residence_tip,history[:used],np.max(slip_re)


def integrate(config,solvent,electric,r,z,vz,status,stress,ids,voltage_scale,save_every_steps=None):
    from ..chemistry.solvent_transport import filled
    from ..electro.electrostatics import EPS0
    from ..fluid.supply import segments
    g,f,c,n=config.geometry,config.fluid,config.cell,config.numerics
    channels=('concentration','velocity_r','velocity_z','shear','equivalent_stress_pa','axial_extension_stress_pa')
    liquid=np.ascontiguousarray(np.stack([filled(getattr(solvent,key),solvent.active) for key in channels],axis=-1))
    echannels=('field_r','field_z','grad_e2_r','grad_e2_z')
    if electric.ionic is not None:
        source=electric.ionic;rr=(source.radial_edges[:-1]+source.radial_edges[1:])/2;zz=(source.axial_edges[:-1]+source.axial_edges[1:])/2
        fields=np.stack([filled(getattr(source,key),source.active) for key in echannels],axis=-1)
    else:
        rr,zz=electric.r,electric.z
        fields=np.stack([getattr(electric,key) for key in echannels],axis=-1)
        if config.electrical.liquid_model=='conducting':fields=np.zeros_like(fields)
    physical=np.array([c.diameter_um*.5e-6,c.density,EPS0*config.chemistry.liquid_relative_permittivity,c.zeta_mv*.001,
        c.clausius_mossotti,g.a,g.b,g.c,g.inner_tip,g.liquid_start,segments(config)[-1]['gz'],n.particle_dt_s,
        config.supply.shear_damage_exponent,f.sample_viscosity,f.sheath_viscosity,f.sample_density,f.sheath_density,voltage_scale])
    names=('shear_dose_pa_s','damage_exposure','peak_equivalent_stress_pa','extension_dose_pa_s')
    matrix=np.ascontiguousarray(np.stack([stress[key] for key in names],axis=-1))
    target=n.workers if n.workers else min(8,max(1,(os.cpu_count() or 1)//2-1))
    set_num_threads(min(target,numba_config.NUMBA_NUM_THREADS,len(r)))
    max_steps=int(np.ceil(n.particle_tip_timeout_s/n.particle_dt_s))
    interval=max(1,max_steps//400) if save_every_steps is None else save_every_steps
    result=_integrate(r,z,vz,status,matrix,ids,liquid,solvent.r,solvent.z,np.ascontiguousarray(fields),rr,zz,
        physical,max_steps,interval,config.supply.lubrication_enabled,config.supply.saffman_enabled)
    for j,key in enumerate(names):stress[key][:]=matrix[:,j]
    return result
