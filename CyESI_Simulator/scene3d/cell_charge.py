"""Effective surface charge and induced membrane charge along actual 3-D paths.

These are separate observables: a slip-plane Debye--Huckel charge proxy, the
Smoluchowski electrophoretic force-equivalent charge, and an equal/opposite
capacitive polarization. They are diagnostics and never add another qE force
to particles that already include electrophoresis. Structural membrane charge
needs pH, acid/base sites and ion adsorption data and remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import hashlib,json,math
import numpy as np

EPS0=8.8541878128e-12
FARADAY=96485.33212
R_GAS=8.314462618
E_CHARGE=1.602176634e-19


@dataclass
class ChargeParameters:
    membrane_capacitance_f_m2: float=.01
    cytosol_conductivity_s_m: float=.5
    membrane_leak_conductance_s_m2: float=0.
    cation_diffusivity_m2_s: float=1.96e-9
    anion_diffusivity_m2_s: float=1.45e-9
    zeta_mv: float | None=None
    relative_permittivity: float | None=None
    temperature_k: float | None=None
    max_linear_reduced_zeta: float=.5
    max_linear_membrane_voltage_v: float=.25

    def validate(self):
        for key,value in vars(self).items():
            if value is None:continue
            if isinstance(value,bool) or not math.isfinite(float(value)):raise ValueError('Invalid charge parameter: '+key)
        if min(self.membrane_capacitance_f_m2,self.cytosol_conductivity_s_m,
               self.cation_diffusivity_m2_s,self.anion_diffusivity_m2_s,
               self.max_linear_reduced_zeta,self.max_linear_membrane_voltage_v)<=0:
            raise ValueError('Capacitance, conductivities, diffusivities and linear-model guards must be positive')
        if self.membrane_leak_conductance_s_m2<0:raise ValueError('Membrane leakage conductance must be nonnegative')
        if self.relative_permittivity is not None and self.relative_permittivity<=0:raise ValueError('Relative permittivity must be positive')
        if self.temperature_k is not None and self.temperature_k<=0:raise ValueError('Temperature must be positive')
        return self

    def to_dict(self):return asdict(self)


def charge_closures(radius,ionic_strength,field,parameters=None,zeta_mv=-20.,relative_permittivity=60.,temperature_k=298.15):
    """Vectorised local equilibrium targets and explicit proxy charges (SI)."""
    p=(parameters or ChargeParameters()).validate()
    zeta=(zeta_mv if p.zeta_mv is None else p.zeta_mv)*.001
    eps=EPS0*(relative_permittivity if p.relative_permittivity is None else p.relative_permittivity)
    temp=temperature_k if p.temperature_k is None else p.temperature_k
    I=np.maximum(np.asarray(ionic_strength,float),0.)
    kappa=np.sqrt(2*FARADAY**2*I/(eps*R_GAS*temp))
    debye=np.divide(1.,kappa,out=np.full_like(kappa,np.inf),where=kappa>0)
    # 1 mol/m^3 = 1 mM for the monovalent ammonium/formate salt.
    sigma_ion=FARADAY**2*(p.cation_diffusivity_m2_s+p.anion_diffusivity_m2_s)*I/(R_GAS*temp)
    q_zeta=4*np.pi*eps*radius*zeta*(1+kappa*radius)
    q_ep=np.full_like(I,6*np.pi*eps*radius*zeta)
    return dict(q_zeta_c=q_zeta,q_electrophoretic_effective_c=q_ep,debye_length_m=debye,
        ionic_conductivity_s_m=sigma_ion,electric_v_m=np.asarray(field,float),
        reduced_zeta=abs(zeta)*FARADAY/(R_GAS*temp),permittivity_f_m=eps,temperature_k=temp)


def relax_membrane(old,target_before,target_after,dt,tau):
    """Exact vector RC response to a linearly varying Schwan target."""
    if dt<=0:return np.asarray(old,float).copy()
    decay=np.exp(-dt/tau);one_minus=-np.expm1(-dt/tau)
    return np.asarray(target_after)+(np.asarray(old)-target_before)*decay-(np.asarray(target_after)-target_before)/dt*tau*one_minus


def liquid_samples(volume,points):
    """Interpolate liquid-side fields without mixing quartz/gas electric values."""
    from scipy.ndimage import distance_transform_edt
    from scipy.interpolate import RegularGridInterpolator
    if not hasattr(volume,'_charge_liquid_samplers'):
        liquid=volume.material==1
        nearest=distance_transform_edt(~liquid,return_distances=False,return_indices=True)
        volume._charge_liquid_samplers={name:RegularGridInterpolator(volume.grid.centers,
            getattr(volume,name)[tuple(nearest)],bounds_error=False,fill_value=np.nan)
            for name in ('salt_mol_m3','electric')}
    axes=volume.grid.centers
    clipped=np.clip(points,[a[0] for a in axes],[a[-1] for a in axes])
    return tuple(volume._charge_liquid_samplers[name](clipped) for name in ('salt_mol_m3','electric'))


def charge_at_track(scene,index,parameters=None):
    p=(parameters or getattr(scene,'charge_parameters',None) or ChargeParameters()).validate()
    index=int(index);cfg=scene.data.config
    cache=getattr(scene,'_charge_cache',None)
    if cache is None:cache={};scene._charge_cache=cache
    token=(scene.source_id,index,json.dumps(p.to_dict(),sort_keys=True))
    if token in cache:return cache[token]
    t,xyz,_,_=scene._track_xyz(index);tr=scene.model.tracks[index];N=len(t)
    C=np.asarray(tr['C'],float);I=np.full(N,cfg.chemistry.ammonium_formate_mm);field=np.zeros((N,3))
    native=xyz[:,2]>=cfg.geometry.liquid_start-1e-12
    if native.any():
        if hasattr(scene,'charge_samples'):
            salt,electric=scene.charge_samples(xyz[native])
        else:
            axes=scene.volume.grid.centers;lo=np.array([a[0] for a in axes]);hi=np.array([a[-1] for a in axes])
            points=np.clip(xyz[native],lo,hi)
            salt,electric=liquid_samples(scene.volume,points)
        I[native]=np.maximum(0.,np.asarray(salt).reshape(-1))
        field[native]=np.asarray(electric).reshape(-1,3)
    radius=cfg.cell.diameter_um*.5e-6
    local=charge_closures(radius,I,field,p,cfg.cell.zeta_mv,cfg.chemistry.liquid_relative_permittivity,cfg.fluid.temperature_k)
    sigma=np.maximum(local['ionic_conductivity_s_m']+cfg.electrical.sheath_conductivity_s_m,1e-15)
    impedance=1/p.cytosol_conductivity_s_m+1/(2*sigma)
    shunt=1+radius*p.membrane_leak_conductance_s_m2*impedance
    tau=radius*p.membrane_capacitance_f_m2*impedance/shunt
    target=1.5*radius*field/shunt[:,None];vm=np.zeros_like(field)
    for j in range(1,N):
        vm[j]=relax_membrane(vm[j-1],target[j-1],target[j],float(t[j]-t[j-1]),float((tau[j-1]+tau[j])*.5))
    magnitude=np.linalg.norm(vm,axis=1)
    direction=np.divide(vm,magnitude[:,None],out=np.zeros_like(vm),where=magnitude[:,None]>1e-30)
    global_direction=direction.copy()
    if scene.path is not None:
        stations=scene.path.length_m+xyz[:,2]-cfg.geometry.inner_tip
        _,T,A,B=scene.path.sample(stations)
        global_direction=direction[:,0,None]*A+direction[:,1,None]*B+direction[:,2,None]*T
    pole=np.pi*radius**2*p.membrane_capacitance_f_m2*magnitude
    value=dict(parameters=p.to_dict(),trajectory_index=index,transport_sample_id=int(scene.data.ids[index]),
        time_s=t,positions_local_m=xyz,methanol_fraction=C,ionic_strength_mol_m3=I,
        q_zeta_c=local['q_zeta_c'],q_electrophoretic_effective_c=local['q_electrophoretic_effective_c'],
        positive_pole_charge_c=pole,negative_pole_charge_c=-pole,induced_net_charge_c=np.zeros(N),
        membrane_pole_voltage_v=magnitude,membrane_voltage_vector_v=vm,target_voltage_vector_v=target,
        polarization_direction_local=direction,polarization_direction_global=global_direction,
        debye_length_m=local['debye_length_m'],medium_conductivity_s_m=sigma,
        membrane_charging_time_s=tau,electric_v_m=field,radius_m=radius,
        structural_membrane_net_charge_c=None,
        validity=dict(linear_dh_small_potential=local['reduced_zeta']<=p.max_linear_reduced_zeta,
            reduced_zeta=local['reduced_zeta'],max_linear_reduced_zeta=p.max_linear_reduced_zeta,
            membrane_voltage_within_linear_guard=magnitude<=p.max_linear_membrane_voltage_v,
            membrane_voltage_guard_v=p.max_linear_membrane_voltage_v),
        notes=[
            'q_zeta is a linear Debye-Huckel slip-plane equivalent charge, not measured structural membrane charge.',
            'q_electrophoretic_effective = 6*pi*epsilon*R*zeta is a force-equivalent diagnostic. Electrophoresis is already integrated; no additional qE force is applied.',
            'Positive/negative hemispheric capacitive charges integrate to zero induced net charge.',
            'Schwan RC response assumes a spherical membrane, uniform local field, constant capacitance and specified cytosol/leak conductivity.',
            'High salt, methanol-dependent dissociation/activity, permittivity, zeta changes, pH and membrane chemical charge regulation require calibration.',
            'Long-pipe electric field is assumed zero because upstream electrical boundary conditions are unavailable; it was not solved as a 3-D long-tube electrostatic domain.',
            'The membrane voltage guard flags a linear-model limitation; it is not a viability or electroporation probability.'])
    cache[token]=value
    return value


def charge_at_event(scene,event_id,time_s):
    """Current selected-origin state, with an analytic within-interval RC step."""
    i=scene._validate_event(event_id);m=scene.model;k=int(m.records['template_id'][i]);h=charge_at_track(scene,k)
    age=float(time_s-m.records['injection_s'][i]);t=h['time_s']
    if age<0:return None
    clipped=float(np.clip(age,t[0],t[-1]));j=int(np.clip(np.searchsorted(t,clipped,side='right')-1,0,len(t)-1))
    result=dict(event_id=i,age_s=age,state_age_s=clipped,extrapolated=age>t[-1]+1e-10,
        stage='last_computed_transport_state' if age>t[-1]+1e-10 else 'computed_transport_state',
        structural_membrane_net_charge_c=None,force_already_in_electrophoresis=True)
    for key in ('q_zeta_c','q_electrophoretic_effective_c','debye_length_m','medium_conductivity_s_m',
                'membrane_charging_time_s','methanol_fraction','ionic_strength_mol_m3'):
        result[key]=float(np.interp(clipped,t,h[key]))
    vm=h['membrane_voltage_vector_v'][j]
    if j<len(t)-1 and clipped>t[j]:
        target=np.array([np.interp(clipped,t,h['target_voltage_vector_v'][:,a]) for a in range(3)])
        tau=float(np.interp(clipped,t,h['membrane_charging_time_s']))
        vm=relax_membrane(vm,h['target_voltage_vector_v'][j],target,clipped-t[j],tau)
    magnitude=float(np.linalg.norm(vm));direction=vm/max(magnitude,1e-30)
    global_direction=direction.copy()
    if scene.path is not None:
        z=float(np.interp(clipped,t,h['positions_local_m'][:,2]))
        _,T,A,B=scene.path.sample(scene.path.length_m+z-scene.data.config.geometry.inner_tip)
        global_direction=direction[0]*A+direction[1]*B+direction[2]*T
    p=scene.charge_parameters
    pole=np.pi*h['radius_m']**2*p.membrane_capacitance_f_m2*magnitude
    result.update(positive_pole_charge_c=pole,negative_pole_charge_c=-pole,induced_net_charge_c=0.,
        membrane_pole_voltage_v=magnitude,polarization_direction_local=direction.tolist(),
        polarization_direction_global=global_direction.tolist(),
        q_zeta_elementary_charges=result['q_zeta_c']/E_CHARGE,
        q_electrophoretic_elementary_charges=result['q_electrophoretic_effective_c']/E_CHARGE,
        validity=dict(linear_dh_small_potential=h['validity']['linear_dh_small_potential'],
            membrane_voltage_within_linear_guard=magnitude<=p.max_linear_membrane_voltage_v,
            structural_charge_unknown=True,post_lip_not_extrapolated=not result['extrapolated']))
    return result
