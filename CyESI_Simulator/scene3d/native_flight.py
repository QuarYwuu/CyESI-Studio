"""Conditional charged lipid-carrier flight in three Cartesian dimensions.

This reuses the established shell/evaporation material closures, but advances
all three carrier velocity/position components in the native gas and electric
background. Carrier formation, charge allocation and cell-drop association are
assumptions. A frozen gas field plus suction is not a coupled inlet gas solve.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq

from ..cell_state.parameters import StateParameters
from ..cell_state.membrane import LipidShell
from ..cell_state.inlet import (estimate_inlet,GAS_R,GAMMA,gas_viscosity,
                               suction_velocity,gap_gas_temperature)
from ..cell_state.flight import (EPS0,SIGMA_SB,RHO_M,RHO_W,rayleigh_charge,
                                film_state,evaporation_rates,screened_field_step)


def selected_native_flight(scene,event_id,parameters=None):
    p=deepcopy(parameters or scene.state_parameters or StateParameters()).validate()
    m=scene.model;i=scene._validate_event(event_id);k=int(m.records['template_id'][i])
    if not m.exited[k]:raise ValueError('The selected native trajectory has not crossed the liquid lip')
    c=scene.data.config;g=c.geometry;volume=scene.volume
    middle=lambda a:a[len(a)//2]
    permeability=middle(p.methanol_permeabilities_m_s);dissolution=middle(p.dissolution_rates_s)
    radius0=middle(p.drop_radii_um)*1e-6;charge_fraction=middle(p.charge_fractions);factor=middle(p.flow_factors)
    shell=LipidShell(c.cell.diameter_um*.5e-6,p,permeability,dissolution)
    state=shell.initial();state.track_index=k;pre=[]
    times,xyz,_,_=scene._track_xyz(k);track=m.tracks[k]
    axes=volume.grid.centers;lower=np.array([a[0] for a in axes]);upper=np.array([a[-1] for a in axes])
    def sample(point,name):return np.asarray(volume.sample(np.clip(point,lower,upper),name)).reshape(-1)
    for left,right in zip(times[:-1],times[1:]):
        elapsed=float(left);ca=float(np.interp(left,times,track['C']));cb=float(np.interp(right,times,track['C']))
        maximum=.2 if max(ca,cb)<1e-10 else p.max_pre_step_ms*.001
        while elapsed<right-1e-13:
            dt=min(maximum,right-elapsed);mid=elapsed+dt*.5
            point=np.array([np.interp(mid,times,xyz[:,a]) for a in range(3)])
            composition=float(np.interp(mid,times,track['C']))
            if point[2]>=g.liquid_start:
                from .cell_charge import liquid_samples
                ef=np.asarray(liquid_samples(volume,point)[1]).reshape(-1)
            else:ef=np.zeros(3)
            conductivity=c.electrical.sheath_conductivity_s_m+1.5*(1-composition)
            shell.step(state,dt,composition,float(np.linalg.norm(ef)),conductivity)
            elapsed+=dt
            if not pre or elapsed-pre[-1]['time_s']>.003:
                pre.append(dict(time_s=elapsed,external_methanol_fraction=composition,**shell.metrics(state)))
            if not shell.metrics(state)['shell_continuum_valid']:break
        if not shell.metrics(state)['shell_continuum_valid']:break
    inlet=estimate_inlet(p);flow_l_min=inlet['baseline_ambient_flow_l_min']*factor
    result,history=_fly(scene,shell,state,xyz[-1],float(track['C'][-1]),radius0,
                        charge_fraction,flow_l_min,p,sample)
    return dict(event_id=i,trajectory_index=k,transport_sample_id=int(scene.data.ids[k]),
        parameters=p.to_dict(),scenario=dict(drop_radius_um=radius0*1e6,charge_fraction=charge_fraction,
        permeability_m_s=permeability,dissolution_rate_s=dissolution,flow_factor=factor),
        inlet=inlet,result=result,history=history,pre_history=pre,conditional=True,
        coordinate_mode='native_cartesian_conditional_carrier',notes=[
            'All three carrier coordinates and velocity components advance through native Cartesian gas/E fields.',
            'Carrier formation, radius, charge and the association with this cell remain scenario assumptions.',
            'Frozen nozzle gas plus approximate inlet suction does not solve their joint momentum balance.',
            'Rayleigh instability terminates prediction; daughter drops and cell allocation are not invented.',
            'Shell continuum validity is not biological viability; aperture entry is not mass-spectral identification.'])


def _fly(scene,shell,initial_state,lip,initial_c,radius0,charge_fraction,flow_l_min,p,sample):
    c=scene.data.config;g=c.geometry;state=initial_state.clone();rc=shell.radius
    core=4/3*np.pi*rc**3;outer=4/3*np.pi*(radius0**3-rc**3)
    if outer<=0:raise ValueError('Conditional carrier radius must exceed the cell radius')
    mm=outer*initial_c*RHO_M;mw=outer*(1-initial_c)*RHO_W
    gamma=.072*(1-initial_c)+c.fluid.surface_tension*initial_c
    q=charge_fraction*rayleigh_charge(radius0,gamma)*np.sign(c.electrical.voltage_kv)
    all_rate=(c.fluid.qs+c.fluid.qh)/(4/3*np.pi*radius0**3)
    current=q*all_rate
    tip=abs(float(sample(np.r_[lip[:2],-1e-10],'potential')[0]))
    common=dict(initial_drop_radius_um=radius0*1e6,charge_fraction=charge_fraction,
        inlet_flow_l_min=flow_l_min,inferred_spray_current_na=current*1e9,
        estimated_tip_voltage_v=tip,all_drop_rate_s=all_rate,carrier_charge_c=q,
        coating_initial_methanol_fraction=initial_c)
    common.update(electrical_field_scope='Native 3-D electric field retained without rescaling; the selected carrier does not represent a monodisperse population of every spray droplet',
        imposed_or_estimated_spray_current_na=(c.electrical.spray_current_na or 0.)*np.sign(c.electrical.voltage_kv),
        hypothetical_monodisperse_current_na=current*1e9)
    if not shell.metrics(state)['shell_continuum_valid']:
        return dict(common,status='pre_nozzle_shell_model_lost',arrived=False,entry_state_resolved=False,**shell.metrics(state)),[]
    # Carrier size/charge is a conditional selected-origin case. Its assumed
    # monodisperse all-drop current must not silently change the shared field.
    gap=g.ms_gap_mm*.001;a=p.inlet_diameter_mm*.5e-3;flow=flow_l_min/60000
    point=np.asarray(lip,float).copy();point[2]=0.
    radial=point[:2]/max(np.linalg.norm(point[:2]),1e-30)
    if np.linalg.norm(radial)<.5:radial=np.array([1.,0.])
    point[:2]+=radial*p.alignment_offset_um*1e-6
    velocity=sample(np.r_[lip[:2],-1e-10],'velocity')
    velocity[:2]+=radial*velocity[2]*np.tan(np.deg2rad(p.initial_angle_deg))
    temp=c.fluid.temperature_k;t=0.;em=0.;ew=0.;max_error=0.;outer_tracer=0.
    initial_tracer=float(state.tracer@shell.transport.volumes)
    initial_m=mm+RHO_M*float(state.methanol@shell.transport.volumes)
    initial_w=mw+RHO_W*(core-float(state.methanol@shell.transport.volumes))
    electric=lambda x:sample(x,'electric')
    previous=electric(point);polarization=previous.copy()
    gas=scene.volume.material==0
    nearest=distance_transform_edt(~gas,return_distances=False,return_indices=True)
    gas_sampler=RegularGridInterpolator(scene.volume.grid.centers,scene.volume.velocity[tuple(nearest)],bounds_error=False,fill_value=0.)
    status='flight_timeout';history=[];last=-np.inf;max_re=0.;max_ratio=0.
    def row():
        _,r,co=film_state(rc,mm,mw)
        return dict(time_after_lip_s=t,x_m=float(point[0]),y_m=float(point[1]),z_m=float(point[2]),
            r_m=float(np.linalg.norm(point[:2])),drop_radius_m=r,temperature_k=temp,
            external_methanol_fraction=co,**shell.metrics(state))
    history.append(row())
    while t<p.max_flight_time_ms*.001:
        outer,radius,cout=film_state(rc,mm,mw);gamma=.072*(1-cout)+c.fluid.surface_tension*cout
        if abs(q)>=rayleigh_charge(radius,gamma):status='rayleigh_instability_unresolved';break
        if radius-rc<.1e-6:status='coating_depleted_unresolved';break
        if state.membrane_fraction<p.minimum_membrane_fraction:status='membrane_continuum_lost';break
        if state.max_pore_area_fraction>p.maximum_pore_area_fraction:status='pore_area_model_out_of_range';break
        r=float(np.linalg.norm(point[:2]));radial=point[:2]/max(r,1e-30)
        su=np.asarray(suction_velocity(r,point[2],gap,a,flow))
        ug=np.r_[radial*su[0],su[1]]
        if p.use_saved_nozzle_gas:
            background=np.asarray(gas_sampler(point)).reshape(3)
            if np.isfinite(background).all():
                max_ratio=max(max_ratio,float(np.linalg.norm(background)/max(np.linalg.norm(ug),1e-20)));ug+=background
        tg=gap_gas_temperature(point[2],gap,a,flow,p);rho=p.ambient_pressure_pa/(GAS_R*tg);mu=gas_viscosity(tg)
        speed=float(np.linalg.norm(ug-velocity));re=2*radius*rho*speed/mu;max_re=max(max_re,re)
        if re>1000:status='drag_model_out_of_range';break
        mean_c=float(state.methanol@shell.transport.volumes)/core
        mass=mm+mw+c.cell.density*core+(RHO_M-RHO_W)*mean_c*core
        drag=6*np.pi*mu*radius*(1+.15*re**.687);relax=mass/drag
        dt=min(p.max_flight_step_us*1e-6,.15*relax,p.max_flight_time_ms*.001-t)
        if gap-point[2]<1e-9:
            point[2]=gap;status='entered_aperture' if r+radius<=a else 'impacted_inlet_face';break
        try:dm=evaporation_rates(mm,mw,radius,temp,tg,speed,p) if p.evaporation else np.zeros(2)
        except ValueError:status='evaporation_model_out_of_range';break
        for amount,rate in zip([mm,mw],dm):
            if rate<0 and amount>1e-24:dt=min(dt,.03*amount/(-rate))
        if dt<1e-12:status='integration_or_component_depletion_unresolved';break
        target=ug+q*electric(point)*relax/mass
        def advance(h):
            decay=np.exp(-h/relax)
            return target*h+(velocity-target)*relax*(1-decay),target+(velocity-target)*decay
        delta,next_velocity=advance(dt)
        if delta[2]>=gap-point[2]:
            dt=brentq(lambda h:advance(h)[0][2]-(gap-point[2]),0.,dt,xtol=1e-15)
            delta,next_velocity=advance(dt)
        conductivity=c.electrical.sheath_conductivity_s_m+(1-cout)*1.5
        new_e=electric(point+delta)
        polarization,internal=screened_field_step(polarization,previous,new_e,dt,
            EPS0*c.chemistry.liquid_relative_permittivity,conductivity)
        if not p.electrical_screening:internal=new_e
        previous=new_e
        cout_new,tracer_new,transferred,_=shell.step(state,dt,cout,float(np.linalg.norm(internal)),conductivity,
            outer_tracer/max(outer,1e-300),outer)
        mm-=transferred*RHO_M;mw+=transferred*RHO_W;outer_tracer=tracer_new*outer
        mm+=dm[0]*dt;mw+=dm[1]*dt
        if mm < -1e-24 or mw < -1e-24:status='negative_solvent_inventory';break
        mm=max(0.,mm);mw=max(0.,mw);em-=dm[0]*dt;ew-=dm[1]*dt
        cp=mm*2530+mw*4180+max(mass-mm-mw,0)*4180
        pr=GAMMA*GAS_R/(GAMMA-1)*mu/.026;nu=2+.6*np.sqrt(re)*pr**(1/3)
        heat=2*np.pi*radius*nu*.026*(tg-temp)
        dist=max(gap-point[2],0.);hot=p.hot_face_radius_mm*.001
        view=.5*(dist/np.sqrt(dist**2+a*a)-dist/np.sqrt(dist**2+hot**2))
        heat+=4*np.pi*radius**2*p.effective_emissivity*SIGMA_SB*view*((p.inlet_wall_temperature_c+273.15)**4-temp**4)
        heat+=1.17e6*dm[0]+2.44e6*dm[1];temp+=dt*heat/max(cp,1e-30)
        point+=delta;velocity=next_velocity;t+=dt
        cm=float(state.methanol@shell.transport.volumes)
        max_error=max(max_error,abs(mm+RHO_M*cm+em-initial_m)/max(initial_m,1e-30),
            abs(mw+RHO_W*(core-cm)+ew-initial_w)/max(initial_w,1e-30),
            abs(float(state.tracer@shell.transport.volumes)+outer_tracer-initial_tracer)/max(initial_tracer,1e-30))
        if t-last>=1e-5:history.append(row());last=t
        if point[2]>=gap:
            status='entered_aperture' if np.linalg.norm(point[:2])+radius<=a else 'impacted_inlet_face';break
        if point[2]<-.1e-3 or np.linalg.norm(point[:2])>a*.5:status='outside_paraxial_flow_validity';break
    final=row()
    if abs(history[-1]['time_after_lip_s']-t)<1e-12:history[-1]=final
    else:history.append(final)
    _,radius,cout=film_state(rc,mm,mw);arrived=status=='entered_aperture'
    return dict(common,status=status,arrived=arrived,entry_state_resolved=arrived,flight_time_ms=t*1e3,
        final_x_um=point[0]*1e6,final_y_um=point[1]*1e6,final_r_um=np.linalg.norm(point[:2])*1e6,
        final_z_mm=point[2]*1e3,final_drop_radius_um=radius*1e6,final_temperature_c=temp-273.15,
        final_coating_methanol_fraction=cout,coating_thickness_um=(radius-rc)*1e6,
        tracer_released_before_lip_fraction=1-initial_tracer/core,
        tracer_in_carrier_fraction=outer_tracer/core,
        tracer_carried_total_fraction=(outer_tracer+float(state.tracer@shell.transport.volumes))/core,
        max_relative_species_balance_error=max_error,max_droplet_reynolds=max_re,
        max_nozzle_background_to_suction_ratio=max_ratio,**shell.metrics(state)),history
