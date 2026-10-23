"""Conditional charged carrier flight and binary-solvent evaporation.

No child droplets are invented: Rayleigh instability, loss of the coating or
loss of the membrane continuum terminates the corresponding prediction.
"""
import numpy as np
from scipy.optimize import brentq
from .inlet import GAS_R,GAMMA,gas_viscosity,suction_velocity,gap_gas_temperature

EPS0=8.8541878128e-12
R_GAS=8.314462618
SIGMA_SB=5.670374419e-8
RHO_M,RHO_W=792.,1000.


def rayleigh_charge(radius,surface_tension):
    return 8*np.pi*np.sqrt(EPS0*surface_tension*radius**3)


def saturation_pressures(temperature):
    """NIST Antoine in Pa; low-T methanol uses a labelled Clapeyron extension."""
    t=float(temperature)
    if not 256<=t<=373:raise ValueError("Droplet temperature outside the implemented evaporation-property range")
    if t>356.83:
        methanol=1e5*10**(5.15853-1569.613/(t-34.846))
    elif t>=288.1:
        methanol=1e5*10**(5.20409-1581.341/(t-33.50))
    else:
        anchor=1e5*10**(5.20409-1581.341/(288.1-33.50))
        methanol=anchor*np.exp(1.17e6*.0320419/R_GAS*(1/288.1-1/t))
    water=1e5*10**(4.6543-1435.264/(t-64.848))
    return methanol,water


def evaporation_rates(methanol_mass,water_mass,radius,temperature,gas_temperature,relative_speed,p):
    n=np.array([methanol_mass/.0320419,water_mass/.01801528]);x=n/max(n.sum(),1e-300)
    vapor=np.array(saturation_pressures(temperature))*x
    if vapor.sum()>=p.ambient_pressure_pa*.95:raise ValueError("Near-boiling multicomponent evaporation requires a Stefan-flow/boiling model")
    ambient_t=p.ambient_temperature_c+273.15
    ambient_water=p.relative_humidity*saturation_pressures(ambient_t)[1]
    rho_surface=vapor*np.array([.0320419,.01801528])/(R_GAS*temperature)
    rho_far=np.array([0.,ambient_water*.01801528/(R_GAS*ambient_t)])
    rho_gas=p.ambient_pressure_pa/(GAS_R*gas_temperature)
    viscosity=gas_viscosity(gas_temperature)
    diffusivities=np.array([1.55e-5,2.5e-5])*(gas_temperature/298.15)**1.75
    reynolds=2*radius*rho_gas*relative_speed/viscosity
    schmidt=viscosity/(rho_gas*diffusivities)
    sherwood=2+.6*np.sqrt(reynolds)*schmidt**(1/3)
    # Dilute Fick / Ranz-Marshall approximation, including possible water
    # condensation; vapor loading and mixture nonideality remain limitations.
    return -2*np.pi*radius*sherwood*diffusivities*(rho_surface-rho_far)


def film_state(core_radius,methanol_mass,water_mass):
    volume=methanol_mass/RHO_M+water_mass/RHO_W
    radius=(core_radius**3+3*volume/(4*np.pi))**(1/3)
    return volume,radius,(methanol_mass/RHO_M)/max(volume,1e-300)


def screened_field_step(polarization,previous_field,new_field,dt,permittivity,conductivity):
    """Exact dielectric-relaxation response to a linearly varying gas field."""
    tau=permittivity/conductivity;decay=np.exp(-dt/tau)
    slope=(new_field-previous_field)/dt
    polarization=new_field+(polarization-previous_field)*decay-slope*tau*(1-decay)
    internal=3*EPS0/(permittivity+2*EPS0)*(new_field-polarization)
    return polarization,internal


def fly(data,shell,initial_state,drop_radius,charge_fraction,flow_l_min,parameters,keep_history=False):
    p=parameters;c=data.config;g=c.geometry;rc=shell.radius
    state=initial_state.clone();track_index=initial_state.track_index
    track=data.track(track_index);at_lip=data.snapshot(track_index,track["end_time"])
    core_volume=4*np.pi*rc**3/3
    outer_volume=4*np.pi/3*(drop_radius**3-rc**3)
    if outer_volume<=0:raise ValueError("A carrier must be larger than the lipid-shell cell")
    initial_c=at_lip.methanol
    mm=outer_volume*initial_c*RHO_M;mw=outer_volume*(1-initial_c)*RHO_W
    surface_tension=.072*(1-initial_c)+c.fluid.surface_tension*initial_c
    q=charge_fraction*rayleigh_charge(drop_radius,surface_tension)
    # Monodisperse electrical-load closure for a hypothetical full spray.
    # Most drops contain no cells; Q/V sets ALL-drop rate, not cell rate.
    drop_volume=4*np.pi*drop_radius**3/3
    all_drop_rate=(c.fluid.qs+c.fluid.qh)/drop_volume
    inferred_current=q*all_drop_rate
    resistance=max((c.electrical.contact_distance_mm-c.geometry.inner_recess_mm)*1e-3,0.)/(
        c.electrical.sheath_conductivity_s_m*np.pi*(g.c*g.c-g.b*g.b))
    source_voltage=abs(c.electrical.voltage_kv*1000)
    tip_voltage=source_voltage-inferred_current*resistance
    common={"initial_drop_radius_um":drop_radius*1e6,"charge_fraction":charge_fraction,
        "inlet_flow_l_min":flow_l_min,"inferred_spray_current_na":inferred_current*1e9,
        "estimated_tip_voltage_v":tip_voltage,"all_drop_rate_s":all_drop_rate,
        "carrier_charge_c":q,"coating_initial_methanol_fraction":initial_c}
    if tip_voltage<0:
        return dict(common,status="infeasible_electrical_load",arrived=False,entry_state_resolved=False),[]
    if not shell.metrics(state)["shell_continuum_valid"]:
        return dict(common,status="pre_nozzle_shell_model_lost",arrived=False,entry_state_resolved=False,**shell.metrics(state)),[]
    source_lip=abs((data.summary.get("electric",{}).get("ionic") or {}).get("lip_potential_v",source_voltage))
    field_scale=tip_voltage/max(source_lip,1e-20)
    q*=np.sign(c.electrical.voltage_kv)
    gap=g.ms_gap_mm*1e-3;a=p.inlet_diameter_mm*.5e-3;flow=flow_l_min/60000
    z=0.;r=at_lip.r_m+p.alignment_offset_um*1e-6
    vz=float(data.sample(at_lip.r_m,-1e-10,"velocity_z_m_s"));vr=vz*np.tan(np.deg2rad(p.initial_angle_deg))
    temp=c.fluid.temperature_k;t=0.;evaporated_m=0.;evaporated_w=0.;max_mass_error=0.
    outer_tracer=0.;initial_tracer=float(np.dot(state.tracer,shell.transport.volumes))
    initial_methanol=mm+RHO_M*float(np.dot(state.methanol,shell.transport.volumes))
    initial_water=mw+RHO_W*(core_volume-float(np.dot(state.methanol,shell.transport.volumes)))
    def electric(rr,zz):
        return np.array([float(data.sample(rr,zz,"field_r_v_m")),float(data.sample(rr,zz,"field_z_v_m"))])*field_scale
    previous_e=electric(r,z);polarization=previous_e.copy()  # initially charged interface
    history=[];status="flight_timeout";last_output=-np.inf;max_re=0.;max_gas_ratio=0.
    while t<p.max_flight_time_ms*1e-3:
        outer_volume,radius,cout=film_state(rc,mm,mw)
        gamma=.072*(1-cout)+c.fluid.surface_tension*cout
        if abs(q)>=rayleigh_charge(radius,gamma):status="rayleigh_instability_unresolved";break
        if radius-rc<.1e-6:status="coating_depleted_unresolved";break
        if state.membrane_fraction<p.minimum_membrane_fraction:status="membrane_continuum_lost";break
        if state.max_pore_area_fraction>p.maximum_pore_area_fraction:status="pore_area_model_out_of_range";break
        ug=np.array(suction_velocity(r,z,gap,a,flow))
        if p.use_saved_nozzle_gas and getattr(data,"spray",None) is not None:
            frame=len(data.spray["time_s"])-1
            phase=float(data.spray_sample(r,z,"liquid_fraction",frame))
            if np.isfinite(phase) and phase<.1:
                names=["velocity_r_history_m_s","velocity_z_history_m_s"]
                fallback=["velocity_r_m_s","velocity_z_m_s"]
                background=np.array([float(data.spray_sample(r,z,name if name in data.spray else old,frame)) for name,old in zip(names,fallback)])
                if np.isfinite(background).all():
                    max_gas_ratio=max(max_gas_ratio,float(np.linalg.norm(background)/max(np.linalg.norm(ug),1e-20)))
                    ug+=background
        tg=gap_gas_temperature(z,gap,a,flow,p)
        rho_gas=p.ambient_pressure_pa/(GAS_R*tg);mu=gas_viscosity(tg)
        relative=ug-np.array([vr,vz]);speed=float(np.linalg.norm(relative))
        re=2*radius*rho_gas*speed/mu;max_re=max(max_re,re)
        if re>1000:status="drag_model_out_of_range";break
        mean_c=float(np.dot(state.methanol,shell.transport.volumes)/core_volume)
        mass=mm+mw+c.cell.density*core_volume+(RHO_M-RHO_W)*mean_c*core_volume
        drag=6*np.pi*mu*radius*(1+.15*re**.687)
        relaxation=mass/drag
        dt=min(p.max_flight_step_us*1e-6,.15*relaxation,p.max_flight_time_ms*1e-3-t)
        # Do not take infinitely many steps approaching the observation plane.
        if gap-z<1e-9:
            z=gap;status="entered_aperture" if abs(r)+radius<=a else "impacted_inlet_face";break
        try:
            dm=evaporation_rates(mm,mw,radius,temp,tg,speed,p) if p.evaporation else np.zeros(2)
        except ValueError:
            status="evaporation_model_out_of_range";break
        for amount,rate in zip([mm,mw],dm):
            if rate<0 and amount>1e-24:dt=min(dt,.03*amount/(-rate))
        if dt<1e-12:status="integration_or_component_depletion_unresolved";break
        current_e=electric(r,z)
        acceleration=q*current_e/mass
        target=ug+acceleration*relaxation
        velocity=np.array([vr,vz]);decay=np.exp(-dt/relaxation)
        displacement=target*dt+(velocity-target)*relaxation*(1-decay)
        next_velocity=target+(velocity-target)*decay
        if displacement[1]>=gap-z:
            remaining=gap-z
            dt=brentq(lambda h:target[1]*h+(velocity[1]-target[1])*relaxation*(-np.expm1(-h/relaxation))-remaining,
                      0.,dt,xtol=1e-15)
            decay=np.exp(-dt/relaxation)
            displacement=target*dt+(velocity-target)*relaxation*(1-decay)
            next_velocity=target+(velocity-target)*decay
        # Coating conductivity/permittivity are local mixture approximations.
        conductivity=c.electrical.sheath_conductivity_s_m+(1-cout)*1.5
        eps=EPS0*c.chemistry.liquid_relative_permittivity
        new_e=electric(r+displacement[0],min(z+displacement[1],gap))
        polarization,internal_e=screened_field_step(polarization,previous_e,new_e,dt,eps,conductivity)
        if not p.electrical_screening:internal_e=new_e
        previous_e=new_e
        cout_new,tracer_new,transferred,_=shell.step(state,dt,cout,float(np.linalg.norm(internal_e)),conductivity,
            outer_tracer/max(outer_volume,1e-300),outer_volume)
        mm-=transferred*RHO_M;mw+=transferred*RHO_W
        outer_tracer=tracer_new*outer_volume
        # Conservative exchange replaces intracellular water by equal volume
        # methanol; both solvent inventories and evaporated mass are audited.
        mm+=dm[0]*dt;mw+=dm[1]*dt
        if mm < -1e-24 or mw < -1e-24:status="negative_solvent_inventory";break
        mm=max(0.,mm);mw=max(0.,mw)
        evaporated_m-=dm[0]*dt;evaporated_w-=dm[1]*dt
        cp=mm*2530+mw*4180+max(mass-mm-mw,0)*4180
        pr=GAMMA*GAS_R/(GAMMA-1)*mu/.026
        nu=2+.6*np.sqrt(re)*pr**(1/3)
        heat=2*np.pi*radius*nu*.026*(tg-temp)
        s=max(gap-z,0.);hot=p.hot_face_radius_mm*1e-3
        view=.5*(s/np.sqrt(s*s+a*a)-s/np.sqrt(s*s+hot*hot))
        heat+=4*np.pi*radius*radius*p.effective_emissivity*SIGMA_SB*view*((p.inlet_wall_temperature_c+273.15)**4-temp**4)
        heat+=1.17e6*dm[0]+2.44e6*dm[1]
        temp+=dt*heat/max(cp,1e-30)
        r+=displacement[0];z+=displacement[1];vr,vz=next_velocity;t+=dt
        core_m=float(np.dot(state.methanol,shell.transport.volumes))
        err_m=abs(mm+RHO_M*core_m+evaporated_m-initial_methanol)/max(initial_methanol,1e-30)
        err_w=abs(mw+RHO_W*(core_volume-core_m)+evaporated_w-initial_water)/max(initial_water,1e-30)
        err_s=abs(float(np.dot(state.tracer,shell.transport.volumes))+outer_tracer-initial_tracer)/max(initial_tracer,1e-30)
        max_mass_error=max(max_mass_error,err_m,err_w,err_s)
        if keep_history and t-last_output>=1e-5:
            history.append(dict(time_after_lip_s=t,z_m=z,r_m=r,drop_radius_m=radius,
                temperature_k=temp,external_methanol_fraction=cout_new,**shell.metrics(state)))
            last_output=t
        if z>=gap:
            status="entered_aperture" if abs(r)+radius<=a else "impacted_inlet_face";break
        if z<-.1e-3 or abs(r)>a*.5:status="outside_paraxial_flow_validity";break
    arrived=status=="entered_aperture"
    final_volume,radius,cout=film_state(rc,mm,mw)
    return dict(common,status=status,arrived=arrived,entry_state_resolved=arrived,
        flight_time_ms=t*1e3,final_r_um=r*1e6,final_z_mm=z*1e3,
        final_drop_radius_um=radius*1e6,final_temperature_c=temp-273.15,
        final_coating_methanol_fraction=cout,coating_thickness_um=(radius-rc)*1e6,
        tracer_released_before_lip_fraction=1-initial_tracer/core_volume,
        tracer_in_carrier_fraction=outer_tracer/core_volume,
        tracer_carried_total_fraction=(outer_tracer+float(np.dot(state.tracer,shell.transport.volumes)))/core_volume,
        max_relative_species_balance_error=max_mass_error,max_droplet_reynolds=max_re,
        max_nozzle_background_to_suction_ratio=max_gas_ratio,
        **shell.metrics(state)),history
