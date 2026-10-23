"""Join measured geometry, CFD exposure histories and explicit state scenarios."""
from itertools import product
from dataclasses import dataclass
import numpy as np
from .parameters import StateParameters
from .inlet import estimate_inlet
from .membrane import LipidShell
from .flight import fly


@dataclass
class StateAnalysis:
    parameters: StateParameters
    source: str
    source_config: dict
    inlet: dict
    outlet_rates: dict
    cases: list
    rate_cases: list
    histories: list
    pre_histories: list
    limitations: list


def outlet_rates(config,summary,rate_config=None):
    def finite_number(value):
        return (isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value)))
    inputs=rate_config or config
    incoming=inputs.cell.concentration_per_ml*inputs.fluid.sample_flow_ul_min/60000
    pending=inputs.fluid.sample_flow_ul_min!=config.fluid.sample_flow_ul_min
    p=summary.get("particles",{})
    fraction=p.get("transport_fraction")
    if (fraction is None and finite_number(p.get("count")) and p["count"] > 0
            and finite_number(p.get("exited", 0))):
        fraction=p.get("exited",0)/p["count"]
    if not finite_number(fraction) or not 0 <= fraction <= 1:
        fraction = None
    rate=None if fraction is None else incoming*fraction
    lower_fraction = upper_fraction = None
    bounds_note = "Incomplete transport status population; censoring bounds unavailable"
    status_keys = ("count", "exited", "wall_contacts", "censored_or_reverse")
    counts = [p.get(key) for key in status_keys]
    valid_counts = all(
        finite_number(value) and value >= 0 and float(value).is_integer()
        for value in counts)
    if valid_counts:
        count, exits, contacts, censored = map(int, counts)
        valid_population = count > 0 and exits+contacts+censored == count
        agrees = (valid_population and fraction is not None and np.isfinite(fraction)
                  and np.isclose(fraction, exits/count, rtol=1e-9, atol=1e-10))
        if agrees and not pending:
            lower_fraction = exits/count
            upper_fraction = (exits+censored)/count
            bounds_note = ("Model censoring bounds: lower counts resolved lip exits; upper additionally assumes all "
                           "censored/reverse samples later exit. Conditional on the sampled population and contact model; "
                           "statistical sampling uncertainty and unmodeled losses require separate assessment")
        else:
            bounds_note = ("Changed flow requires new transport responses" if pending else
                           "Inconsistent transport counts or exit fraction; censoring bounds unavailable")
    delta=config.events.resolving_time_ms*1e-3
    windows=[dict(row,observed_nozzle_cells_s=row.get('observed_nozzle_cells_s',row['nozzle_exits']/row['duration_s']))
             for row in summary.get('events',{}).get('summaries',[])] if not pending else []
    return {"injection_cells_s":incoming,"modeled_transport_fraction":fraction,"nozzle_cells_s":rate,
        "modeled_transport_lower_fraction":lower_fraction,
        "unresolved_transport_upper_fraction":upper_fraction,
        "modeled_nozzle_lower_cells_s":None if lower_fraction is None else incoming*lower_fraction,
        "unresolved_nozzle_upper_cells_s":None if upper_fraction is None else incoming*upper_fraction,
        "transport_bounds_note":bounds_note,
        "transport_pending":pending,"rate_input_sample_flow_ul_min":inputs.fluid.sample_flow_ul_min,
        "transport_censored_samples":p.get("censored_or_reverse"),
        "time_isolated_nozzle_cells_s":None if rate is None else rate*np.exp(-2*rate*delta),
        "isolation_definition":f"preceding AND following event gaps >= {delta:g} s; independent Poisson steady-feed approximation",
        "observation_windows":windows,
        "ms_detected_single_cell_events_s":None,
        "detection_note":"entrance-plane crossing is not ion detection; ionization, transmission and detector response unknown"}


def droplet_occupancy(cell_rate,drop_rate):
    """Uniform independent loading diagnostic; focused breakup may violate it."""
    mu=cell_rate/drop_rate
    return {"mean_cells_per_drop":mu,"one_cell_drop_events_s":cell_rate*np.exp(-mu),
        "nonempty_drop_single_cell_fraction":mu/np.expm1(mu) if mu>1e-12 else 1.,
        "occupancy_model":"uniform independent Poisson loading; diagnostic only, not resolved focused jet breakup"}


def reweight_concentration(result,data):
    """Reuse independent carrier histories, recomputing all rate/occupancy terms."""
    if getattr(data,'rate_preview',False):raise ValueError(' Sample flow changed ,  Update the source first  CFD,  Recompute membrane / Transfer responses for the updated state ')
    from dataclasses import replace
    from ..particle.events import concentration_only_change
    if not concentration_only_change(result.source_config,data.config):
        raise ValueError(" Flow or transport parameters changed ,  Recompute cell state ")
    rates=outlet_rates(data.config,data.summary);incoming=rates['nozzle_cells_s']
    rows=[]
    for original in result.rate_cases:
        row=dict(original);n=row['sampled_tracks']
        drop_rate=(data.config.fluid.qs+data.config.fluid.qh)/(4*np.pi/3*(row['initial_drop_radius_um']*1e-6)**3)
        occupancy=droplet_occupancy(incoming,drop_rate)
        low=incoming*row['arrived']/n if row['electrical_load_admissible'] else None
        row.update(modeled_carrier_entry_cells_s=low,
            unresolved_entry_upper_cells_s=incoming*(row['arrived']+row['unresolved'])/n if row['electrical_load_admissible'] else None,
            modeled_one_cell_carrier_entry_events_s=None if low is None else low*np.exp(-occupancy['mean_cells_per_drop']),
            **occupancy)
        rows.append(row)
    return replace(result,source=data.source,source_config=data.config.to_dict(),outlet_rates=rates,rate_cases=rows)


def pre_nozzle(data,index,shell,keep_history=False):
    track=data.track(index);state=shell.initial();state.track_index=index
    history=[];p=shell.p
    for left,right in zip(track["t"][:-1],track["t"][1:]):
        t=float(left)
        c_left=float(np.interp(left,track["t"],track["C"]));c_right=float(np.interp(right,track["t"],track["C"]))
        # Long buffer segment is smooth; the exposure-changing segment uses
        # millisecond steps. Implicit diffusion avoids explicit h^2/D limits.
        maximum=.2 if max(c_left,c_right)<1e-10 else p.max_pre_step_ms*1e-3
        while t<right-1e-13:
            dt=min(maximum,right-t);mid=t+dt/2
            rr=float(np.interp(mid,track["t"],track["r"]));zz=float(np.interp(mid,track["t"],track["z"]))
            concentration=float(np.interp(mid,track["t"],track["C"]))
            field=float(np.hypot(data.sample(rr,zz,"field_r_v_m"),data.sample(rr,zz,"field_z_v_m")))
            conductivity=data.config.electrical.sheath_conductivity_s_m+1.5*(1-concentration)
            shell.step(state,dt,concentration,field,conductivity)
            t+=dt
            if keep_history and (not history or t-history[-1]["time_s"]>.003):
                history.append(dict(time_s=t,external_methanol_fraction=concentration,**shell.metrics(state)))
            if not shell.metrics(state)["shell_continuum_valid"]:
                return state,history
    return state,history


def analyze(data,parameters=None,progress=lambda text:None):
    if getattr(data,'rate_preview',False):raise ValueError(' Sample flow changed ,  Recompute flow and transport first ,  Then compute membrane state ')
    p=(parameters or StateParameters()).validate()
    rc=data.config.cell.diameter_um*.5e-6
    if min(p.drop_radii_um)*1e-6<=rc:raise ValueError("Every carrier radius must exceed the cell radius")
    inlet=estimate_inlet(p);rates=outlet_rates(data.config,data.summary)
    if rates["nozzle_cells_s"] is None:raise ValueError("Source results do not contain a transport fraction")
    eligible=[i for i in range(len(data.ids)) if data.snapshot(i,data.track(i)["end_time"]).z_m>=-1e-9]
    if not eligible:raise ValueError("No saved trajectories cross the middle lip")
    selected=np.array(eligible)[np.unique(np.linspace(0,len(eligible)-1,min(p.sample_tracks,len(eligible)),dtype=int))]
    pre={};pre_histories=[]
    for permeability,rate,index in product(p.methanol_permeabilities_m_s,p.dissolution_rates_s,selected):
        shell=LipidShell(rc,p,permeability,rate)
        state,history=pre_nozzle(data,int(index),shell,keep_history=(int(index)==int(selected[0])))
        pre[(permeability,rate,int(index))]=(shell,state)
        if history:pre_histories.append(dict(permeability_m_s=permeability,dissolution_rate_s=rate,history=history))
    cases=[];rate_cases=[];histories=[]
    combinations=list(product(p.drop_radii_um,p.charge_fractions,p.flow_factors,p.methanol_permeabilities_m_s,p.dissolution_rates_s))
    for number,(radius,charge,flow_factor,permeability,rate) in enumerate(combinations):
        group=[]
        for index in selected:
            shell,state=pre[(permeability,rate,int(index))]
            row,history=fly(data,shell,state,radius*1e-6,charge,
                inlet["baseline_ambient_flow_l_min"]*flow_factor,p,keep_history=(number%max(1,len(combinations)//8)==0 and int(index)==int(selected[0])))
            row.update(case_id=len(cases),transport_sample_id=int(data.ids[int(index)]),
                permeability_m_s=permeability,dissolution_rate_s=rate,flow_factor=flow_factor)
            cases.append(row);group.append(row)
            if history:histories.append(dict(case_id=row["case_id"],history=history))
        admissible=all(x["status"]!="infeasible_electrical_load" for x in group)
        arrivals=sum(x["arrived"] for x in group)
        rejected=sum(x["status"]=="impacted_inlet_face" for x in group)
        unresolved=len(group)-arrivals-rejected
        incoming=rates["nozzle_cells_s"]
        low=incoming*arrivals/len(group) if admissible else None
        high=incoming*(arrivals+unresolved)/len(group) if admissible else None
        occupancy=droplet_occupancy(incoming,group[0]["all_drop_rate_s"])
        rate_cases.append(dict(initial_drop_radius_um=radius,charge_fraction=charge,flow_factor=flow_factor,
            permeability_m_s=permeability,dissolution_rate_s=rate,sampled_tracks=len(group),
            arrived=arrivals,unresolved=unresolved,rejected=rejected,electrical_load_admissible=admissible,
            modeled_carrier_entry_cells_s=low,unresolved_entry_upper_cells_s=high,
            sampled_carrier_transmission_fraction=None if not admissible else arrivals/len(group),
            modeled_one_cell_carrier_entry_events_s=None if low is None else low*np.exp(-occupancy["mean_cells_per_drop"]),
            biological_intact_cell_entry_rate_s=None,ms_detected_single_cell_rate_s=None,**occupancy))
        if number%max(1,len(combinations)//15)==0:
            progress(f"Lipid-shell state scenarios: {number+1}/{len(combinations)} groups")
    limits=[
        "Conditional lipid vesicle approximation; no cortex, organelles, metabolism, osmotic volume change or shape dynamics.",
        "Membrane permeability/dissolution and generic tracer properties are uncalibrated scenario inputs; ranges are not confidence intervals.",
        "Uniform membrane thinning from k*C^2 is phenomenological; membrane lipid fraction is not viability or lysis probability.",
        "Initial cell-bearing droplets are assumed at the middle lip; their size/charge distribution is not predicted by the current two-phase solver.",
        "Monodisperse charge-flux closure estimates supply voltage drop; real spray current and sheath conductivity are unmeasured.",
        "Upstream exposure and flow remain those of the selected CFD result; inferred droplet load does not recompute upstream electroosmosis.",
        "Conducting coating screens the cell; dielectric-relaxation and spherical RC models omit detailed multilayer interface fields.",
        "Fanno estimate and paraxial disk suction approximate inlet flow; no full heated cone/aperture CFD or gap radial heat convection.",
        "When enabled, the saved gas snapshot is added as a frozen nozzle-flow perturbation; nonlinear flow interaction and transient evolution are not resolved.",
        "320 C is the inlet wall setpoint; particle heat balance includes convection, radiation and latent cooling only before entry.",
        "Raoult/Fick binary evaporation uses ideal activities and approximate gas transport; low-T methanol uses a Clapeyron extrapolation.",
        "Rayleigh instability, depleted coating, large offsets and loss of shell continuum produce unresolved states, not invented daughter drops.",
        "Released material lost upstream is not automatically assigned to the carrier or counted as an intact-cell arrival.",
        "Poisson occupancy assumes independent uniform loading; hydrodynamic focusing and breakup correlations need experimental calibration.",
        "The 5 ms source event-isolation example is not a Q Exactive acquisition cycle or instrument resolution specification.",
        "Reported entry rates are conditional geometric carrier-event estimates, not biological survival or MS detection efficiencies."]
    return StateAnalysis(p,data.source,data.config.to_dict(),inlet,rates,cases,rate_cases,histories,pre_histories,limits)
