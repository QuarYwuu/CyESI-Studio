"""Inspectable scenario output; unresolved endpoints remain blank, never zero."""
from pathlib import Path
from collections import Counter
import html
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..report.export import clean,write_csv
from ..visualization.figures import serialized_plotting


def load_analysis(directory):
    """Restore exported scenario responses for concentration-only updates."""
    from .analysis import StateAnalysis
    from .parameters import StateParameters
    root=Path(directory)
    summary=json.loads((root/'state_summary.json').read_text(encoding='utf-8'))
    def parse(value):
        if value=='':return None
        if value in ('True','False'):return value=='True'
        try:return float(value)
        except ValueError:return value
    def read(name):
        path=root/name
        if not path.exists():return []
        with path.open(encoding='utf-8-sig',newline='') as stream:
            return [{k:parse(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    def histories(name,keys):
        groups={}
        for row in read(name):
            key=tuple(row.pop(k) for k in keys)
            groups.setdefault(key,[]).append(row)
        return [dict(zip(keys,key),history=points) for key,points in groups.items()]
    return StateAnalysis(StateParameters.load(root/'state_parameters.json'),summary['source'],summary['source_config'],
        summary['inlet'],summary['outlet_rates'],read('state_cases.csv'),read('single_cell_rates.csv'),
        histories('flight_histories.csv',['case_id']),
        histories('pre_nozzle_histories.csv',['permeability_m_s','dissolution_rate_s']),summary['limitations'])


def summarize(result):
    arrived=[x for x in result.cases if x["arrived"]]
    strong_background=[x for x in arrived if x.get("max_nozzle_background_to_suction_ratio",0)>.3]
    def span(name):
        values=[x[name] for x in arrived if x.get(name) is not None]
        return [float(min(values)),float(max(values))] if values else None
    return {"model":"conditional spherical lipid-shell / charged carrier model; uncalibrated",
        "source":result.source,"source_config":result.source_config,"inlet":result.inlet,"outlet_rates":result.outlet_rates,
        "scenario_counts":dict(Counter(x["status"] for x in result.cases)),
        "scenario_total":len(result.cases),"arrived_scenarios":len(arrived),
        "flow_approximation_quality":{"arrived_cases_with_nozzle_background_above_30_percent":len(strong_background),
            "max_nozzle_background_to_suction_ratio":max((x.get("max_nozzle_background_to_suction_ratio",0) for x in arrived),default=None),
            "interpretation":"large ratios require a jointly solved suction/nozzle NS field; current additive histories are conditional approximations"},
        "arrived_scenario_ranges":{name:span(name) for name in ["flight_time_ms","final_temperature_c",
            "intracellular_methanol_fraction","intracellular_tracer_fraction","membrane_lipid_remaining_fraction",
            "max_transmembrane_potential_v","final_drop_radius_um","tracer_carried_total_fraction"]},
        "biological_intact_cell_rate_s":None,"ms_detection_rate_s":None,
        "range_interpretation":"parameter-scenario envelopes; no probabilities or confidence intervals assigned",
        "limitations":result.limitations}


def _save(fig,root,name):
    fig.savefig(root/f"{name}.png",dpi=150);fig.savefig(root/f"{name}.svg");plt.close(fig)


@serialized_plotting
def figures(result,root):
    arrived=[x for x in result.cases if x["arrived"]]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    fig.suptitle("Lipid-shell state at the entrance | Conditional scenarios, uncalibrated",fontweight="bold")
    for item in result.histories:
        points=item["history"]
        axes[0,0].plot([x["z_m"]*1e3 for x in points],[x["temperature_k"]-273.15 for x in points],lw=.8,alpha=.7)
        axes[0,1].plot([x["z_m"]*1e3 for x in points],[x["drop_radius_m"]*1e6 for x in points],lw=.8,alpha=.7)
    axes[0,0].set(xlabel="z after middle lip (mm)",ylabel="Carrier/core temperature (C)",title="Wall setpoint is NOT particle temperature")
    axes[0,1].set(xlabel="z after middle lip (mm)",ylabel="Carrier radius (um)",title="Evaporation; selected computed histories")
    if arrived:
        pts=axes[1,0].scatter([x["permeability_m_s"] for x in arrived],
            [x["intracellular_methanol_fraction"]*100 for x in arrived],
            c=[x["dissolution_rate_s"] for x in arrived],cmap="viridis",s=14,alpha=.5)
        fig.colorbar(pts,ax=axes[1,0],label="Assumed dissolution k (1/s)")
        axes[1,1].scatter([x["membrane_lipid_remaining_fraction"]*100 for x in arrived],
            [x["intracellular_tracer_fraction"]*100 for x in arrived],s=13,alpha=.35)
    axes[1,0].set(xscale="log",xlabel="Assumed methanol permeability (m/s)",ylabel="Intracellular methanol (%)",title="Physical diffusion; parameter envelope")
    axes[1,1].set(xlabel="Remaining model membrane lipid (%)",ylabel="Intracellular generic tracer (%)",title="Neither axis is biological viability")
    _save(fig,root,"cell_state")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    rates=result.outlet_rates;labels=["Injection","Nozzle exits","Time-isolated nozzle"]
    values=[rates["injection_cells_s"],rates["nozzle_cells_s"],rates["time_isolated_nozzle_cells_s"]]
    axes[0].bar(labels,values,color=["#467ead","#2ca59c","#b18b43"])
    for i,value in enumerate(values):axes[0].text(i,value+.15,f"{value:.2f}",ha="center")
    axes[0].set(ylabel="Cells or events / s",title="Steady expectation; event isolation definition in report",ylim=(0,max(1.,max(values)*1.22)))
    radii=sorted(set(x["initial_drop_radius_um"] for x in result.rate_cases))
    for radius in radii:
        group=[x for x in result.rate_cases if x["initial_drop_radius_um"]==radius and x["electrical_load_admissible"]]
        if group:
            low=min(x["modeled_carrier_entry_cells_s"] for x in group)
            high=max(x["unresolved_entry_upper_cells_s"] for x in group)
            axes[1].plot([radius,radius],[low,high],lw=7,color="#2ca59c",alpha=.65)
            axes[1].scatter([radius,radius],[low,high],s=25,color="#277d99")
    axes[1].axhline(rates["nozzle_cells_s"],ls="--",lw=.8,color="#555",label="Nozzle-cell expectation")
    axes[1].set(xlabel="Assumed carrier radius (um)",ylabel="Modeled carrier entry cells/s",title="Scenario envelope includes unresolved-entry bounds",ylim=(-.5,rates["nozzle_cells_s"]*1.15))
    axes[1].legend(fontsize=9)
    _save(fig,root,"single_cell_rates")


def export(result,directory):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    result.parameters.save(root/"state_parameters.json")
    (root/"transport_source_config.json").write_text(json.dumps(result.source_config,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=summarize(result)
    (root/"state_summary.json").write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write_csv(root/"state_cases.csv",result.cases)
    write_csv(root/"single_cell_rates.csv",result.rate_cases)
    write_csv(root/"flight_histories.csv",({"case_id":item["case_id"],**x} for item in result.histories for x in item["history"]))
    write_csv(root/"pre_nozzle_histories.csv",({"permeability_m_s":item["permeability_m_s"],
        "dissolution_rate_s":item["dissolution_rate_s"],**x} for item in result.pre_histories for x in item["history"]))
    figures(result,root)
    from ..acquisition.q_exactive import analyze as acquisition_analysis
    from ..acquisition.report import export as export_acquisition
    export_acquisition(acquisition_analysis(result.outlet_rates['nozzle_cells_s'],
        source=result.source,source_config=result.source_config),root/'acquisition')
    rows="".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k,v in summary["arrived_scenario_ranges"].items())
    statuses="".join(f"<li>{html.escape(k)}: {v}</li>" for k,v in summary["scenario_counts"].items())
    warnings="".join(f"<li>{html.escape(x)}</li>" for x in result.limitations)
    outlet=result.outlet_rates;flow=result.inlet
    observed_rows="".join(f"<tr><td>{x['duration_s']:g} s</td><td>{x['nozzle_exits']}</td><td>{x['nozzle_exits']/x['duration_s']:.3f} cells/s</td></tr>" for x in outlet.get('observation_windows',[]))
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q Exactive  Cell-state model at the inlet </title><style>body{{font:16px/1.7 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#20354b;background:#f5f7fa}}section{{padding:20px;background:white;margin:18px 0;border-radius:10px}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #ddd;padding:7px}}.notice{{color:#815410;background:#fff2d9;padding:18px}}strong{{font-size:23px}}</style>
<h1>Q Exactive  Inlet cell state and single-source rate </h1><p class="notice"> Parameterized model ,  Biological calibration pending . Aqueous-core cell approximation ＋ Lipid membrane ;  Initial cell-bearing droplet size ,  Specified charge and membrane parameters . Parameter ranges denote sensitivity settings .</p>
<section><h2> Single-source rate </h2><p> Injection  <strong>{outlet['injection_cells_s']:.2f} cells/s</strong>;  Steady model outlet rate  <strong>{outlet['nozzle_cells_s']:.2f} cells/s</strong>;  Based on adjacent event intervals ,  Temporally isolated event rate  <strong>{outlet['time_isolated_nozzle_cells_s']:.2f}/s</strong>.</p><p>{html.escape(outlet['isolation_definition'])}</p>
<p> Carrier-droplet entry rates by setting  <a href="single_cell_rates.csv">single_cell_rates.csv</a>;  Unresolved breakup / Membrane-limit cases give unresolved upper bounds . Intact viable-cell entry rate ,  Instrument detection rate unspecified . Aperture crossing ,  Single-cell droplet purity ,  Temporal isolation and  MS  Detection are distinct metrics .Poisson  Independent uniform-loading occupancy reference ;  Focused-jet loading requires a separate model .</p><p> Continuous-source window statistics : </p><table><tr><td> Observation window </td><td> Outlet event count </td><td> Window-averaged outlet rate </td></tr>{observed_rows}</table><img src="single_cell_rates.png"></section>
<section><h2> Instrument boundary and aspiration estimate </h2><p> Specified input  Q Exactive,  Inlet ID  {flow['inlet_diameter_mm']:g} mm,  Wall temperature  {flow['wall_temperature_c']:g}°C. Reference transfer-capillary length  {flow['transfer_tube_length_mm']:g} mm. Reference aspiration estimate  <strong>{flow['baseline_ambient_flow_l_min']:.3g} L/min</strong> ( Actual volume flow at ambient conditions ),  Multiplier sweep  {html.escape(str(result.parameters.flow_factors))}.</p><p> Frictional choked compressible-flow estimate ,  Paraxial suction-disk approximation ;  Gap temperature determined from the thermal model  320°C. Particle energy balance includes evaporative cooling and radiation ,  Transfer domain ends at the inlet plane ,  Heated-capillary continuation excluded .</p><p> Frozen final-frame auxiliary gas field .{summary['flow_approximation_quality']['arrived_cases_with_nozzle_background_above_30_percent']}  Entry cases with auxiliary-gas speed exceeding aspiration by  30%;  Joint aspiration-flow modelling required for these cases / Emitter  CFD  Check ,  Parameterized time and temperature ranges .</p></section>
<section><h2> State ranges among arriving cases </h2><table>{rows}</table><img src="cell_state.png"><p> Neutral passive intracellular tracer ;  In the  70 cm  Liquid path permits passive leakage . Cell metabolism ,  Active transport and metabolite-specific permeability require calibration . Previously released material tracked separately from the carrier .</p><ul>{statuses}</ul></section>
<section><h2> Source files and model scope </h2><p><a href="state_parameters.json"> All model settings </a> · <a href="state_summary.json"> Summary </a> · <a href="state_cases.csv"> Per-case states </a> · <a href="flight_histories.csv"> Selected transfer histories </a> · <a href="pre_nozzle_histories.csv"> Internal membrane histories </a></p><ul>{warnings}</ul></section>
<section><h2> References </h2><p><a href="https://tools.thermofisher.com/content/sfs/manuals/IonTransferTube.pdf">Thermo  Transfer-capillary dimensions </a> · <a href="https://www.grc.nasa.gov/www/k-12/airplane/mflchk.html">NASA  Critical mass flow </a> · <a href="https://webbook.nist.gov/cgi/cbook.cgi?ID=C67561&amp;Mask=4">NIST  Methanol properties </a> · <a href="https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&amp;Mask=4">NIST  Water properties </a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC1751390/"> Single-cell poration model </a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC1305119/"> Vesicle deformation and poration experiments </a></p><p> Method references ,  Lymphocyte-specific / Methanol permeability and dissolution require calibration .</p></section></html>'''
    text=text.replace('</h1>','</h1><p><a href="acquisition/acquisition_report.html"> Inspect  Q Exactive  Acquisition method ,  Resolution and theoretical single-source rates </a> ( Separate from  5 ms  Source-event diagnostics )</p>',1)
    (root/"cell_state_report.html").write_text(text,encoding="utf-8")
    return root
