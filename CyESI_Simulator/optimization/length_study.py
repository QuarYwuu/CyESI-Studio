"""Tube-length, orientation and shear comparisons.

Each changed flow requires a new tip solution. Force comparisons use matched
inlet samples. Independent residence delays preserve a Poisson arrival process;
they are separate from the molecular contribution to signal duration.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import html
import json
import math
import numpy as np
from ..config import Config
from ..acquisition.q_exactive import AcquisitionParameters, analyze as acquisition
from ..fluid.supply import resistance, hydrostatic_assist_pa, infer_pressure, segments


@dataclass
class StudyParameters:
    lengths_cm: list[float] = field(default_factory=lambda: [5., 10., 20., 40., 70., 100.])
    orientations: list[str] = field(default_factory=lambda: ["horizontal", "down", "up", "horizontal_down", "down_horizontal"])
    cells: int = 128
    ablations: bool = True
    reference_length_cm: float = 70.
    # Assumed mechanical-hazard coefficients, not lymphocyte viability fits.
    damage_coefficients: list[float] = field(default_factory=lambda: [0., .001, .01])

    def validate(self):
        if not self.lengths_cm or any(not math.isfinite(x) or x <= 0 for x in self.lengths_cm):
            raise ValueError("Tube lengths must be positive finite cm values")
        allowed = {"horizontal", "down", "up", "horizontal_down", "down_horizontal", "custom"}
        if not self.orientations or set(self.orientations)-allowed: raise ValueError("Invalid orientation cases")
        if not isinstance(self.cells, int) or isinstance(self.cells, bool) or self.cells < 8: raise ValueError("At least 8 transport samples needed")
        if not math.isfinite(self.reference_length_cm) or self.reference_length_cm <= 0: raise ValueError("Invalid reference length")
        if any(not math.isfinite(k) or k < 0 for k in self.damage_coefficients): raise ValueError("Damage coefficients must be nonnegative")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig"))).validate()


def analyze(config=None, parameters=None, acquisition_parameters=None, progress=lambda x: None, on_case=lambda state: None):
    from ..engine import liquid_solution
    from ..particle.cell_transport import simulate
    from ..electro.electrostatics import solve as electric_solution
    config = deepcopy(config or Config()).validate()
    p = (parameters or StudyParameters()).validate()
    ap = (acquisition_parameters or AcquisitionParameters()).validate()
    from ..performance import worker_budget
    lengths=list(dict.fromkeys(p.lengths_cm+([p.reference_length_cm] if p.ablations else [])))
    workers=worker_budget(config.numerics.workers,len(lengths))
    if workers>1:
        return _parallel_analyze(config,p,ap,lengths,workers,progress,on_case)
    cases = [(length, orientation, "full") for length in p.lengths_cm for orientation in p.orientations]
    if p.ablations:
        if p.reference_length_cm not in p.lengths_cm:
            cases.extend((p.reference_length_cm, orientation, "full") for orientation in p.orientations)
        for orientation in p.orientations:
            cases.extend((p.reference_length_cm, orientation, force) for force in
                         ("without_gravity", "without_saffman", "without_lubrication"))
            if config.supply.wall_capture_rate_s:
                cases.append((p.reference_length_cm, orientation, "without_adhesion"))
    cache = {}; rows = []; acquisition_rows = []; damage_rows = []; histories = {}; initial_state=None
    for i, (length, orientation, variant) in enumerate(cases):
        progress(f"Length/shear study {i+1}/{len(cases)}: {length:g} cm, {orientation}, {variant}")
        c = deepcopy(config); c.geometry.sample_length_mm = length*10
        c.supply.orientation = orientation; c.cell.count = p.cells
        c.numerics.saved_trajectories = min(p.cells, 24)
        c.spray.enabled = False
        if variant == "without_gravity": c.supply.gravity_enabled = False
        if variant == "without_saffman": c.supply.saffman_enabled = False
        if variant == "without_lubrication": c.supply.lubrication_enabled = False
        if variant == "without_adhesion": c.supply.wall_capture_rate_s = 0.
        identity = f"case_{i:03d}"
        row = {"case_id": identity, "length_cm": length, "orientation": orientation,
               "variant": variant, "sample_control": c.supply.sample_control,
               "sample_pressure_mbar": c.supply.sample_pressure_mbar, "cells": p.cells}
        try:
            c.validate()
            # Orientation affects the axisymmetric fluid only through gravity
            # head and hence actual Q. Particle gravity remains fully segmented.
            key = (length if c.supply.sample_control == "pressure" else None,
                   hydrostatic_assist_pa(c) if c.supply.sample_control == "pressure" else None)
            if key not in cache:
                flow, solvent, coupling = liquid_solution(c, progress, initial_state)
                initial_state=(solvent,getattr(solvent,'ionic',None))
                electric = electric_solution(c, getattr(solvent, "ionic", None))
                cache[key] = (flow, solvent, coupling, electric, c.fluid.sample_flow_ul_min)
            flow, solvent, coupling, electric, actual_q = cache[key]
            c.fluid.sample_flow_ul_min = actual_q
            particles = simulate(c, solvent, electric, progress=progress)
            ok = particles.exited; d = particles.diagnostics; stress = particles.stress
            source_rate = c.cell.concentration_per_ml*actual_q/60000
            nozzle_rate = source_rate*ok.mean()
            n_ok = int(ok.sum()); fraction = float(ok.mean())
            # Wilson 95% interval quantifies sampling only, not model uncertainty.
            den = 1+1.96**2/p.cells; center = (fraction+1.96**2/(2*p.cells))/den
            half = 1.96*math.sqrt(fraction*(1-fraction)/p.cells+1.96**2/(4*p.cells**2))/den
            row.update({"status": "ok", "actual_sample_flow_ul_min": actual_q,
                "source_cells_s": source_rate, "nozzle_cells_s": float(nozzle_rate),
                "transport_fraction": fraction, "transport_sampling_ci95_low": center-half,
                "transport_sampling_ci95_high": center+half,
                "mean_residence_s": d["mean_residence_s"],
                "residence_p95_s": float(np.quantile(particles.residence_time[ok], .95)) if n_ok else None,
                "residence_std_s": float(np.std(particles.residence_time[ok])) if n_ok else None,
                "wall_stress_pa": 4*c.fluid.sample_viscosity*c.fluid.qs/(np.pi*c.geometry.a**3),
                "mean_shear_dose_pa_s": d["mean_shear_dose_pa_s"],
                "peak_stress_p95_pa": float(np.quantile(stress["peak_equivalent_stress_pa"][ok], .95)) if n_ok else None,
                "mean_extension_dose_pa_s": float(np.mean(stress["extension_dose_pa_s"][ok])) if n_ok else None,
                "mean_near_wall_time_s": d["mean_near_wall_time_s"],
                "near_wall_exposed_fraction": float(np.mean(stress["near_wall_time_s"] > 0)),
                "wall_captures": int(np.sum(particles.status == 2)), "censored": int(np.sum(particles.status == 3)),
                "outlet_r95_um": d["outlet_r95_um"], "focusing_fraction": d["focusing_fraction_of_exits"],
                "mean_methanol_dose_fraction_s": float(np.mean(particles.methanol_dose[ok])) if n_ok else None,
                "mean_outlet_methanol_fraction": d["mean_outlet_methanol_fraction"],
                "hydrostatic_assist_pa": hydrostatic_assist_pa(c),
                "developed_pressure_for_flow_mbar": infer_pressure(c),
                "fem_sample_inlet_pressure_pa": coupling.get("supply", {}).get("iterations", [{}])[-1].get("tip_inlet_pressure_pa"),
                "pressure_residual_pa": coupling.get("supply", {}).get("pressure_residual_pa"),
                "liquid_coupling_converged": coupling["converged"],
                "wall_capture_rate_s": c.supply.wall_capture_rate_s,
                "conditional_mechanical_retained_fraction": d["conditional_mechanically_retained_fraction"],
                "actual_ms_detection_cells_s": None})
            ac = acquisition(float(nozzle_rate), ap)
            for ac_row in ac["rows"]:
                acquisition_rows.append({"case_id": identity, "length_cm": length, "orientation": orientation,
                    "variant": variant, **{k: v for k, v in ac_row.items() if k != "expected_windows"}})
            # Do not substitute mechanical survival for metabolite detection:
            # a disrupted cell can still supply an identifiable ion packet.
            for k in p.damage_coefficients:
                retained = float(np.mean(ok*np.exp(-k*stress["damage_exposure"])))
                damage_rows.append({"case_id": identity, "length_cm": length, "orientation": orientation,
                    "variant": variant, "assumed_hazard_coefficient": k,
                    "hazard_exponent": c.supply.shear_damage_exponent,
                    "conditional_mechanical_retained_fraction": retained,
                    "conditional_mechanically_retained_cells_s": source_rate*retained,
                    "calibrated": False})
            if particles.bulk_history is not None and variant == "full":
                histories[identity] = particles.bulk_history
        except (ValueError, RuntimeError) as error:
            row.update(status="failed", error=str(error))
        rows.append(row)
        on_case({"config":config.to_dict(),"parameters":asdict(p),"completed_cases":len(rows),
                 "total_cases":len(cases),"rows":rows,"acquisition_rows":acquisition_rows,"damage_scenarios":damage_rows})
    # Counterfactual differences use absolute rates and fractions, not a made-up
    # positive 'shear improvement' score. Gravity ablation changes hydrostatics.
    for row in rows:
        baseline = next((r for r in rows if r["variant"] == "full" and r["orientation"] == row["orientation"]
                         and r["length_cm"] == row["length_cm"] and r["status"] == "ok"), None)
        if row["status"] == "ok" and baseline is not None:
            row["nozzle_rate_delta_from_full_s"] = row["nozzle_cells_s"]-baseline["nozzle_cells_s"]
            row["residence_delta_from_full_s"] = (row["mean_residence_s"]-baseline["mean_residence_s"]
                if row["mean_residence_s"] is not None and baseline["mean_residence_s"] is not None else None)
    return {"config": config.to_dict(), "parameters": asdict(p), "acquisition_parameters": ap.to_dict(),
        "execution":{"workers":1,"groups":"length","particle_backend":config.numerics.particle_backend},
        "rows": rows, "acquisition_rows": acquisition_rows, "damage_scenarios": damage_rows,
        "histories": histories, "limitations": [
            "Sample pressure and sheath flow use different control modes; each changed Q has a newly resolved tip field.",
            "Pressure includes developed resistance, reservoir head, tube elevation and internal FEM backpressure; external transient spray feedback is not yet closed.",
            "Horizontal sedimentation uses 3D particle cross-sections in developed straight fluid segments; bends preserve the transverse frame, without Dean flow or bend impact.",
            "The last resolved millimetres use axisymmetric cell dynamics; transverse settling and nonaxisymmetric fluid stress in that short tip remain unresolved.",
            "Nonadhesive finite-size wall exclusion does not imply irreversible cell loss. Capture hazard and roughness gap require quartz/cell measurements.",
            "Saffman lift requires axial slip; force-free developed shear does not automatically focus a rigid cell.",
            "Equivalent stress includes hoop, shear and extension on the resolved mesh; sharp lip stresses need rounding and mesh convergence.",
            "Mechanical survival scenarios are uncalibrated hazard sensitivities, not lymphocyte viability or extraction yield.",
            "Independent transit-time dispersion preserves steady Poisson arrivals; it is not assigned as the width of one cell's MS signal.",
            "Q Exactive results are unique pure sampling opportunities, using the instrument timing inputs; signal transmission, identification and true MS detection remain uncalibrated.",
            "Sampling confidence intervals cover finite trajectory sampling only. No experimental validation or model-parameter uncertainty is implied."]}


def _length_job(payload):
    """One length per process retains its orientation cache and warm starts."""
    from ..performance import emit
    config,parameters,ap=payload
    config.numerics.workers=1
    return analyze(config,parameters,ap,progress=emit)


def _parallel_analyze(config,p,ap,lengths,workers,progress,on_case):
    from ..performance import process_jobs
    cases=[(length,orientation,'full') for length in p.lengths_cm for orientation in p.orientations]
    if p.ablations:
        if p.reference_length_cm not in p.lengths_cm:cases.extend((p.reference_length_cm,o,'full') for o in p.orientations)
        for orientation in p.orientations:
            cases.extend((p.reference_length_cm,orientation,v) for v in ('without_gravity','without_saffman','without_lubrication'))
            if config.supply.wall_capture_rate_s:cases.append((p.reference_length_cm,orientation,'without_adhesion'))
    jobs=[];indices=[]
    for length in lengths:
        local=deepcopy(p);local.lengths_cm=[x for x in p.lengths_cm if x==length] or [length]
        local.ablations=p.ablations and length==p.reference_length_cm
        jobs.append((deepcopy(config),local,ap));indices.append([i for i,c in enumerate(cases) if c[0]==length])
    rows={};acq={};damage={};histories={};limitations=[]
    progress(f'Parallel length study: {workers} processes, {len(jobs)} length groups; cached flow within each group')
    for group,result in process_jobs(_length_job,jobs,workers,progress):
        remap={}
        for index,row in zip(indices[group],result['rows'],strict=True):
            old=row['case_id'];new=f'case_{index:03d}';remap[old]=new;row['case_id']=new;rows[index]=row
        for key,target in (('acquisition_rows',acq),('damage_scenarios',damage)):
            for row in result[key]:
                row['case_id']=remap[row['case_id']];target.setdefault(row['case_id'],[]).append(row)
        histories.update({remap[key]:value for key,value in result['histories'].items()})
        limitations=result['limitations']
        ordered=[rows[k] for k in sorted(rows)]
        on_case({'config':config.to_dict(),'parameters':asdict(p),'completed_cases':len(rows),'total_cases':len(cases),
            'rows':ordered,'acquisition_rows':[x for key in sorted(acq) for x in acq[key]],
            'damage_scenarios':[x for key in sorted(damage) for x in damage[key]]})
        progress(f'Parallel length study completed {len(rows)}/{len(cases)} cases')
    return {'config':config.to_dict(),'parameters':asdict(p),'acquisition_parameters':ap.to_dict(),
        'rows':[rows[k] for k in sorted(rows)],'acquisition_rows':[x for key in sorted(acq) for x in acq[key]],
        'damage_scenarios':[x for key in sorted(damage) for x in damage[key]],'histories':histories,'limitations':limitations,
        'execution':{'workers':workers,'groups':'length','native_threads_per_worker':1,'particle_backend':config.numerics.particle_backend}}


def export(result, directory):
    from ..report.export import clean, write_csv
    root = Path(directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in result.items() if k != "histories"}
    (root/"length_study.json").write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for key, filename in (("rows", "length_study.csv"), ("acquisition_rows", "acquisition.csv"), ("damage_scenarios", "mechanical_sensitivity.csv")):
        write_csv(root/filename, result[key])
    arrays = {f"{case}_{key}": value for case, history in result["histories"].items() for key, value in history.items()}
    np.savez_compressed(root/"long_tube_trajectories.npz", **arrays)
    figures(result, root)
    labels = {"horizontal": " Horizontal ", "down": " Vertical down ", "up": " Vertical up ", "horizontal_down": " Horizontal first half → Downward second half ", "down_horizontal": " Downward first half → Horizontal second half ", "custom": " Custom segments "}
    table = []
    for r in result["rows"]:
        if r["status"] != "ok":
            table.append(f'<tr><td>{r["length_cm"]:g}</td><td>{labels[r["orientation"]]}</td><td colspan="6"> Computation failed : {html.escape(r["error"])}</td></tr>');continue
        fmt = lambda x: " Unknown " if x is None else f"{x:.4g}"
        table.append(f'<tr><td>{r["length_cm"]:g}</td><td>{labels[r["orientation"]]} / {r["variant"]}</td><td>{fmt(r["actual_sample_flow_ul_min"])}</td><td>{fmt(r["mean_residence_s"])}</td><td>{fmt(r["wall_stress_pa"])}</td><td>{fmt(r["mean_shear_dose_pa_s"])}</td><td>{r["near_wall_exposed_fraction"]:.1%}</td><td>{fmt(r["nozzle_cells_s"])}</td></tr>')
    limits = ''.join(f'<li>{html.escape(s)}</li>' for s in result['limitations'])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title> Sample capillary length ,  Shear and single-source rates </title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#f5f8fa;color:#20344b}}img{{width:100%}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ccd}}section{{padding:20px;background:white;margin:20px 0}}</style>
<h1> Sample capillary length ,  Orientation ,  Shear and single-source event rates </h1><section><p> Boundary :  Sample  {html.escape(result['config']['supply']['sample_control'])};  Gauge pressure  {result['config']['supply']['sample_pressure_mbar']:g} mbar;  Constant sheath flow  {result['config']['fluid']['sheath_flow_ul_min']:g} μL/min.</p>
<p> Concentration  {result['config']['cell']['concentration_per_ml']:g} cells/mL;  Per condition  {result['parameters']['cells']}  Transport responses . Exited → Packet delivery fraction  {result['acquisition_parameters']['signal_delivery_fraction']:.0%},  Single-source packet width  {result['acquisition_parameters']['signal_width_ms']:g} ms  Configurable parameters .</p>
<p> Constant-pressure approximation  Q∝1/L,  Mean fluid renewal time ∝L²,  Wall shear stress ∝Q. Wall shear at constant flow is independent of length . Longer capillaries increase exposure ,  Transport fraction and source purity require separate evaluation .</p>
<p> Continuous injection rate  =  Concentration × Actual sample flow ;  Outlet rate  =  Injection rate × Transport fraction .MS  Acquisition rate also depends on actual  IT,  Scan cycle ,  Packet width and source overlap . Independent delays preserve  Poisson  Source timing statistics .</p></section>
<img src="length_physics.png"><img src="length_rates.png"><img src="tube_paths.png">
<section><h2> Full model and force ablations </h2><p> Wall-capture rate  {result['config']['supply']['wall_capture_rate_s']:g} /s;  Set to  0  Non-adhesive case ,  Finite-sample outlet fraction  100%. Mechanical-damage sensitivity model ,  Viability calibration external to this model .</p><p>without_gravity  Hydrostatic head and buoyant cell weight disabled together ;  Other ablations retain supply boundaries ,  Only the specified particle force changes . Signed differences .</p><table><tr><th> Capillary length  cm</th><th> Orientation  /  Model </th><th> Actual  Q μL/min</th><th> Mean transit time  s</th><th> Wall shear  Pa</th><th> Shear dose  Pa·s</th><th> Near-wall fraction </th><th> Exited  cells/s</th></tr>{''.join(table)}</table></section>
<section><h2> Data and model scope </h2><p><a href="length_study.csv"> Length and shear  CSV</a> · <a href="acquisition.csv"> Resolution settings  /  Actual  IT CSV</a> · <a href="mechanical_sensitivity.csv"> Mechanical-damage sensitivity </a> · <a href="long_tube_trajectories.npz"> 3D transverse trajectories </a> · <a href="length_study.json"> Parameters and diagnostics </a></p><ul>{limits}</ul></section></html>'''
    (root/"length_study_report.html").write_text(text, encoding="utf-8")
    return root


def figures(result, root):
    from ..visualization.figures import serialized_plotting
    return serialized_plotting(_figures)(result, root)


def _figures(result, root):
    from ..visualization.figures import plt
    colors = plt.get_cmap('tab10')
    base = [r for r in result["rows"] if r["status"] == "ok" and r["variant"] == "full"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    metrics = [("actual_sample_flow_ul_min", "Sample flow (uL/min)"), ("mean_residence_s", "Cell residence time (s)"),
               ("wall_stress_pa", "Developed wall shear (Pa)"), ("mean_shear_dose_pa_s", "Cell stress exposure (Pa s)"),
               ("mean_near_wall_time_s", "Near-wall exposure (s)"), ("outlet_r95_um", "Nozzle r95 (um)")]
    for color, orientation in enumerate(result["parameters"]["orientations"]):
        rows = sorted((r for r in base if r["orientation"] == orientation), key=lambda r: r["length_cm"])
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.plot([r["length_cm"] for r in rows], [r[key] if r[key] is not None else np.nan for r in rows], 'o-', color=colors(color), label=orientation)
            ax.set(xlabel="Inner tube length (cm)", ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Length / orientation / stress | sample pressure or flow control, sheath fixed flow')
    fig.savefig(root/'length_physics.png');fig.savefig(root/'length_physics.svg');plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for color, orientation in enumerate(result['parameters']['orientations']):
        rows = sorted((r for r in base if r['orientation'] == orientation), key=lambda r:r['length_cm'])
        for ax, key, ylabel in ((axes[0,0], 'nozzle_cells_s', 'Steady nozzle cells/s'), (axes[0,1], 'transport_fraction', 'Transport fraction')):
            ax.plot([r['length_cm'] for r in rows], [r[key] for r in rows], 'o-', color=colors(color), label=orientation)
            ax.set(xlabel='Inner tube length (cm)', ylabel=ylabel);ax.grid(alpha=.2)
        actual_it = max(result['acquisition_parameters']['actual_injection_times_ms'])
        for ax, resolution in zip([axes[0,2], *axes[1,:]], (17500,35000,70000,140000)):
            selected = sorted((r for r in result['acquisition_rows'] if r['variant']=='full' and r['orientation']==orientation and r['resolution_at_mz200']==resolution and r['actual_injection_ms']==actual_it), key=lambda r:r['length_cm'])
            ax.plot([r['length_cm'] for r in selected], [r['unique_pure_single_cell_events_s'] for r in selected], 'o-', color=colors(color))
            ax.set(xlabel='Inner tube length (cm)', ylabel='Unique pure opportunities/s', title=f'R={resolution:,}; actual IT scenario={actual_it:g} ms');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].set_ylim(0,1.05)
    fig.suptitle('Nozzle rate and Q Exactive Full MS | conditional opportunities, uncalibrated detection\n'
        f'C={result["config"]["cell"]["concentration_per_ml"]:g} cells/mL; delivery={result["acquisition_parameters"]["signal_delivery_fraction"]:g}; '
        f'packet width={result["acquisition_parameters"]["signal_width_ms"]:g} ms; wall capture={result["config"]["supply"]["wall_capture_rate_s"]:g}/s')
    fig.savefig(root/'length_rates.png');fig.savefig(root/'length_rates.svg');plt.close(fig)
    selected=[]
    for orientation in result['parameters']['orientations']:
        rows=[r for r in base if r['orientation']==orientation]
        if rows:selected.append(min(rows,key=lambda r:abs(r['length_cm']-result['parameters']['reference_length_cm'])))
    fig, axes=plt.subplots(max(1,len(selected)),2,figsize=(12,max(4,len(selected)*2.5)),squeeze=False,layout='constrained')
    radius=result['config']['geometry']['inner_id_um']/2
    for i,row in enumerate(selected):
        h=result['histories'].get(row['case_id'])
        if h is None:continue
        for j in range(min(12,len(h['ids']))):
            axes[i,0].plot(h['distance_m'][:,j]*100,h['x_m'][:,j]*1e6,lw=.8)
            axes[i,1].plot(h['x_m'][:,j]*1e6,h['y_m'][:,j]*1e6,lw=.8)
        theta=np.linspace(0,2*np.pi,200);axes[i,1].plot(radius*np.cos(theta),radius*np.sin(theta),'k-')
        axes[i,0].set(title=row['orientation'],xlabel='Path distance from inlet (cm)',ylabel='Transverse x (um)',ylim=(-radius,radius))
        axes[i,1].set(xlabel='x (um)',ylabel='y (um)',aspect='equal',xlim=(-radius,radius),ylim=(-radius,radius))
    fig.suptitle('Finite sphere center paths | cross-section settling; quartz wall and exclusion clearance')
    fig.savefig(root/'tube_paths.png');fig.savefig(root/'tube_paths.svg');plt.close(fig)
