import csv
import html
import json
from pathlib import Path
import numpy as np
from ..engine import derived
from ..visualization.figures import save_figures, save_animations, FILENAMES


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    return value


def write_csv(path, rows):
    rows = list(rows)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader(); writer.writerows(clean(rows))


def export(result, directory, animations=True, progress=lambda x: None):
    root = Path(directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    result.config.save(root/"config.json")
    summary = {"software": "CyESI Multiphysics Simulator 0.1.0", "derived": derived(result),
        "flow": result.flow.diagnostics, "solvent": result.solvent.diagnostics,
        "electric": result.electric.diagnostics, "particles": result.particles.diagnostics,
        "coupling": result.coupling,
        "events": {k: v for k, v in result.events.items() if not isinstance(v, np.ndarray)},
        "spray": None if result.spray is None else result.spray.diagnostics,
        "warnings": result.warnings}
    (root/"summary.json").write_text(json.dumps(clean(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    s, p, e = result.solvent, result.particles, result.electric
    ionic=getattr(s,"ionic",None)
    mechanical_fields = {f'cell_{key}': value for key,value in (p.stress or {}).items()}
    bulk_fields = {f'bulk_trajectory_{key}':value for key,value in (p.bulk_history or {}).items()}
    if p.bulk_history:
        np.savez_compressed(root/'long_tube_trajectories.npz', **p.bulk_history)
    liquid_electric={} if ionic is None else {"liquid_field_r_v_m":ionic.field_r,"liquid_field_z_v_m":ionic.field_z,"liquid_potential_v":ionic.potential}
    np.savez_compressed(root/"fields.npz", liquid_r_edges_m=s.radial_edges, liquid_z_edges_m=s.axial_edges,
        liquid_active=s.active, velocity_r_m_s=s.velocity_r, velocity_z_m_s=s.velocity_z,
        pressure_pa=s.pressure, methanol_fraction=s.concentration,
        electric_r_edges_m=e.radial_edges, electric_z_edges_m=e.axial_edges, potential_v=e.potential,
        field_r_v_m=e.field_r, field_z_v_m=e.field_z, electric_domains=e.domains,
        trajectory_ids=p.saved_ids, trajectory_time_s=p.trajectory_time,
        trajectory_r_m=p.trajectory_r, trajectory_z_m=p.trajectory_z,
        trajectory_methanol_fraction=p.trajectory_methanol,
        response_residence_time_s=p.residence_time,response_exited=p.exited,
        equivalent_stress_pa=s.equivalent_stress_pa,axial_extension_stress_pa=s.axial_extension_stress_pa,
        **mechanical_fields,**bulk_fields,**liquid_electric)
    np.savez_compressed(root/"mesh.npz", points_m=result.flow.mesh_data.mesh.p*result.flow.mesh_data.scale,
                        triangles=result.flow.mesh_data.mesh.t,
                        boundary_edges=result.flow.mesh_data.mesh.facets[:, result.flow.mesh_data.mesh.boundary_facets()])
    write_csv(root/"cells.csv", ({"cell_id": i, "initial_r_um": p.initial_radius[i]*1e6,
        "status": int(p.status[i]), "residence_s": p.residence_time[i],
        "tip_residence_s": p.tip_residence_time[i], "outlet_r_um": p.outlet_radius[i]*1e6,
        "outlet_methanol_fraction": p.outlet_methanol[i], "methanol_dose_fraction_s": p.methanol_dose[i],
        **{key:value[i] for key,value in (p.stress or {}).items()}}
        for i in range(result.config.cell.count)))
    write_csv(root/"events.csv", ({"event_id": int(result.events["event_id"][i]),
        "trajectory_template_id": int(result.events["template_id"][i]), "injection_s": a, "nozzle_s": b,
        "hypothetical_detection_s": None if result.events["hypothetical_detection_s"] is None else result.events["hypothetical_detection_s"][i]}
        for i, (a, b) in enumerate(zip(result.events["injection_s"], result.events["nozzle_s"])) ))
    write_csv(root/"exposure.csv", ({"cell_id": int(i), "time_s": p.trajectory_time[k, j],
        "r_um": p.trajectory_r[k, j]*1e6, "z_mm": p.trajectory_z[k, j]*1e3,
        "methanol_fraction": p.trajectory_methanol[k, j]}
        for j, i in enumerate(p.saved_ids) for k in range(len(p.trajectory_time))))
    if result.scan: write_csv(root/"scan.csv", result.scan)
    if result.spray is not None:
        sp = result.spray
        from ..spray.level_set import electrical_arrays
        np.savez_compressed(root/"spray.npz", radial_edges_m=sp.radial_edges, axial_edges_m=sp.axial_edges,
            time_s=sp.times, liquid_fraction=sp.phase, velocity_r_m_s=sp.velocity_r,
            velocity_z_m_s=sp.velocity_z, pressure_pa=sp.pressure, active=sp.active,
            velocity_r_history_m_s=sp.velocity_r_history,velocity_z_history_m_s=sp.velocity_z_history,
            pressure_history_pa=sp.pressure_history,**electrical_arrays(sp))
        write_csv(root/"droplets.csv", sp.droplets)
    ionic=getattr(s,"ionic",None)
    if ionic is not None:
        np.savez_compressed(root/"ions.npz",radial_edges_m=ionic.radial_edges,axial_edges_m=ionic.axial_edges,
            salt_mol_m3=ionic.salt_mol_m3,potential_v=ionic.potential,field_r_v_m=ionic.field_r,
            field_z_v_m=ionic.field_z,current_radial_a=ionic.current_r_a,current_axial_a=ionic.current_z_a,
            conductivity_s_m=ionic.conductivity,active=ionic.active)
    save_figures(result, root, progress)
    if animations:
        progress("Encoding computed trajectory / interface animations")
        save_animations(result, root)
    from ..acquisition.q_exactive import analyze as acquisition_analysis
    from ..acquisition.report import export as export_acquisition
    progress("Estimating Q Exactive Full MS sampling opportunities")
    export_acquisition(acquisition_analysis(result.events['steady_nozzle_cells_s'],
        source=str(root),source_config=result.config.to_dict()),root/'acquisition')
    _html(result, summary, root)
    return root


def _html(result, summary, root):
    p, c = result.particles.diagnostics, result.config
    def value(v, unit="", digits=3):
        return " Not computed " if v is None else f"{v:.{digits}g}{unit}"
    warnings = "".join(f"<li>{html.escape(x)}</li>" for x in result.warnings)
    figures = "".join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a><figcaption>{name}</figcaption></figure>' for name in FILENAMES)
    rows = "".join(f"<tr><td>{x['duration_s']:g} s</td><td>{x['injected']}</td><td>{x['nozzle_exits']}</td><td>{x['nozzle_exits']/x['duration_s']:.3f} cells/s</td><td>{x['detected'] if x['detected'] is not None else ' Collection efficiency unspecified '}</td></tr>" for x in result.events["summaries"])
    nozzle_rate=result.events.get("steady_nozzle_cells_s",result.events['rate_cells_s']*p['transport_fraction'])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CyESI numerical report</title><style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 22px;color:#20344b;background:#f5f7fa}}h1{{line-height:1.2}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card,section,figure{{background:white;padding:20px;border-radius:10px;margin:18px 0}}.card{{flex:1;min-width:180px}}strong{{font-size:25px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#edf1f4;padding:2px 5px}}</style>
<h1>CyESI Multiphysics Simulator</h1><p> Axisymmetric simulation report  ·  Parameters saved in  <a href="config.json">config.json</a>,  Diagnostics saved in  <a href="summary.json">summary.json</a>.</p>
<div class="cards"><div class="card"> Capillary volume  /  Flow rate <br><strong>{value(p['sample_volume_over_flow_s'],' s')}</strong></div><div class="card"> Mean transit time among exits <br><strong>{value(p['mean_residence_s'],' s')}</strong></div><div class="card"> Inlet event rate <br><strong>{result.events['rate_cells_s']:.2f} cells/s</strong></div><div class="card"> Outlet cell centres  r95<br><strong>{value(p['outlet_r95_um'],' μm')}</strong></div></div>
<section><h2> System and time reference </h2><p> Sample suspension : {html.escape(c.chemistry.sample_medium)};  Sample transport length  {c.geometry.sample_length_mm:g} mm;  Sheath methanol volume fraction  {c.chemistry.sheath_methanol_fraction:.0%}. Sheath outlet  z=0,  Sample recess  {c.geometry.inner_recess_mm:g} mm,  Gas recess  {c.geometry.gas_recess_mm:g} mm, MS  Ground plane  +{c.geometry.ms_gap_mm:g} mm.</p><p> Fully developed cylindrical capillary flow ,  Axisymmetric tip  P2/P1  Finite elements . Connected capillary and tip transit . Exit defined by sphere centre crossing the sheath outlet ;  Emitter to  MS  Transfer and capture require the carrier model ,  Emitter event rates reported separately from instrument detection .</p><p> Additional capillary pressure drop  {summary['derived']['developed_bulk_pressure_drop_pa']/1000:.3g} kPa. Cell state :  Exited  {p['exited']},  Wall contact  {p['wall_contacts']},  Censored or reversed  {p['censored_or_reverse']};  Mean times conditioned on exit .</p></section>
<section><h2> Continuous source and acquisition rates </h2><p> Steady model outlet rate  <strong>{nozzle_rate:.2f} cells/s</strong>,  Inlet rate multiplied by the exiting response fraction ;  Censored trajectories reported separately . Continuous supply ,  Continuous injection ,  Finite observation window for a continuous source . Startup mode : {html.escape(c.events.startup)}. Negative injection times represent prehistory .{c.cell.count}  Transport response sample count .MS  Carrier entry and membrane state available in the transfer model ;  Aperture entry metric .</p><table><tr><th> Duration </th><th> Injections in window </th><th> Emitter events </th><th> Window-averaged outlet rate </th><th> Specified detection model </th></tr>{rows}</table></section>
<section><h2> Numerical methods and model scope </h2><p> Axisymmetric internal flow  Navier–Stokes / Stokes  Weak form ;  Neutral methanol species ,  Advection-diffusion transport .resistive  Electroneutral binary-salt model  Nernst–Planck  Ambipolar diffusion closure ,  Conservative current and quartz electroosmotic slip . Electrode distance from outlet  {c.electrical.contact_distance_mm:g} mm;  Open-circuit limit when current is unspecified . Gas-liquid velocity ,  Pressure ,  Surface tension and  Maxwell  Stress ,  Recessed outlets included in the domain . Enabled  free_charge_enabled  Free-charge transport ,  Implicit conductive relaxation and upstream  RC  Circuit ,  Specified support conductivity . One-way pressure coupling to the steady internal solution ;  Full ionic reactions ,  Evaporation energy and  MS  Capture closure . Computed  Taylor  Cone and droplets .</p><ul>{warnings}</ul></section>
{figures}<section><h2> Source data </h2><p><a href="cells.csv"> All cells </a> · <a href="events.csv">Poisson  Events </a> · <a href="exposure.csv"> Exposure trajectories </a> · <a href="fields.npz"> Field data </a> · <a href="mesh.npz"> Mesh </a>{' · <a href="scan.csv"> Parameter sweep </a>' if result.scan else ''}</p><p>NPZ  Numerical arrays ,  Available  <code>numpy.load(..., allow_pickle=False)</code>  Load .PNG  Preview , SVG  Export . Steady spray assessment requires longer integrations and mesh convergence .</p></section>
<section><h2> Method references </h2><p><a href="https://scikit-fem.readthedocs.io/en/latest/">scikit-fem  Official documentation </a> · <a href="https://github.com/kinnala/scikit-fem/blob/master/docs/examples/ex18.py">Taylor–Hood  Mixed-formulation example </a> · <a href="https://doc.comsol.com/6.3/doc/com.comsol.help.particle/particle_ug_fluid_flow.08.42.html">Saffman  Lift-model domain </a> · <a href="https://doc.comsol.com/6.4/doc/com.comsol.help.polymer/polymer_ug_fluidflow_multi.07.22.html"> Two-phase equations and surface tension </a> · <a href="https://doc.comsol.com/6.3/doc/com.comsol.help.mems/mems_ug_acdc.06.096.html"> Charge relaxation and conduction </a></p></section></html>'''
    text=text.replace('</h1>','</h1><p><a href="acquisition/acquisition_report.html">Q Exactive  Resolution settings  / 30 ms Maximum IT:  Theoretical acquisition rates and source purity </a></p>',1)
    supply=summary['derived']
    pressure_note=f" Sample supply : {c.supply.sample_control};  Gauge pressure  {c.supply.sample_pressure_mbar:g} mbar;  Solved flow rate  {c.fluid.sample_flow_ul_min:.6g} μL/min;  Constant sheath flow  {c.fluid.sheath_flow_ul_min:g} μL/min;  Orientation  {c.supply.orientation};  Hydrostatic contribution  {supply['hydrostatic_assist_pa']:.5g} Pa."
    text=text.replace('</h1>','</h1><p>'+html.escape(pressure_note)+'</p>',1)
    if (root/'cell_trajectories.gif').exists():
        text=text.replace('</html>','<section><h2>Finite-sphere trajectories | time since entering resolved tip</h2><p> Finite-sphere response replay ;  Time since resolved-tip entry ,  Tip-domain transit time .</p><img src="cell_trajectories.gif" alt="Finite-sphere trajectories"><p><a href="cell_trajectories.gif"> Open trajectory animation </a></p></section></html>')
    (root/"report.html").write_text(text, encoding="utf-8")
