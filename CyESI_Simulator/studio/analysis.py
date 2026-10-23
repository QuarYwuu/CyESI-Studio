"""Read-only analysis views and result export.

Views use the current scene. Solver runs and parameter studies are submitted
as background jobs.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
import csv
import html
import json
import zipfile
import numpy as np

from ..scene3d.data import _clean


def overview(scene):
    model=scene.model; c=scene.data.config
    return _clean(dict(kind='overview', source_id=scene.source_id,
        dimension=3 if hasattr(scene,'volume') else 2, summary=model.summary(),
        rates=model.rates, acquisition_rows=model.theory.get('rows',[]),
        parameters=c.to_dict(), notes=[
            ' Source rate from actual sample flow  ×  And concentration ,  Response samples set numerical sampling density .',
            ' Radial focusing and axial ordering are distinct ;  Single-source acquisition is a signal-level metric .']))


def acquisition(scene):
    m=scene.model
    result=dict(kind='acquisition', source_id=scene.source_id, rows=m.theory.get('rows',[]),
        theory=m.theory, summary=m.summary(), timing=m.row,
        scans=m.scans, ledger=m.ledger[:2000], ledger_count=len(m.ledger),
        ledger_truncated=len(m.ledger)>2000,
        arrival_times_s=np.asarray(m.records['nozzle_s'])[np.isfinite(m.records['nozzle_s'])][:4000],
        notes=[' Resolution comparison at matched outlet rate ;  One source realization .',
               ' Acquisition windows ,  Transfer parameters ,  Complete / Partial / Mixed and missed events .'])
    study=getattr(scene,'interaction_result',None)
    if study is not None:
        from ..particle.interaction_analysis import acquisition_for_study
        result['local_interactions']=acquisition_for_study(m,study)
    return _clean(result)


def quality(scene):
    if not hasattr(scene,'volume'):
        from ..diagnostics.model_assessment import assess_model
        return _clean(assess_model(scene.data))
    v=scene.volume
    diag=v.diagnostics
    checks=[]
    for group in ('supply','flow','pressure','coupling','electric','solvent','species'):
        value=diag.get(group)
        if value is not None:
            checks.append(dict(label=group,status='pass' if isinstance(value,dict) and value.get('converged') is True else 'assumption',detail=value))
    checks.extend([
        dict(label=' 3D local field ',status='assumption',detail=diag),
        dict(label=' Capillary and Cartesian domains ',status='assumption',detail=' Hydraulic resistance and transverse particle transport ;  Cartesian grid near the emitter ,  Reduced capillary and local Cartesian  CFD.'),
        dict(label=' Experimental calibration  /  Mesh and time-step convergence ',status='missing',detail=' Numerical verification and experimental validation are separate ;  Parameter-specific convergence study .'),
        dict(label=' Experimental detection efficiency ',status='missing',detail=' Single-source  Full MS  Acquisition rate ,  Carrier entry and intact-cell status are separate metrics .')])
    phase=getattr(v,'phase',None)
    if phase is not None:checks.append(dict(label=' Interface charge and species conservation ',status='assumption',detail=phase.diagnostics))
    study=getattr(scene,'interaction_result',None)
    if study is not None:checks.append(dict(label=' Local cell interactions ',status='assumption',detail=study.diagnostics))
    return _clean(dict(status='assumption',summary=' Cartesian domain ,  Numerical diagnostics and specified parameters ',checks=checks,
        scope_note=' Local Cartesian grid  +  Reduced capillary model ;  Separate interface and transport clocks .'))


def interactions(scene):
    study=getattr(scene,'interaction_result',None)
    if study is None:return dict(kind='interactions',computed=False,notes=[' No interaction study .'])
    from ..particle.interaction_analysis import acquisition_for_study
    return _clean(dict(kind='interactions',computed=True,parameters=study.parameters.to_dict(),
        diagnostics=study.diagnostics,acquisition=acquisition_for_study(scene.model,study),
        time_s=study.time_s,events=study.events))


def comparison(service):
    from ..acquisition.q_exactive import analyze
    current,_=service.snapshot();acq=deepcopy(current.model.acquisition)
    acq.actual_injection_times_ms=[current.model.parameters.actual_it_ms]
    rows=[]
    for name,scene in list(service._scenes.items()):
        c=scene.data.config;m=scene.model
        theory=analyze(m.rates.get('nozzle_cells_s') or 0.,acq)
        method=next(row for row in theory['rows'] if row['resolution_at_mz200']==current.model.parameters.resolution)
        particles=scene.data.summary.get('particles',{})
        rows.append(dict(name=' Axisymmetric solution ' if name=='2d' else ' Cartesian solution ',mode=name,source_id=scene.source_id,
            sample_flow_ul_min=c.fluid.sample_flow_ul_min,sheath_flow_ul_min=c.fluid.sheath_flow_ul_min,
            concentration_per_ml=c.cell.concentration_per_ml,diameter_um=c.cell.diameter_um,
            length_cm=c.geometry.sample_length_mm/10,voltage_kv=c.electrical.voltage_kv,
            injection_rate_s=m.source_rate,nozzle_cells_s=m.rates.get('nozzle_cells_s'),
            mean_residence_s=particles.get('mean_residence_s'),outlet_r95_um=particles.get('outlet_r95_um'),
            unique_pure_single_cell_events_s=method['unique_pure_single_cell_events_s']))
    pinned=getattr(service,'pinned_metrics',None)
    if pinned is not None:rows.insert(0,pinned)
    return dict(kind='comparison',rows=_clean(rows),method=dict(resolution=current.model.parameters.resolution,
        actual_it_ms=current.model.parameters.actual_it_ms),notes=[
            ' Each solution retains its parameters ;  Different flow conditions form separate experiments ,  Compare matched conditions for dimensional studies .',
            ' Shared acquisition settings ;  Reference retains its acquisition settings .'])


def pin(service,body):
    if body.get('clear') is True:
        service.pinned_metrics=None
    else:
        current,_=service.snapshot();mode='3d' if hasattr(current,'volume') else '2d'
        item=next(row for row in comparison(service)['rows'] if row['mode']==mode and row['source_id']==current.source_id)
        service.pinned_metrics=dict(item,name=' Saved reference  · '+str(body.get('name') or item['name'])[:80],
            resolution=current.model.parameters.resolution,actual_it_ms=current.model.parameters.actual_it_ms)
    return dict(accepted=True,comparison=comparison(service))


def get_analysis(service,kind='overview'):
    scene,revision=service.snapshot()
    if kind=='comparison':return comparison(service)
    if kind in ('size','length','scan'):
        saved=getattr(service,'study_results',{}).get(kind)
        if saved is None:
            from .artifacts import archived_study
            saved=archived_study(service,kind)
        return _clean(saved or dict(kind=kind,computed=False,rows=[],notes=[' Run a parameter study to view results .']))
    readers=dict(overview=overview,acquisition=acquisition,quality=quality,interactions=interactions)
    if kind not in readers:raise ValueError("Unknown analysis page")
    if not hasattr(service,'analysis_cache'):service.analysis_cache={}
    key=(revision,kind)
    with service.lock:cached=service.analysis_cache.get(key)
    if cached is None:
        cached=readers[kind](scene)
        with service.lock:
            service.analysis_cache[key]=cached
            if len(service.analysis_cache)>16:
                service.analysis_cache.pop(next(iter(service.analysis_cache)))
    return cached


def _csv(path,rows):
    rows=_clean(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or ['event_id'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def export_report(service,body):
    # A frozen scene is exported; no time-dependent UI state is read later.
    scene,revision=service.snapshot()
    def work(progress):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        root=service.studio_root/'reports'/stamp;root.mkdir(parents=True)
        progress(' Exporting parameters ,  Field sections and event ledger ')
        c=scene.data.config;c.save(root/'config.json')
        record=dict(metadata=scene.metadata(),revision=revision,overview=overview(scene),
            acquisition=acquisition(scene),quality=quality(scene),interactions=interactions(scene),
            studies=getattr(service,'study_results',{}),comparison=comparison(service))
        (root/'results.json').write_text(json.dumps(_clean(record),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        _csv(root/'cell_events.csv',scene.model.ledger)
        r=scene.model.records
        _csv(root/'continuous_feed.csv',[dict(event_id=int(i),injection_s=float(t),response_index=int(k),nozzle_s=float(z))
            for i,t,k,z in zip(r['event_id'],r['injection_s'],r['template_id'],r['nozzle_s'])])
        _csv(root/'acquisition.csv',scene.model.theory.get('rows',[]))
        _csv(root/'comparison.csv',record['comparison']['rows'])
        from .slices import field_slice
        from ..visualization.figures import serialized_plotting
        @serialized_plotting
        def plots():
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fields=['speed','pressure','methanol','electric']
            fig=Figure(figsize=(12,7),layout='constrained');FigureCanvasAgg(fig)
            for ax,name in zip(fig.subplots(2,2).flat,fields):
                cut=field_slice(scene,name,'x',0.)
                values=np.array([[np.nan if v is None else v for v in row] for row in cut['values']])
                mesh=ax.pcolormesh(np.array(cut['horizontal_edges_m'])*1000,
                    np.array(cut['vertical_edges_m'])*1e6,values,cmap='viridis',shading='flat')
                ax.set(xlabel='z (mm)',ylabel='transverse (um)',title=name+' / '+cut['scale']['unit'])
                fig.colorbar(mesh,ax=ax)
            fig.savefig(root/'fields.png',dpi=160)
            fig=Figure(figsize=(12,5),layout='constrained');FigureCanvasAgg(fig)
            axes=fig.subplots(1,2)
            widths=[];delays=[]
            for track in scene.model.tracks:
                good=np.isfinite(track['t'])&np.isfinite(track['C'])
                if good.any():axes[0].plot(np.asarray(track['t'])[good],np.asarray(track['C'])[good],alpha=.2,color='#168d87')
                delays.append(track['end_time'])
            axes[0].set(xlabel='Time since entry (s)',ylabel='Methanol fraction',ylim=(0,1))
            axes[1].hist(np.asarray(delays)[np.isfinite(delays)],bins=25,color='#168d87')
            axes[1].set(xlabel='Residence time (s)',ylabel='Saved responses')
            fig.savefig(root/'transport.png',dpi=160)
        plots()
        progress(' Generating report files ')
        title='CyESI Studio ·  Multiphysics workspace '
        summary=scene.model.summary()
        cards=''.join('<div><small>'+label+'</small><strong>'+html.escape(str(_clean(summary.get(key))))+'</strong></div>'
            for key,label in [('injection_rate_s',' Input rate  / s⁻¹'),('full_population_nozzle_rate_s',' Predicted outlet rate  / s⁻¹'),('pure_recovery_of_nozzle_cohort',' Single-source acquisition fraction ')])
        table=''.join('<tr><td>'+html.escape(str(x['label']))+'</td><td>'+html.escape(str(x['status']))+'</td><td>'+html.escape(str(x.get('detail','')))+'</td></tr>' for x in record['quality']['checks'])
        text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CyESI Studio</title>
<style>body{font:15px/1.8 system-ui;color:#183544;background:#f1f6f6;max-width:1100px;margin:40px auto;padding:0 24px}section,.cards>div{background:white;padding:24px;border:1px solid #dce6e7;border-radius:16px;margin:16px 0}h1{font-size:34px}small{color:#627b89}strong{display:block;font-size:28px}.cards{display:flex;gap:16px}.cards>div{flex:1}img{width:100%}td{border-bottom:1px solid #e2ebee;padding:12px;word-break:break-word}a{color:#087d74}table{width:100%;table-layout:fixed}</style>'''
        text+=f'<h1>{title}</h1><p> Record  {stamp} · '+(' Local Cartesian field ' if hasattr(scene,'volume') else ' Axisymmetric fields ')+'</p><div class="cards">'+cards+'</div>'
        text+='<section><h2> Simulation source </h2><p>'+html.escape(scene.data.source)+'</p><p> Current solution and continuous events ,  Uses cached  CFD. Single-source acquisition ,  Carrier entry and intact-cell metrics .</p></section>'
        text+='<section><h2> Computed fields </h2><img src="fields.png" alt="computed fields"></section><section><h2> Transport and solvent exposure </h2><img src="transport.png" alt="transport"></section>'
        text+='<section><h2> Methods and numerical checks </h2><table>'+table+'</table></section><section><h2> Source data </h2>'
        text+=' · '.join(f'<a href="{filename}">{label}</a>' for filename,label in [('config.json',' Parameters '),('results.json',' Analysis and provenance '),('continuous_feed.csv',' Cell arrivals '),('cell_events.csv',' Acquisition ledger '),('acquisition.csv',' Resolution settings ')])+'</section></html>'
        (root/'report.html').write_text(text,encoding='utf-8')
        with zipfile.ZipFile(root/'report.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in root.iterdir():
                if path.name!='report.zip':archive.write(path,path.name)
        service.export_records.insert(0,dict(id=stamp,revision=revision,source_id=scene.source_id,
            report_url=f'/downloads/{stamp}/report.html',zip_url=f'/downloads/{stamp}/report.zip',path=str(root)))
        return None
    return service.start('export',work)
