"""Cached parameter studies using background jobs."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
from ..scene3d.data import _clean
from ..workbench.compute import digest


def _scan_figures(rows,root,parameter=None):
    """Scientific values from completed cases, including visible failures."""
    import numpy as np
    from ..visualization.figures import serialized_plotting
    @serialized_plotting
    def render():
        import matplotlib.pyplot as plt
        successful=[(i,row) for i,row in enumerate(rows) if row.get('status')=='ok']
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        fig.get_layout_engine().set(rect=(0.,.05,1.,.95))
        specifications=[('steady_nozzle_rate_cells_s','Resolved nozzle rate (cells/s)'),
            ('unique_pure_single_cell_events_s','Unique pure-origin\nopportunities (events/s)'),
            ('mean_residence_s','Mean residence among exits (s)'),
            ('outlet_r95_um','Outlet centre r95 (um)')]
        labels=[str(row.get(parameter,'')) if parameter else str(i+1) for i,row in enumerate(rows)]
        horizontal={'sheath_flow_ul_min':'Sheath flow (uL/min)',
            'sample_flow_ul_min':'Sample flow (uL/min)',
            'voltage_kv':'Applied voltage (kV)',
            'gas_pressure_kpa':'Gas pressure (kPa)'}.get(parameter,'Numerical case')
        for ax,(key,label) in zip(axes.flat,specifications):
            values=[float(row[key]) if row.get(key) is not None else np.nan for _,row in successful]
            positions=[i for i,_ in successful]
            ax.plot(positions,values,'o-',color='#167b88',lw=1.8,ms=6)
            ax.set(xticks=range(len(rows)),xticklabels=labels,xlabel=horizontal,ylabel=label)
            ax.grid(axis='y',alpha=.22)
            for i,row in enumerate(rows):
                if row.get('status')!='ok':
                    ax.axvspan(i-.15,i+.15,color='#d26369',alpha=.18)
                    ax.text(i,.02,'failed',rotation=90,transform=ax.get_xaxis_transform(),ha='center',color='#9a3740')
        fig.suptitle('Independent axisymmetric parameter study | same selected acquisition scenario',fontsize=13)
        fig.text(.01,.005,'Pure-origin acquisition opportunities use assumed packet width and delivery; actual MS identification is uncalibrated.',fontsize=8,color='#536170')
        fig.savefig(root/'parameter_scan.png',dpi=150)
        fig.savefig(root/'parameter_scan.pdf')
        plt.close(fig)
    render()


def run_study(service,body):
    kind=body.get('kind','size')
    if kind not in ('size','length','scan'):raise ValueError("Unknown parameter study")
    scene,_=service.snapshot();config=deepcopy(scene.data.config)
    cells=body.get('cells',64)
    if isinstance(cells,bool) or not isinstance(cells,int) or not 16<=cells<=10000:
        raise ValueError("Response sample count must be an integer from 16 to 10000.")
    config.numerics.workers=body.get('workers',1)
    if kind=='size':
        from ..optimization.size_study import SizeParameters
        p=SizeParameters(diameters_um=body.get('values',[.5,2.,8.,20.]),cells=cells,
            saved_trajectories=min(cells,40),conditional_inlet=body.get('conditional_inlet',False))
        p.validate(config)
    elif kind=='length':
        from ..optimization.length_study import StudyParameters
        p=StudyParameters(lengths_cm=body.get('values',[20.,40.,70.,80.,100.]),
            orientations=body.get('orientations',['horizontal','up','down_horizontal']),
            cells=cells,ablations=body.get('ablations',True),reference_length_cm=config.geometry.sample_length_mm/10)
        p.validate()
    else:
        p=None
        if not isinstance(body.get('full',False),bool):raise ValueError("full must be a boolean.")
        parameter=body.get('parameter')
        from ..optimization.parameter_scan import SWEEPS
        if parameter is not None:
            if parameter not in SWEEPS:raise ValueError("Unknown sweep parameter")
            values=body.get('values',SWEEPS[parameter])
            if not isinstance(values,list) or not 1<=len(values)<=100:raise ValueError("A sweep must contain 1 to 100 numeric values.")
            import math
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                raise ValueError("Sweep values must be finite numbers")
            for value in values:
                trial=deepcopy(config)
                setattr(trial.electrical if parameter=='voltage_kv' else trial.fluid,parameter,value)
                trial.validate()
    config.validate()
    options=dict(kind=kind,config=config.to_dict(),body=body,acquisition=scene.model.acquisition.to_dict())
    key=digest(options)
    def work(progress):
        if not hasattr(service,'study_cache'):service.study_cache={}
        if key in service.study_cache:
            progress(' Loading saved study ;  Using cached results ')
            service.study_results[kind]=service.study_cache[key]
            return None
        root=service.studio_root/'studies'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+kind)
        root.mkdir(parents=True)
        progress(' Axisymmetric sensitivity study :  Independent of current 2D / 3D field ')
        ap=deepcopy(scene.model.acquisition)
        if kind=='size':
            from ..optimization.size_study import analyze,save_data
            result=analyze(config,p,ap,progress)
            save_data(result,root,source=scene.data.source)
            from ..visualization.size_comparison import render
            from ..visualization.figures import serialized_plotting
            serialized_plotting(render)(result,root,progress)
            payload=json.loads((root/'size_summary.json').read_text(encoding='utf-8'))
        elif kind=='length':
            from ..optimization.length_study import analyze,export
            result=analyze(config,p,ap,progress)
            export(result,root)
            payload=_clean(result)
        else:
            from ..optimization.parameter_scan import scan, _evaluate, SWEEPS
            if body.get('parameter') is not None:
                axis=body['parameter'];shared={};rows=[]
                base={name:getattr(config.electrical if name=='voltage_kv' else config.fluid,name) for name in SWEEPS}
                values=body.get('values',SWEEPS[axis])
                for i,value in enumerate(values):
                    progress(f' Parameter sweep  {axis}: {i+1}/{len(values)} · {value:g}')
                    row=_evaluate(config,axis,base|{axis:float(value)},cells,shared,progress)
                    rows.append(row)
            else:
                rows=scan(config,full=body.get('full',False),cells=cells,progress=progress)
            from ..acquisition.q_exactive import analyze as acquisition_analysis
            for row in rows:
                if row.get('status')=='ok':
                    theory=acquisition_analysis(row['steady_nozzle_rate_cells_s'],ap)
                    chosen=next((r for r in theory['rows'] if r['resolution_at_mz200']==scene.model.parameters.resolution and abs(r['actual_injection_ms']-scene.model.parameters.actual_it_ms)<1e-8),None)
                    row['unique_pure_single_cell_events_s']=None if chosen is None else chosen['unique_pure_single_cell_events_s']
            payload=dict(rows=rows,config=config.to_dict())
            from .analysis import _csv
            _csv(root/'scan.csv',rows)
            _scan_figures(rows,root,body.get('parameter'))
        payload=_clean(dict(kind=kind,computed=True,model_dimension=2,
            base_source_id=scene.source_id,**payload,
            notes=[' Axisymmetric parameter study ;  Axisymmetric parameter sweep .',
                   ' Size studies share the background flow ;  Matching studies use the session cache .'],directory=str(root)))
        (root/'studio_study.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        from .artifacts import study_catalog
        record=next(x for x in study_catalog(service) if Path(x['path'])==root.resolve())
        payload.update(record=record,figures=[x for x in record['files'] if x['type']=='png'])
        service.study_cache[key]=payload;service.study_results[kind]=payload
        if len(service.study_cache)>8:service.study_cache.pop(next(iter(service.study_cache)))
        return None
    return service.start('study_'+kind,work)
