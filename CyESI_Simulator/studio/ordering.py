"""Long-capillary population studies and saved study records."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hashlib
import json
import numpy as np
from ..particle.collective_transport import CollectiveParameters,CollectiveResult,simulate
from ..scene3d.data import _clean
from ..workbench.compute import digest


def source_signature(scene,parameters):
    records=scene.model.records
    mask=np.asarray(records['injection_s'])<parameters.start_s+parameters.duration_s
    return digest(dict(source=scene.source_id,rate=scene.model.source_rate,
        ids=np.asarray(records['event_id'])[mask].tolist(),
        attempts=np.asarray(records['injection_s'])[mask].tolist()))


def current(service):
    scene,_=service.snapshot()
    study=getattr(service,'ordering_studies',{}).get(scene.source_id)
    if study is not None and study.parameters.start_s+study.parameters.duration_s>scene.model.parameters.duration_s:return None
    if study is not None and study.diagnostics.get('source_fingerprint')!=source_signature(scene,study.parameters):return None
    return study


def overview(service):
    from ..fluid.supply import segments
    scene,_=service.snapshot();c=scene.data.config;study=current(service)
    R=c.cell.diameter_um*.5e-6;a=c.geometry.a;u=c.fluid.qs/(np.pi*a*a)
    rate=scene.model.source_rate
    result=dict(computed=study is not None,source_id=scene.source_id,
        parameters=CollectiveParameters().to_dict() if study is None else study.parameters.to_dict(),
        scales=dict(length_cm=c.geometry.sample_length_mm/10,diameter_um=c.cell.diameter_um,
            inner_id_um=c.geometry.inner_id_um,actual_sample_flow_ul_min=c.fluid.sample_flow_ul_min,
            concentration_per_ml=c.cell.concentration_per_ml,source_rate_cells_s=rate,
            mean_fluid_speed_mm_s=u*1e3,reynolds=c.fluid.sample_density*u*2*a/c.fluid.sample_viscosity,
            particle_reynolds=c.fluid.sample_density*u*2*a/c.fluid.sample_viscosity*(R/a)**2,
            nominal_spacing_um=None if not rate else u/rate*1e6),
        notes=[' Stochastic source arrivals ;  Arrival intervals follow velocities ,  Coupling and contact ,  Computed output statistics Poisson.',
               ' Long-capillary domain → Resolved-tip entrance . Updated positions / Tip continuation uses the updated inlet state ,  Separate downstream MS Detection rate .',
               ' Independent reference ,  Tip studies and 2D / 3D fields retained ,  System-level reference metrics .'],
        saved_records=catalog(service),explanations=[
            dict(name=' Confined hydrodynamics ',effect=' Wall-reflected disturbances ,  Modify neighbour velocities ;  Spacing may decrease or increase .',implementation=' Regularized cylindrical wall Stokes Collocation ＋ Self-mobility calibration ,  Reciprocal and dissipative mobility ;  Finite wall patch and spherical blob Approximation .'),
            dict(name=' Near-field lubrication  /  Finite-size exclusion ',effect=' Squeeze resistance slows approach ;  Finite-sphere non-overlap constraint . No prescribed equilibrium spacing .',implementation=' Implicit normal squeeze resistance ,  Separate exclusion and binding models .'),
            dict(name=' Deformation and membrane mechanics ',effect=' Shape stress modifies reflected disturbances and neighbour velocities .',implementation=' Optional small-strain Kelvin Tensor ＋ Stress feedback ;  Specified effective membrane modulus ,  Spherical contact envelope .'),
            dict(name=' Finite inertia ',effect=' Disturbance convection modifies Stokes Symmetry ,  Affects migration and pair dynamics .',implementation=' First-order linearized convection Born Volume integral ,  Restricted to Re≤0.5;  Full finite Re Particle wakes and train formation require separate validation .'),
            dict(name=' Arrival statistics ',effect=' Stochastic feed ,  Clustered or nonstationary arrivals ;  Output intervals follow the transport solution .',implementation=' Current source IDs / Time ,  All event crossings at each station ;  Source model Poisson,  Export for comparison with measurements .')])
    if study is not None:
        result.update(diagnostics=study.diagnostics,analysis=study.analysis,
            time_range_s=[float(study.time_s[0]),float(study.time_s[-1])],
            record=getattr(study,'record',None),files=getattr(study,'files',[]))
    return _clean(result)


def run(service,body):
    if 'load_id' in body:return load(service,body['load_id'])
    if body.get('clear') is True:
        scene,_=service.snapshot()
        def clear(progress):
            getattr(service,'ordering_studies',{}).pop(scene.source_id,None)
            return None
        return service.start('ordering',clear)
    scene,_=service.snapshot()
    if set(body)-set(CollectiveParameters.__dataclass_fields__):raise ValueError("Unknown population parameter")
    p=CollectiveParameters(**body).validate()
    if p.start_s+p.duration_s>scene.model.parameters.duration_s:raise ValueError("Study window exceeds source duration")
    fingerprint=source_signature(scene,p)
    key=digest(dict(model_version=2,source=fingerprint,parameters=p.to_dict()))
    def work(progress):
        if not hasattr(service,'ordering_cache'):service.ordering_cache={}
        if not hasattr(service,'ordering_studies'):service.ordering_studies={}
        if key in service.ordering_cache:
            progress(' Loading matching population study ,  Current fields / Population ')
            service.ordering_studies[scene.source_id]=service.ordering_cache[key]
            return None
        from threadpoolctl import threadpool_limits
        # Small neighbour matrices are faster without nested BLAS thread pools.
        with threadpool_limits(limits=1,user_api='blas'):
            study=simulate(scene,p,progress)
        study.diagnostics['source_fingerprint']=fingerprint
        progress(' Exporting spacing  /  Crossing events  /  Diagnostics and figures ')
        root=service.studio_root/('collective_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        study.save(root);scene.data.config.save(root/'config.json')
        from .analysis import _csv
        rows=[]
        for j,event in enumerate(study.event_ids):
            row=dict(event_id=int(event),injection_s=float(study.injection_s[j]))
            for k in range(5):
                row[f'coupled_crossing_{k}_s']=study.crossings_s[j,k]
                if study.baseline_crossings_s is not None:row[f'independent_crossing_{k}_s']=study.baseline_crossings_s[j,k]
            rows.append(row)
        _csv(root/'crossings.csv',rows)
        _figures(study,root,scene.data.config)
        files={path.name:dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
               for path in root.iterdir() if path.is_file()}
        manifest=dict(schema_version=1,status='complete',source_id=scene.source_id,files=files)
        (root/'record.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        progress(' Population record complete ;  Current flow solution ')
        study.record=str(root);study.files=[dict(name=name,url=f'/ordering-files/{root.name}/{name}') for name in files]
        service.ordering_cache[key]=study;service.ordering_studies[scene.source_id]=study
        if len(service.ordering_cache)>4:service.ordering_cache.pop(next(iter(service.ordering_cache)))
        return None
    return service.start('ordering',work)


def catalog(service):
    """List completed local records only; never accept a client file path."""
    root=service.studio_root
    if not root.is_dir():return []
    rows=[]
    for path in sorted(root.glob('collective_*/record.json'),reverse=True):
        if not path.resolve().is_relative_to(root.resolve()):continue
        try:
            value=json.loads(path.read_text(encoding='utf-8'))
            if value.get('status')=='complete':
                rows.append(dict(id=path.parent.name,source_id=value['source_id']))
        except (ValueError,OSError,KeyError):continue
    return rows


def load(service,record_id):
    row=next((r for r in catalog(service) if r['id']==record_id),None)
    scene,_=service.snapshot()
    if row is None or row['source_id']!=scene.source_id:raise ValueError("This record belongs to another experiment. Load the corresponding simulation.")
    root=(service.studio_root/record_id).resolve()
    def work(progress):
        progress(' Verifying record hashes and source events ')
        record=json.loads((root/'record.json').read_text(encoding='utf-8'))
        for name,entry in record['files'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError(' Population record integrity check failed : '+name)
        study=CollectiveResult.load(root)
        if study.parameters.start_s+study.parameters.duration_s>scene.model.parameters.duration_s:raise ValueError("Extend the source duration first")
        mask=np.asarray(scene.model.records['injection_s'])<study.parameters.start_s+study.parameters.duration_s
        if not np.array_equal(study.event_ids,np.asarray(scene.model.records['event_id'])[mask]) or not np.array_equal(study.injection_s,np.asarray(scene.model.records['injection_s'])[mask]):
            raise ValueError("Source seed or warm-up does not match the record")
        study.diagnostics['source_fingerprint']=source_signature(scene,study.parameters)
        study.record=str(root);study.files=[dict(name=name,url=f'/ordering-files/{root.name}/{name}') for name in record['files']]
        progress(' Loading verified population record ,  Using the current ')
        if not hasattr(service,'ordering_studies'):service.ordering_studies={}
        service.ordering_studies[scene.source_id]=study
        return None
    return service.start('ordering',work)


def _figures(study,root,config):
    from ..visualization.figures import serialized_plotting
    @serialized_plotting
    def draw():
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(12,8),layout='constrained');FigureCanvasAgg(fig)
        ax=fig.subplots(2,2);colors={'coupled':'#008c86','independent':'#b67b35'}
        for mode,color in colors.items():
            rows=study.analysis['planes']
            cv=[row.get(mode,{}).get('cv') for row in rows]
            ax[0,0].plot([r['fraction'] for r in rows],cv,'o-',label=mode,color=color)
            last=rows[-1].get(mode,{})
            gaps=last.get('intervals_s',[])
            if gaps:ax[0,1].hist(gaps,bins=20,alpha=.45,label=mode,color=color)
        ax[0,0].axhline(1.,ls=':',color='#788995',label='Poisson reference only')
        ax[0,0].set(xlabel='Fraction of long-pipe length',ylabel='CV of crossing intervals')
        ax[0,0].legend(fontsize=8)
        ax[0,1].set(xlabel='Bulk-outlet interval (s)',ylabel='Actual crossings')
        if ax[0,1].patches:ax[0,1].legend(fontsize=8)
        t=study.time_s;start=config.geometry.inner_tip-config.geometry.sample_length_mm*.001
        # Selected saved curves are a display sample only; statistical panels
        # above always use all physical crossings in the observation window.
        for j in np.linspace(0,max(0,len(study.event_ids)-1),min(30,len(study.event_ids)),dtype=int):
            good=(study.states[:,j]==1)&np.isfinite(study.positions_m[:,j]).all(axis=1)
            if good.any():
                x=study.positions_m[good,j]
                ax[1,0].plot((x[:,2]-start)*1000,x[:,0]*1e6,alpha=.45,lw=.7,color='#008c86')
                ax[1,1].plot(t[good],study.shape_strain[good,j],alpha=.5,color='#008c86')
        ax[1,0].set(xlabel='Long-pipe station (mm)',ylabel='Transverse x (um)')
        ax[1,1].set(xlabel='Common source clock (s)',ylabel='Conditional shape strain')
        fig.suptitle('Common-clock long-capillary study | reduced confined model')
        fig.savefig(root/'collective_comparison.png',dpi=150)
        fig2=Figure(figsize=(11,4.3),layout='constrained');FigureCanvasAgg(fig2)
        axs=fig2.subplots(1,2)
        for mode,color in colors.items():
            pair=study.analysis.get('pair_dynamics',{}).get(mode,{})
            bins=[b for b in pair.get('bins',[]) if b['count'] and b['gap_high_d'] is not None]
            if bins:
                axs[0].plot([(b['gap_low_d']+b['gap_high_d'])/2 for b in bins],
                    [b['mean_relative_speed_um_s'] for b in bins],'o-',color=color,label=mode)
            gaps=pair.get('axial_gaps_in_diameters',[])
            if gaps:axs[1].hist(np.log10(np.maximum(gaps,1e-8)),bins=35,alpha=.45,color=color,label=mode)
        axs[0].axhline(0.,ls=':',color='#788995');axs[0].set_xscale('log')
        axs[0].set(xlabel='Same-lane centre separation / cell diameter',ylabel='Mean front minus rear speed (um/s)')
        axs[1].set(xlabel='log10(axial neighbour spacing / diameter)',ylabel='Saved neighbour-pair samples')
        for a in axs:
            if a.get_legend_handles_labels()[0]:a.legend(fontsize=8)
        fig2.suptitle('Observed pair dynamics | repeated frames are correlated samples')
        fig2.savefig(root/'pair_dynamics.png',dpi=150)
    draw()


def frame(service,time_s):
    from ..fluid.supply import segments
    scene,_=service.snapshot();study=current(service)
    if study is None:raise ValueError("Compute a population study for this source")
    start=scene.data.config.geometry.inner_tip-scene.data.config.geometry.sample_length_mm*.001
    result=dict(source_id=study.source_id,time_s=float(time_s),range_s=[float(study.time_s[0]),float(study.time_s[-1])],
        coupled=study.frame(time_s),independent=study.frame(time_s,True),
        start_z_m=start,end_z_m=scene.data.config.geometry.liquid_start,
        radius_m=scene.data.config.geometry.a,cell_radius_m=scene.data.config.cell.diameter_um*.5e-6)
    route=segments(scene.data.config);boundaries=np.array([row['end_m'] for row in route])
    for mode in ('coupled','independent'):
        for row in result[mode]:
            point=np.asarray(row['local_xyz_m']);row['global_xyz_m']=scene._global(point).tolist()
            segment=route[min(int(np.searchsorted(boundaries,point[2]-start,side='right')),len(route)-1)]
            gravity=np.array([segment['gx'],segment['gy']]);norm=np.linalg.norm(gravity)
            row['transverse_down_m']=float(point[:2]@gravity/norm) if norm>0 else float(point[0])
    return _clean(result)
