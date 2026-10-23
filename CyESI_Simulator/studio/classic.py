"""Preserve the original 2-D figures and Pillow visual laboratory.

These read existing numerical responses. Their old response-replay clock is
labelled separately from Studio's continuous population clock.
"""
from __future__ import annotations
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from datetime import datetime
from threading import RLock
import numpy as np

LABELS={
 'figure_01_geometry':' Geometry and outlet positions ',
 'figure_02_velocity_pressure':' Velocity  /  Pressure ',
 'figure_03_focusing_shear':' Hydrodynamic focusing  /  Shear stress ',
 'figure_04_trajectories':' Finite-sphere trajectories  /  Inlet and outlet distributions ',
 'figure_05_residence_events':' Residence time  /  Continuous events ',
 'figure_06_methanol':' Methanol exposure  /  Solvent transport ',
 'figure_07_electric':' Potential  /  Electric field ',
 'figure_08_two_phase':' Two-phase tip transient ',
 'figure_09_optimization':' Parameter sweep  /  Optimization studies ',
 'figure_10_gas':' Nitrogen velocity  /  Pressure ',
 'cell_trajectories':' Cell trajectory animation ', 'two_phase':' Interface animation ', 'report':' Simulation report '}


def source_scene(service):
    with service.lock:scene=service._scenes.get('2d')
    if scene is None:raise ValueError(' Load or compute axisymmetric results ,  Then open the figure gallery ')
    return scene


def catalog(service):
    scene=source_scene(service);source=Path(scene.data.source)
    generated=getattr(service,'classic_records',{}).get(scene.source_id)
    root=Path(generated) if generated is not None else source
    items=[]
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.stem in LABELS and path.suffix in ('.png','.gif','.html'):
                items.append(dict(name=path.name,label=LABELS[path.stem],kind='animation' if path.suffix=='.gif' else 'report' if path.suffix=='.html' else 'figure',url=f'/classic-files/{scene.source_id}/{path.name}'))
    if not hasattr(service,'classic_sources'):service.classic_sources={}
    service.classic_sources[scene.source_id]=root.resolve()
    return dict(source_id=scene.source_id,source=str(root),mode='2d',archived=generated is None,items=items,
        notes=[' Saved gallery and  GIF  Source record and response clock ;  Continuous cells in “ 2D process view ”.',
               ' Gallery uses its saved parameters ;  Update figures from current arrays ,  Uses cached  CFD.',
               ' Reference cone ,  Droplet and lysis reference models ,  See analytical references in Methods .'])


def render_gallery(service,body):
    scene=source_scene(service)
    def work(progress):
        from .classic_figures import render_existing
        root=service.studio_root/'classic'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        render_existing(scene.data,root,progress)
        if not hasattr(service,'classic_records'):service.classic_records={}
        service.classic_records[scene.source_id]=str(root)
        catalog(service)
        return None
    return service.start('classic_figures',work)


def render_scene(service,options):
    from ..visualization.explorer_scene import SceneRenderer
    from ..visualization.explorer_data import ExplanationSettings
    from ..scene3d.server import _number
    scene=source_scene(service)
    page=options.get('page','device')
    if page not in ('device','nozzle'):raise ValueError(' Unknown 2D view ')
    width=int(_number(options.get('width',1280),'image width',900,1800))
    height=int(_number(options.get('height',760),'image height',560,1200))
    progress=_number(options.get('progress',.78),'replay progress',0.,1.)
    phase=_number(options.get('phase',0.),'demonstration clock',0.,1e7)
    selected=options.get('id',scene.model.default_event)
    index=0 if selected is None else int(scene.model.records['template_id'][scene._validate_event(selected)])
    explanation=options.get('explanation','false') in ('true','1',True)
    settings=ExplanationSettings(**{name:_number(options[name],name,0. if name!='jet_diameter_um' else .01,1e6)
        for name in ExplanationSettings.__dataclass_fields__ if name in options}).validate()
    # Image generation is serialized because the original renderer owns its
    # Pillow canvas and small field caches. It never solves new physics.
    if not hasattr(service,'classic_render_lock'):service.classic_render_lock=RLock()
    with service.classic_render_lock:
        if not hasattr(service,'classic_renderers'):service.classic_renderers={}
        key=id(scene.data)
        renderer=service.classic_renderers.get(key)
        if renderer is None:
            renderer=SceneRenderer(scene.data);service.classic_renderers[key]=renderer
            if len(service.classic_renderers)>2:service.classic_renderers.pop(next(iter(service.classic_renderers)))
        picture=renderer.render(width,height,page,progress,index,phase,
            show_field=options.get('show_field','true')!='false',
            show_cells=options.get('show_cells','true')!='false',
            show_streams=options.get('show_streams','true')!='false',
            explanation=explanation,settings=settings)
        output=BytesIO();picture.save(output,format='PNG');return output.getvalue()


def demonstration(service,options):
    from ..visualization.explorer_data import ExplanationSettings, rayleigh_reference, illustrative_cell_response
    from ..scene3d.server import _number
    from ..scene3d.data import _clean
    scene=source_scene(service)
    selected=options.get('id',scene.model.default_event)
    if selected is None:raise ValueError(' No cell event in the current source ')
    event=scene.event(selected);track=scene.model.tracks[event['trajectory_index']]
    p=ExplanationSettings(**{name:_number(options[name],name,0. if name!='jet_diameter_um' else .01,1e6)
        for name in ExplanationSettings.__dataclass_fields__ if name in options}).validate()
    t=np.asarray(track['t']);c=np.asarray(track['C'])
    # Match the old visual lab: an explicit hypothetical hold at the final
    # archived solvent concentration follows the real transport exposure.
    if p.extra_exposure_ms>0:
        t=np.r_[t,t[-1]+np.linspace(0.,p.extra_exposure_ms*.001,41)[1:]]
        c=np.r_[c,np.full(40,c[-1])]
    integrity,remaining,released=illustrative_cell_response(t,c,p)
    return _clean(dict(kind='explicit_mechanism_assumption',source_id=scene.source_id,event_id=event['event_id'],
        parameters=vars(p),time_since_entry_s=t,methanol_fraction=c,illustrative_integrity=integrity,
        illustrative_remaining=remaining,illustrative_released=released,
        response=dict(illustrative_integrity=float(integrity[-1]),
            illustrative_remaining=float(remaining[-1]),illustrative_released=float(released[-1]),
            assumed_post_exposure_ms=p.extra_exposure_ms),
        rayleigh_reference=rayleigh_reference(p.jet_diameter_um,scene.data.config.fluid.qs+scene.data.config.fluid.qh),
        notes=[' Reference  Taylor  Cone  / Rayleigh  Droplets  /  Specified membrane response ,  Separate from the two-phase solution .']))
