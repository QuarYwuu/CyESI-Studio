"""Render the 2D figure gallery from saved numerical arrays.

Matching image exports are reused. Other plots use the stored fields and paths.
Plots requiring unavailable face fluxes, inlet radii or voltage comparisons
remain unavailable.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import shutil

import numpy as np

from ..config import Config
from ..visualization import figures as original


TITLES = [
    ' Geometry and outlet recess ', ' Velocity and pressure ', ' Hydrodynamic focusing and shear stress ',
    ' Finite-sphere trajectories and outlet distribution ', ' Residence times and continuous events ', ' Methanol exposure history ',
    ' Potential and electric field ', ' Computed interface transient ', ' Parameter sweep comparison ', ' Gas velocity and pressure ',
]


def _rows(path):
    if not path.is_file(): return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _numeric(value):
    if value in ('', None): return None
    try: return float(value)
    except (TypeError, ValueError): return value


def _same_configuration(data):
    try: return Config.load(Path(data.source)/'config.json').to_dict() == data.config.to_dict()
    except (OSError, ValueError, KeyError, TypeError): return False


def saved_result(data):
    """A plot-only adapter, with no FEM, particle solver or random resampling."""
    f=data.fields; c=deepcopy(data.config); source=Path(data.source); notices=[]
    nr=len(f['liquid_r_edges_m'])-1; nz=len(f['liquid_z_edges_m'])-1
    solvent=SimpleNamespace(radial_edges=f['liquid_r_edges_m'],axial_edges=f['liquid_z_edges_m'],
        velocity_r=f['velocity_r_m_s'],velocity_z=f['velocity_z_m_s'],pressure=f['pressure_pa'],
        concentration=f['methanol_fraction'],active=np.asarray(f['liquid_active'],bool),
        r=.5*(f['liquid_r_edges_m'][:-1]+f['liquid_r_edges_m'][1:]),
        z=.5*(f['liquid_z_edges_m'][:-1]+f['liquid_z_edges_m'][1:]))
    electric=SimpleNamespace(radial_edges=f['electric_r_edges_m'],axial_edges=f['electric_z_edges_m'],
        potential=f['potential_v'],field_r=f['field_r_v_m'],field_z=f['field_z_v_m'])
    cells=_rows(source/'cells.csv')
    count=len(f.get('response_exited',[])) or len(cells) or c.cell.count
    exited=np.asarray(f.get('response_exited',[int(row['status'])==1 for row in cells]),bool)
    if len(exited)!=count: exited=np.zeros(count,bool); notices.append(' Complete response exit states unavailable .')
    def column(key,scale=1.,default=None):
        values=np.full(count,np.nan)
        for j,row in enumerate(cells):
            i=int(row.get('cell_id',j))
            if 0<=i<count and row.get(key) not in (None,''):
                values[i]=float(row[key])*scale
        if default is not None:
            supplied=np.asarray(default,float)
            if supplied.shape==(count,):values=np.where(np.isfinite(values),values,supplied)
        return values
    residence=column('residence_s',default=f.get('response_residence_time_s'))
    initial=column('initial_r_um',1e-6)
    initial=initial[np.isfinite(initial)]
    diagnostics=deepcopy(data.summary.get('particles',{}))
    diagnostics.setdefault('sample_volume_over_flow_s',
        float(np.pi*c.geometry.a**2*c.geometry.sample_length_mm*.001/max(c.fluid.qs,1e-30)))
    diagnostics.setdefault('outlet_r95_um',None)
    p=SimpleNamespace(saved_ids=np.asarray(data.ids),trajectory_time=f['trajectory_time_s'],
        trajectory_r=f['trajectory_r_m'],trajectory_z=f['trajectory_z_m'],
        trajectory_methanol=f['trajectory_methanol_fraction'],initial_radius=initial,
        residence_time=residence,exited=exited,outlet_radius=column('outlet_r_um',1e-6),
        outlet_methanol=column('outlet_methanol_fraction'),tip_residence_time=column('tip_residence_s'),
        diagnostics=diagnostics)
    # CSV files from compact numeric records may omit tip-time and exposure
    # columns. Fill only saved sample IDs from their exact trajectory histories.
    for j,identifier in enumerate(p.saved_ids):
        i=int(identifier)
        if not 0<=i<count:continue
        tr=data.track(j)
        if not np.isfinite(p.tip_residence_time[i]):p.tip_residence_time[i]=tr['end_time']-tr['bulk_time']
        if exited[i]:
            if not np.isfinite(p.outlet_radius[i]):p.outlet_radius[i]=tr['r'][-1]
            if not np.isfinite(p.outlet_methanol[i]):p.outlet_methanol[i]=tr['C'][-1]
    if len(initial)!=count:notices.append(' Complete inlet positions unavailable ;  Inlet distribution unavailable .')
    if np.any(exited & ~np.isfinite(p.tip_residence_time)):
        notices.append(' Some tip residence times unavailable ;  Exposure scatter uses saved trajectories .')
    event_rows=_rows(source/'events.csv') if _same_configuration(data) else []
    rate=c.cell.concentration_per_ml*c.fluid.sample_flow_ul_min/60000.
    events=dict(rate_cells_s=rate,nozzle_s=np.array([float(row.get('nozzle_s') or 'nan') for row in event_rows]))
    if not event_rows:notices.append(' Continuous event table unavailable ;  Event histogram unavailable ,  Original samples retained .')
    scan=[{k:_numeric(v) for k,v in row.items()} for row in _rows(source/'scan.csv')]
    sp=data.spray
    spray=None if sp is None else SimpleNamespace(radial_edges=sp['radial_edges_m'],axial_edges=sp['axial_edges_m'],
        times=sp['time_s'],phase=sp['liquid_fraction'],active=sp['active'],
        velocity_r=sp['velocity_r_m_s'],velocity_z=sp['velocity_z_m_s'],pressure=sp['pressure_pa'])
    zero=SimpleNamespace(exited=np.zeros(count,bool),residence_time=np.full(count,np.nan))
    result=SimpleNamespace(config=c,solvent=solvent,electric=electric,particles=p,
        particles_zero_voltage=zero,events=events,spray=spray,scan=scan,notices=notices,
        inlet_distribution_available=len(initial)==count,event_table_available=bool(event_rows),
        source=data.source,fields=f)
    return result


def _focusing(result):
    # Streamfunction contours require saved conservative face fluxes.
    s=result.solvent; f=result.fields
    fig,axes=original._new('Figure 3 | Hydrodynamic focusing and shear',cols=2,size=(12,4.7))
    original._map(axes[0],s.radial_edges,s.axial_edges,s.concentration,'Methanol fraction','YlGnBu',s.active,(-1.3,0))
    axes[0].set_title('Saved solvent field; conservative face flux was not archived')
    axes[0].text(.02,.03,'Streamfunction contours unavailable in this archive',transform=axes[0].transAxes,
        fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    stress=f.get('equivalent_stress_pa')
    if stress is None:
        from ..fluid.stress import axisymmetric_stress
        from ..chemistry.solvent_transport import filled
        fluid=result.config.fluid
        mu=np.exp((1-s.concentration)*np.log(fluid.sample_viscosity)+s.concentration*np.log(fluid.sheath_viscosity))
        stress=axisymmetric_stress(s.r,s.z,filled(s.velocity_r,s.active),filled(s.velocity_z,s.active),mu)['equivalent_stress_pa']
    original._map(axes[1],s.radial_edges,s.axial_edges,stress,'Equivalent viscous stress (Pa)','magma',s.active,(-1.3,0))
    return fig


def _trajectories(result):
    p,g=result.particles,result.config.geometry
    fig,axes=original._new('Figure 4 | Finite-sphere cell transport',cols=2,size=(12,4.7))
    for j in range(min(40,len(p.saved_ids))):
        axes[0].plot(p.trajectory_z[:,j]*1e3,p.trajectory_r[:,j]*1e6,lw=.65,alpha=.6,color=original.BLUE)
    axes[0].set(xlim=(-1.4,0),ylim=(0,g.c*1e6),xlabel='z (mm)',ylabel='Cell center r (um)')
    axes[0].axvline(g.inner_tip*1e3,ls='--',color='#888')
    if result.inlet_distribution_available:
        axes[1].hist(p.initial_radius*1e6,bins=24,density=True,alpha=.45,color=original.ORANGE,label='inlet, flux-weighted')
    else:
        axes[1].text(.02,.97,'Full inlet distribution was not archived',transform=axes[1].transAxes,va='top',fontsize=8)
    exits=p.outlet_radius[p.exited & np.isfinite(p.outlet_radius)]
    if len(exits):axes[1].hist(exits*1e6,bins=24,density=True,alpha=.6,color=original.GREEN,label='saved nozzle exits')
    axes[1].set(xlabel='r (um)',ylabel='Probability density (1/um)')
    if axes[1].get_legend_handles_labels()[0]:axes[1].legend()
    return fig


def render_existing(data,directory,progress=lambda value:None,animations=True):
    """Create a source-backed legacy gallery, retaining the original archive."""
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    source=Path(data.source);exact=_same_configuration(data);result=None
    manifest=dict(format_version=1,source=str(data.source),source_config=data.config.to_dict(),
        physics_recomputed=False,figures=[],animations=[],warnings=[])
    for index,(filename,builder,title) in enumerate(zip(original.FILENAMES,original.BUILDERS,TITLES),1):
        progress(' Rendering 2D views  '+str(index)+' / '+str(len(TITLES))+' · '+title)
        files=[]
        if exact:
            for extension in ('png','svg'):
                old=source/(filename+'.'+extension)
                if old.is_file():
                    target=directory/old.name
                    if old.resolve()!=target:shutil.copy2(old,target)
                    files.append(target.name)
        status='copied' if any(name.endswith('.png') for name in files) else 'rendered'
        if status=='rendered':
            if result is None:result=saved_result(data)
            with original._PLOT_LOCK:
                fig=None
                try:
                    fig=(_focusing(result) if index==3 else _trajectories(result) if index==4 else builder(result))
                    if index==5:
                        fig.axes[0].text(.98,.97,'0 kV comparison was not archived',ha='right',va='top',
                            transform=fig.axes[0].transAxes,fontsize=8)
                        if not result.event_table_available:
                            fig.axes[1].text(.5,.5,'Current event table was not archived; no new Poisson draw',
                                ha='center',transform=fig.axes[1].transAxes,fontsize=8)
                    for extension in ('png','svg'):
                        target=directory/(filename+'.'+extension);fig.savefig(target);files.append(target.name)
                finally:
                    if fig is not None:original.plt.close(fig)
        manifest['figures'].append(dict(id=filename,title=title,status=status,files=sorted(set(files))))
    if animations:
        for name in ('cell_trajectories.gif','two_phase.gif'):
            old=source/name
            if exact and old.is_file():
                target=directory/name
                if old.resolve()!=target:shutil.copy2(old,target)
                manifest['animations'].append(dict(id=name[:-4],file=name,status='copied'))
        existing={row['id'] for row in manifest['animations']}
        if 'cell_trajectories' not in existing:
            if result is None:result=saved_result(data)
            progress(' Rendering finite-sphere trajectories  ·  Age-aligned responses ,  Individual arrivals ')
            target=original.save_trajectory_animation(result,directory)
            if target is not None:manifest['animations'].append(dict(id='cell_trajectories',file=target.name,status='rendered'))
        # The separate phase GIF shares its own microsecond snapshot clock.
        if data.spray is not None and 'two_phase' not in existing:
            if result is None:result=saved_result(data)
            from matplotlib.animation import FuncAnimation,PillowWriter
            with original._PLOT_LOCK:
                sp=result.spray;fig,ax=original.plt.subplots(figsize=(7,4),layout='constrained')
                try:
                    image=ax.pcolormesh(sp.axial_edges*1e3,sp.radial_edges*1e6,sp.phase[0],vmin=0,vmax=1,cmap='Blues',shading='flat')
                    fig.colorbar(image,ax=ax,label='Liquid fraction');ax.set(xlabel='z (mm)',ylabel='r (um)')
                    def draw(k):
                        image.set_array(sp.phase[k]);ax.set_title(f'Computed two-phase transient | {sp.times[k]*1e3:.4f} ms')
                        return [image]
                    frames=np.unique(np.linspace(0,len(sp.times)-1,min(50,len(sp.times)),dtype=int))
                    FuncAnimation(fig,draw,frames=frames).save(directory/'two_phase.gif',writer=PillowWriter(fps=15))
                finally:original.plt.close(fig)
            manifest['animations'].append(dict(id='two_phase',file='two_phase.gif',status='rendered'))
    if result is not None:manifest['warnings'].extend(result.notices)
    manifest['warnings'].extend([
        ' Trajectory replay aligned by time since entering the resolved tip ;  Continuous injection and cells  ID  Open the process view .',
        ' Interface animation uses its own clock ,  Transient  Taylor  Cone evolution .'])
    from ..scene3d.data import _clean
    manifest=_clean(manifest)
    (directory/'classic_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return manifest
