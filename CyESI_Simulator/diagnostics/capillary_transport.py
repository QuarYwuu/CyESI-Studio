"""Separate transverse localization, axis centering and temporal ordering.

This module interrogates computed histories. It never introduces an attractive
force, infers a liquid thread from cell positions, or regularizes Poisson input.
"""
import math
import numpy as np
from ..particle.forces import brownian_diffusivity


def sections(history, distances_m, include_points=False):
    """First downstream crossing at each station, excluding unfinished paths.

    These are crossing ensembles of saved responses, not instantaneous in-tube
    concentrations. First crossing also handles a nonmonotone Brownian path.
    """
    names=('time_s','distance_m','x_m','y_m')
    arrays={key:np.asarray(history[key],float) for key in names}
    shape=arrays['time_s'].shape
    if len(shape)!=2 or any(value.shape!=shape for value in arrays.values()):
        raise ValueError('Bulk history must have matching two-dimensional arrays')
    records=[]
    for distance in distances_m:
        points=[];times=[]
        for i in range(shape[1]):
            valid=np.logical_and.reduce([np.isfinite(arrays[name][:,i]) for name in names])
            t,s,x,y=(arrays[name][valid,i] for name in names)
            if len(t)<1 or np.any(np.diff(t)<0):continue
            indices=np.flatnonzero(s>=distance-1e-12)
            if not len(indices):continue
            k=int(indices[0]);weight=0. if k==0 else np.clip((distance-s[k-1])/max(s[k]-s[k-1],1e-30),0,1)
            interpolate=lambda v:float(v[k] if k==0 else v[k-1]+weight*(v[k]-v[k-1]))
            points.append([interpolate(x),interpolate(y)]);times.append(interpolate(t))
        point=np.asarray(points,float).reshape(-1,2)
        if not len(point):
            records.append(dict(distance_m=float(distance),count=0,centroid_x_um=None,centroid_y_um=None,
                centroid_offset_um=None,axis_r95_um=None,spread_r95_um=None,x_width90_um=None,y_width90_um=None,
                mean_crossing_s=None,std_crossing_s=None))
            if include_points:records[-1]['points_um']=[]
            continue
        center=point.mean(axis=0);r=np.linalg.norm(point,axis=1);spread=np.linalg.norm(point-center,axis=1)
        records.append(dict(distance_m=float(distance),count=len(point),centroid_x_um=float(center[0]*1e6),
            centroid_y_um=float(center[1]*1e6),centroid_offset_um=float(np.linalg.norm(center)*1e6),
            axis_r95_um=float(np.quantile(r,.95)*1e6),spread_r95_um=float(np.quantile(spread,.95)*1e6),
            x_width90_um=float(np.diff(np.quantile(point[:,0],[.05,.95]))[0]*1e6),
            y_width90_um=float(np.diff(np.quantile(point[:,1],[.05,.95]))[0]*1e6),
            mean_crossing_s=float(np.mean(times)),std_crossing_s=float(np.std(times))))
        if include_points:records[-1]['points_um']=(point*1e6).tolist()
    return records


def assess_capillary(data):
    """Describe the actual long-pipe closure separately from the tip CFD."""
    c=data.config;g=c.geometry;f=c.fluid;cell=c.cell;fields=data.fields
    a=g.a;d=cell.diameter_um*1e-6;R=d/2;u=f.qs/(math.pi*a*a)
    D=float(brownian_diffusivity(f.temperature_k,f.sample_viscosity,R))
    rate=cell.concentration_per_ml*f.sample_flow_ul_min/60000
    prefixes=('ids','time_s','distance_m','x_m','y_m')
    has_history=all('bulk_trajectory_'+name in fields for name in prefixes)
    stations=[]
    if has_history and g.bulk_length>0:
        history={name:fields['bulk_trajectory_'+name] for name in prefixes}
        count=np.shape(history['time_s'])[1] if np.ndim(history['time_s'])==2 else 0
        if count>512:
            keep=np.unique(np.linspace(0,count-1,512,dtype=int))
            history={name:value[keep] if name=='ids' else value[:,keep] for name,value in history.items()}
        stations=sections(history,np.array([0,.01,.1,.25,.5,.75,1.])*g.bulk_length)
    endpoint=stations[-1] if stations else None
    time=None if endpoint is None else endpoint['mean_crossing_s']
    brownian=cell.use_brownian
    diffusion=None if time is None else math.sqrt(2*D*time)*1e6
    narrow=endpoint is not None and endpoint['count']>=3 and endpoint['spread_r95_um']<cell.diameter_um/2
    off_axis=narrow and endpoint['centroid_offset_um']>g.inner_id_um/4
    tiny_spread=(not brownian and narrow and diffusion is not None and diffusion>endpoint['spread_r95_um'])
    field_scope=[
        dict(field=' Pressure  /  Axial velocity ',long_tube=' Fully developed  Poiseuille +  Hydrostatic pressure  /  Hydraulic resistance network ',tip=' Axisymmetric  FEM Navier–Stokes /  Electroosmotic coupling ',gap=' Reduced capillary domain  3D  Fluid ;  Entrance development ,  Bends and particle disturbances excluded from the baseline '),
        dict(field=' Gravity  /  Buoyancy ',long_tube=' Segment-dependent  3D  Transverse settling and axial slip ',tip=' Local tip axial gravity ',gap=' Non-axisymmetric tip settling excluded '),
        dict(field=' Shear  /  Lift  /  Near-wall resistance ',long_tube=' Slip-dependent  Saffman,  Anisotropic lubrication and binding kinetics ',tip=' Axisymmetric viscous stress and finite-sphere drag ',gap=' Reduced shear-gradient / Wall-induced inertial lift , Faxén,  Rotation and particle-resolved flow '),
        dict(field=' Potential  /  Electrophoresis  / DEP / EOF',long_tube=' Baseline capillary transport uses the hydraulic background ',tip=' Potential  /  Ionic current  / EOF,  Particle electrophoresis and  DEP',gap=' Tip-domain electromigration ;  Upstream electrical boundary required for full-capillary coupling ,  Specify the  kV/ Capillary length as  E'),
        dict(field=' Methanol  /  Salt  /  Temperature ',long_tube=' Fixed sample composition and temperature ',tip=' Neutral methanol transport ,  Electroneutral electrolyte and current ',gap=' Full-capillary methanol back-diffusion omitted / Transient diffusion and Joule heating ;  Methanol has zero net charge '),
        dict(field=' Thermal fluctuations  /  Cell deformation  /  Interactions ',long_tube=f'Brownian {" On " if brownian else " Off "};  Independent rigid spheres ',tip=' Independent finite-sphere baseline ;  Local common-clock interaction study available / Exclusion / Binding / Unbinding ',gap=' Separate population and membrane models ;  Specified interaction parameters ;  Independent noise approximation near walls '),
    ]
    return dict(source=str(data.source),sample_length_mm=g.sample_length_mm,bulk_length_mm=g.bulk_length*1e3,
        orientation=c.supply.orientation,diameter_um=cell.diameter_um,inner_id_um=g.inner_id_um,
        flow_ul_min=f.sample_flow_ul_min,concentration_per_ml=cell.concentration_per_ml,
        mean_fluid_speed_mm_s=u*1e3,channel_reynolds=f.sample_density*u*2*a/f.sample_viscosity,
        particle_shear_reynolds=f.sample_density*u*2*a/f.sample_viscosity*(d/(2*a))**2,
        bulk_brownian_enabled=brownian,brownian_diffusivity_m2_s=D,free_1d_brownian_rms_um=diffusion,
        free_settling_speed_um_s=(cell.density-f.sample_density)*9.80665*d*d/(18*f.sample_viscosity)*1e6,
        brownian_width_warning=bool(tiny_spread),narrow_transverse_cluster_in_model=bool(narrow),
        near_wall_cluster_in_model=bool(off_axis),sections=stations,field_scope=field_scope,
        source_rate_s=rate,mean_poisson_gap_s=None if rate==0 else 1/rate,
        mean_spacing_at_fluid_mean_speed_mm=None if rate==0 else u/rate*1e3,
        stationary_independent_delay_gap_cv=1.,axial_ordering_interactions_solved=False,
        continuous_phase=' Single-phase sample ;  Narrow cell streams preserve aqueous / Methanol phase continuity ',
        spatial_note=' Saved station-crossing distribution ,  Station distribution ;  Inspected up to 512 Saved trajectories . Width about the centroid ,  Axis offset reported separately . Narrow-stream criterion in this model ,  Single-file ordering requires pair-dynamics analysis .',
        temporal_note=' Independent Poisson arrivals and transit delays retain stochastic timing ; CV=1  Source-model assumption ,  Model-derived arrival statistics .',
        brownian_note=' Free one-dimensional diffusion  RMS  Order-of-magnitude estimate ,  Free-space reference ; 8 μm  Thermal fluctuations broaden deterministic streams .')
