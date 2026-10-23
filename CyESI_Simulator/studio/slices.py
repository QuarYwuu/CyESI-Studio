"""Flat views of axisymmetric and Cartesian fields.

Offset cuts of axisymmetric fields are labelled reconstructions. Material
masks separate liquid, quartz and gas.
"""
from __future__ import annotations

import numpy as np

from types import SimpleNamespace
from pathlib import Path

from ..scene3d.fields import FIELD_INFO as CORE_FIELD_INFO, _scale
from ..scene3d.slices import scene_slice, sphere_intersections, flat_slice

FIELD_INFO = CORE_FIELD_INFO | {
    'shear': (' Viscous stress ', 'Pa', 'liquid'),
    'charge_density': (' Free charge density ', 'C/m³', 'fluid'),
    'liquid_fraction': (' Liquid volume fraction ', 'fraction', 'fluid'),
}


def _extra_scale(values, active, name):
    if name in CORE_FIELD_INFO:
        return _scale(values, active, name)
    finite = np.asarray(values)[active & np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if len(finite) else (0., 1.)
    if name == 'liquid_fraction':
        low, high = 0., 1.
    elif name == 'charge_density':
        high = max(abs(low), abs(high), 1e-12); low = -high
    else:
        low = 0.
    if high <= low: high = low+max(1., abs(low)*1e-6)
    return dict(field=name, label=FIELD_INFO[name][0], unit=FIELD_INFO[name][1],
        min=low, max=high, palette='diverging' if name=='charge_density' else 'sequential',
        data_min=None if not len(finite) else float(finite.min()),
        data_max=None if not len(finite) else float(finite.max()))


def _cells(array, re, ze, r, z, active=None):
    """Sample cell-constant values on the stored grid."""
    re, ze, array = np.asarray(re), np.asarray(ze), np.asarray(array)
    if array.shape != (len(re)-1, len(ze)-1):
        raise ValueError('Archived field dimensions do not match its physical grid')
    ri=np.clip(np.searchsorted(re,r,side='right')-1,0,len(re)-2)
    zi=np.clip(np.searchsorted(ze,z,side='right')-1,0,len(ze)-2)
    valid=(r>=re[0])&(r<=re[-1])&(z>=ze[0])&(z<=ze[-1])
    if active is not None: valid &= np.asarray(active, bool)[ri,zi]
    values=array[ri,zi]
    return np.where(valid,values,np.nan), valid


def _ions(data):
    # Loaded numerical records are immutable. Loading this small sidecar once
    # keeps view changes and timeline playback away from repeated disk I/O.
    if not hasattr(data, '_studio_ionic_archive'):
        path=Path(data.source)/'ions.npz'
        if path.is_file():
            with np.load(path,allow_pickle=False) as a:
                data._studio_ionic_archive={key:a[key].copy() for key in a.files}
        else: data._studio_ionic_archive=None
    return data._studio_ionic_archive


def _history(phase, names, selected):
    name=next((key for key in names if key in phase),None)
    if name is None: raise ValueError("This field is absent from the interface record. Run an interface transient.")
    raw=np.asarray(phase[name])
    if raw.ndim==3: return raw[selected]
    if raw.ndim==2 and selected==len(phase['time_s'])-1: return raw
    raise ValueError("Only the final field was saved; earlier snapshots are unavailable.")


def _native_extra(scene, field, axis, position_m, clock, phase_frame, time_s):
    if clock not in ('volume','phase'): raise ValueError('Field clock must be volume or phase')
    volume=scene.volume; selected=None; field_time=None; notes=[]
    if clock=='phase':
        phase=getattr(volume,'phase',None)
        if phase is None or not len(phase.time_s): raise ValueError('No native moving-phase field has been computed')
        selected=int(np.clip(phase_frame if phase_frame>=0 else len(phase.time_s)-1,0,len(phase.time_s)-1))
        field_time=float(phase.time_s[selected])
        material=np.where(phase.material==2,2,np.where(phase.liquid_fraction[selected]>=.5,1,0))
        grid=phase.grid; velocity=phase.velocity[selected]; electric=phase.electric[selected]
        composition=phase.methanol[selected]
    else:
        grid=volume.grid; material=volume.material; velocity=volume.velocity; electric=volume.electric
        composition=volume.methanol
    if field=='charge_density':
        if clock!='phase': raise ValueError("Free-charge density uses the interface solution. Select the interface clock.")
        values=phase.charge_density_c_m3[selected]
    elif field=='liquid_fraction':
        if clock!='phase': raise ValueError("Liquid fraction uses the two-phase solution. Select the interface clock.")
        values=phase.liquid_fraction[selected]
    else:
        from scipy.ndimage import distance_transform_edt
        key=('native_shear',clock,selected)
        cache=getattr(scene,'_studio_derived_fields',None)
        if cache is None:scene._studio_derived_fields=cache={}
        if key not in cache:
            liquid=material==1
            if not liquid.any(): raise ValueError("Liquid stress is unavailable at this time.")
            nearest=distance_transform_edt(~liquid,return_distances=False,return_indices=True)
            filled=np.asarray(velocity)[tuple(nearest)]
            gradients=np.stack([np.stack(np.gradient(filled[...,j],*grid.centers,edge_order=1),axis=-1)
                                for j in range(3)],axis=-2)
            strain=.5*(gradients+np.swapaxes(gradients,-1,-2))
            c=np.clip(composition,0,1); f=scene.data.config.fluid
            mu=np.exp((1-c)*np.log(f.sample_viscosity)+c*np.log(f.sheath_viscosity))
            cache[key]=np.where(liquid,mu*np.sqrt(2*np.sum(strain*strain,axis=(-1,-2))),np.nan)
            if len(cache)>8:cache.pop(next(iter(cache)))
        values=cache[key]
        notes.append(' Stress from velocity gradients : τ_eq = μ√(2D:D),  Consistent with the 2D stress definition ;  Nearest-liquid gradient extension at material boundaries .')
    # Reuse the native exact-plane/slab/mask contract without altering its
    # shared global field registry or any existing application's behavior.
    wrapper=SimpleNamespace(grid=grid, material=material, pressure=np.asarray(values),
                            velocity=electric if field=='charge_density' else velocity)
    result=flat_slice(wrapper,'pressure',axis,position_m)
    mapped=np.asarray([[np.nan if v is None else v for v in row] for row in result['values']],float)
    result['scale']=_extra_scale(mapped,np.isfinite(mapped),field)
    result.update(source_id=scene.source_id,data_clock=clock,field_time_s=field_time,
        phase_frame=selected,transport_time_s=time_s,
        vector_unit='V/m' if field=='charge_density' else 'm/s')
    result['notes'].extend(notes)
    if clock=='phase':result['notes'].append(' Separate interface and transport clocks .')
    if time_s is not None:result['cell_intersections']=sphere_intersections(scene.frame(float(time_s))['cells'],axis,position_m)
    from ..scene3d.data import _clean
    return _clean(result)


def field_slice(scene, field='methanol', axis='x', position_m=0., clock='volume',
                phase_frame=-1, time_s=None):
    if field not in FIELD_INFO: raise ValueError("Unknown field or clock")
    if hasattr(scene, 'volume'):
        if field not in CORE_FIELD_INFO:
            return _native_extra(scene,field,axis,position_m,clock,phase_frame,time_s)
        return scene_slice(scene, field, axis, position_m, clock, phase_frame, time_s)
    if axis not in ('x', 'y', 'z') or not np.isfinite(position_m):
        raise ValueError("Invalid section axis or position")
    if clock not in ('volume', 'phase'):
        raise ValueError("Unknown field or clock")
    if field == 'electric_velocity':
        raise ValueError("Voltage differences require two axisymmetric solutions")
    data=scene.data; f=data.fields; selected=None; field_time=None
    electric=field in ('electric', 'potential')
    phase=data.spray if clock == 'phase' else None
    if clock == 'phase' and phase is None:
        raise ValueError("No interface transient is saved. Enable the two-phase solver and run a calculation.")
    if phase is not None:
        re=np.asarray(phase['radial_edges_m']); ze=np.asarray(phase['axial_edges_m'])
        selected=int(np.clip(phase_frame if phase_frame>=0 else len(phase['time_s'])-1, 0, len(phase['time_s'])-1))
        field_time=float(phase['time_s'][selected])
    else:
        re=np.asarray(f['electric_r_edges_m' if electric else 'liquid_r_edges_m'])
        ze=np.asarray(f['electric_z_edges_m' if electric else 'liquid_z_edges_m'])
    radius=float(re[-1]); position_m=float(position_m)
    normal=ze if axis=='z' else np.r_[-re[:0:-1],re]
    if not normal[0] <= position_m <= normal[-1]:
        raise ValueError('Slice lies outside the axisymmetric volume')
    if axis=='z':
        edges=np.r_[-re[:0:-1],re]; he=edges; ve=edges
        h,v=np.meshgrid((he[:-1]+he[1:])/2,(ve[:-1]+ve[1:])/2)
        r=np.hypot(h,v); z=np.full_like(r,position_m)
        ha,va='x','y'
    else:
        he=ze; ve=np.r_[-re[:0:-1],re]
        z,v=np.meshgrid((he[:-1]+he[1:])/2,(ve[:-1]+ve[1:])/2)
        r=np.hypot(v,position_m); ha,va='z',('y' if axis=='x' else 'x')
    ri=np.clip(np.searchsorted(re,r,side='right')-1,0,len(re)-2)
    zi=np.clip(np.searchsorted(ze,z,side='right')-1,0,len(ze)-2)
    inside=r<=radius
    g=data.config.geometry
    # The physical quartz annuli, including finite end faces, stay solids.
    quartz=(((r>=g.a)&(r<=g.b)&(z<=g.inner_tip)) |
            ((r>=g.c)&(r<=g.d)&(z<=0.)) |
            ((r>=g.gas_id_um*.5e-6)&(r<=g.gas_od_um*.5e-6)&(z<=-g.gas_recess_mm*.001)))
    ur=np.zeros_like(r); uz=np.zeros_like(r)
    if phase is not None:
        phi=np.asarray(phase['liquid_fraction'][selected])[ri,zi]
        phase_solid=~np.asarray(phase.get('active',np.ones((len(re)-1,len(ze)-1),bool)),bool)[ri,zi]
        material=np.where(quartz|phase_solid,2,np.where(phi>=.5,1,0))
        # Vector overlays follow the same snapshot as the scalar, including
        # when the displayed scalar is solvent, salt or liquid fraction.
        electric_vector=field in ('electric','potential','charge_density')
        components=(('field_r_v_m_history',),('field_z_v_m_history',)) if electric_vector else (
            ('velocity_r_history_m_s','velocity_r_m_s'),('velocity_z_history_m_s','velocity_z_m_s'))
        sampled=[]
        for names in components:
            sampled.append(_history(phase,names,selected)[ri,zi] if any(k in phase for k in names) else np.zeros_like(r))
        ur,uz=sampled
        key={'methanol':'methanol_fraction_history','salt':'salt_mol_m3_history',
             'electric':'field_r_v_m_history','potential':'potential_v_history',
             'pressure':'pressure_history_pa','speed':'velocity_z_history_m_s','gas_speed':'velocity_z_history_m_s',
             'charge_density':'charge_density_c_m3_history','liquid_fraction':'liquid_fraction',
             'shear':'velocity_z_history_m_s'}[field]
        alternative={'pressure_history_pa':'pressure_pa','velocity_z_history_m_s':'velocity_z_m_s'}.get(key)
        values=_history(phase,[key]+([alternative] if alternative else []),selected)[ri,zi]
        if field in ('speed','gas_speed'):
            rk='velocity_r_history_m_s' if 'velocity_r_history_m_s' in phase else 'velocity_r_m_s'
            ur=_history(phase,[rk],selected)[ri,zi]
            uz=_history(phase,[key]+([alternative] if alternative else []),selected)[ri,zi]
            values=np.hypot(ur,uz)*(1000 if field=='speed' else 1)
        if field=='electric':values=np.hypot(ur,uz)
        if field=='shear':
            from ..fluid.stress import axisymmetric_stress
            from scipy.ndimage import distance_transform_edt
            active=np.asarray(phase.get('active',np.ones((len(re)-1,len(ze)-1),bool)))
            nearest=distance_transform_edt(~active,return_distances=False,return_indices=True)
            uz0=_history(phase,['velocity_z_history_m_s','velocity_z_m_s'],selected)[tuple(nearest)]
            ur0=_history(phase,['velocity_r_history_m_s','velocity_r_m_s'],selected)[tuple(nearest)]
            c=_history(phase,['methanol_fraction_history'],selected)
            fluid=data.config.fluid
            mu=np.exp((1-np.clip(c,0,1))*np.log(fluid.sample_viscosity)+np.clip(c,0,1)*np.log(fluid.sheath_viscosity))
            stress=axisymmetric_stress(.5*(re[:-1]+re[1:]),.5*(ze[:-1]+ze[1:]),ur0,uz0,mu)
            values=stress['equivalent_stress_pa'][ri,zi]
    elif electric:
        material=np.where(quartz,2,np.where((r<g.c)&(z<=0),1,0))
        ur=np.asarray(f['field_r_v_m'])[ri,zi]; uz=np.asarray(f['field_z_v_m'])[ri,zi]
        potential=np.asarray(f['potential_v'])[ri,zi]
        liquid=material==1
        if data.config.electrical.liquid_model=='conducting':
            ur=np.where(liquid,0.,ur); uz=np.where(liquid,0.,uz)
        local_keys={'field_r_v_m':'liquid_field_r_v_m','field_z_v_m':'liquid_field_z_v_m','potential_v':'liquid_potential_v'}
        ion=None
        channels={}
        for key,local in local_keys.items():
            if local in f:
                channels[key]=_cells(f[local],f['liquid_r_edges_m'],f['liquid_z_edges_m'],r,z,f['liquid_active'])[0]
            else:
                if ion is None: ion=_ions(data)
                if ion is not None and key in ion:
                    channels[key]=_cells(ion[key],ion['radial_edges_m'],ion['axial_edges_m'],r,z,ion.get('active'))[0]
        if data.config.electrical.liquid_model!='conducting':
            if 'field_r_v_m' in channels:ur=np.where(liquid,channels['field_r_v_m'],ur)
            if 'field_z_v_m' in channels:uz=np.where(liquid,channels['field_z_v_m'],uz)
        if 'potential_v' in channels:potential=np.where(liquid,channels['potential_v'],potential)
        values=(np.hypot(ur,uz) if field=='electric' else potential)
    else:
        active=np.asarray(f['liquid_active'])[ri,zi].astype(bool)&inside
        material=np.where(quartz,2,np.where(active,1,0))
        ur=np.asarray(f['velocity_r_m_s'])[ri,zi]; uz=np.asarray(f['velocity_z_m_s'])[ri,zi]
        if field=='speed':values=np.hypot(ur,uz)*1000
        elif field=='methanol':values=np.asarray(f['methanol_fraction'])[ri,zi]
        elif field=='pressure':values=np.asarray(f['pressure_pa'])[ri,zi]
        elif field=='shear':
            if 'equivalent_stress_pa' not in f:raise ValueError("Stress is unavailable. Run axisymmetric transport.")
            values=np.asarray(f['equivalent_stress_pa'])[ri,zi]
        elif field in ('charge_density','liquid_fraction'):
            raise ValueError("This variable belongs to the interface transient. Select the interface clock.")
        elif field=='salt':
            ionic=_ions(data)
            if ionic is None:raise ValueError("Ion concentration is unavailable.")
            name=next((k for k in ('salt_mol_m3','salt_concentration_mol_m3','salt') if k in ionic),None)
            if name is None:raise ValueError("Salt concentration is unavailable.")
            values=_cells(ionic[name],ionic['radial_edges_m'],ionic['axial_edges_m'],r,z,ionic.get('active'))[0]
        elif field=='gas_speed':raise ValueError("The axisymmetric gas field uses the tip transient. Select the interface clock.")
    category=FIELD_INFO[field][2]
    enabled=inside & np.isfinite(values)
    if category=='liquid':enabled &= material==1
    elif category=='gas':enabled &= material==0
    elif category=='fluid':enabled &= material!=2
    scale=_extra_scale(np.asarray(values),enabled,field)
    scale['storage']='Axisymmetric finite-volume cells; Cartesian cut reconstruction'
    vectors=[]
    stride=[max(1,int(np.ceil(n/16))) for n in r.shape]
    hs=(he[:-1]+he[1:])/2; vs=(ve[:-1]+ve[1:])/2
    transverse=ur*np.divide(v,r,out=np.zeros_like(r),where=r>0)
    for i in range(0,len(vs),stride[0]):
        for j in range(0,len(hs),stride[1]):
            if enabled[i,j] and material[i,j]!=2:
                a,b=(ur[i,j]*h[i,j]/max(r[i,j],1e-30),transverse[i,j]) if axis=='z' else (uz[i,j],transverse[i,j])
                if np.isfinite([a,b]).all():vectors.append([float(hs[j]),float(vs[i]),float(a),float(b)])
    idx=int(np.clip(np.searchsorted(normal,position_m,side='right')-1,0,len(normal)-2))
    result=dict(mode='axisymmetric_flat_reconstruction',axis=axis,position_m=position_m,
        plane_index=idx,cell_slab_m=normal[idx:idx+2].tolist(),
        axis_ranges_m=dict(x=[-radius,radius],y=[-radius,radius],z=[float(ze[0]),float(ze[-1])]),
        horizontal_axis=ha,vertical_axis=va,horizontal_edges_m=he.tolist(),vertical_edges_m=ve.tolist(),
        shape=list(r.shape),values=np.where(enabled,values,np.nan).tolist(),material=material.tolist(),
        active_cells=int(enabled.sum()),scale=scale,vectors=vectors,vector_unit='V/m' if electric or field=='charge_density' else 'm/s',
        material_codes=dict(gas=0,liquid=1,quartz=2),source_id=scene.source_id,data_clock=clock,
        field_time_s=field_time,phase_frame=selected,transport_time_s=time_s,
        coordinate_system='Axisymmetric (r,z) solution reconstructed at physical Cartesian coordinates, metres',
        sampling='Piecewise constant original field cells; no interpolation through quartz',
        notes=[' Section reconstructed from axisymmetric fields ;  Axisymmetric reconstruction .',
               ' Separate continuous-feed and interface clocks .'])
    if time_s is not None:result['cell_intersections']=sphere_intersections(scene.frame(float(time_s))['cells'],axis,position_m)
    from ..scene3d.data import _clean
    return _clean(result)
