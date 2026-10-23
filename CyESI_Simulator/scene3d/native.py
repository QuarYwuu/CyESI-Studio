"""Bridge independent Cartesian fields and finite-sphere responses to the viewer."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .data import SceneData, _clean
from .particles import simulate_native
from ..visualization.explorer_data import ExplorerData


class NativeExplorerData(ExplorerData):
    """Saved-response interface with actual Cartesian field sampling available.

    Two-dimensional arrays are an identified meridian slice for compatibility;
    the 3-D renderer consumes ``volume`` and never revolves those arrays.
    """
    def sample(self,r,z,name='methanol_fraction'):
        if not hasattr(self,'volume'):return super().sample(r,z,name)
        r,z=np.broadcast_arrays(r,z);theta=getattr(self,'flight_azimuth',0.)
        points=np.stack([r*np.cos(theta),r*np.sin(theta),z],axis=-1)
        lower=np.array([a[0] for a in self.volume.grid.centers]);upper=np.array([a[-1] for a in self.volume.grid.centers])
        points=np.clip(points,lower,upper)
        key={'methanol_fraction':'methanol','pressure_pa':'pressure','potential_v':'potential',
             'velocity_r_m_s':'velocity','velocity_z_m_s':'velocity','field_r_v_m':'electric','field_z_v_m':'electric'}.get(name)
        if name=='liquid_active':
            g=self.config.geometry;rr=np.abs(r)
            return ((z<=0)&((rr<g.a)|((rr>=g.b)&(rr<g.c))|(z>=g.inner_tip)&(rr<g.c))).astype(float)
        if key is None:return super().sample(r,z,name)
        value=self.volume.sample(points,key)
        if name in ('velocity_r_m_s','field_r_v_m'):
            value=value[...,0]*np.cos(theta)+value[...,1]*np.sin(theta)
        elif name in ('velocity_z_m_s','field_z_v_m'):value=value[...,2]
        if name.startswith('field_'):
            # No native long-pipe electric boundary-value problem was solved.
            value=np.where(z<self.config.geometry.liquid_start,0.,value)
        return value

    def spray_sample(self,r,z,name,frame):
        if name=='liquid_fraction':
            return self.sample(r,z,'liquid_active')
        key=name.replace('_history','')
        return self.sample(r,z,key)


class NativeSceneData(SceneData):
    def __init__(self,data,volume,particles,base,path):
        self.volume=volume;self.native_particles=particles
        self.reference_source=getattr(base,'reference_source',base.data.source)
        super().__init__(data,base.model.acquisition,base.model.parameters,
                         base.state_parameters,path=path)
        self._native_geometry={}
        from .cell_charge import ChargeParameters
        self.charge_parameters=deepcopy(getattr(base,'charge_parameters',ChargeParameters()))
        self._charge_cache={}
        import hashlib,json
        digest=hashlib.sha256(self.source_id.encode())
        digest.update(json.dumps(vars(volume.settings),sort_keys=True,default=str).encode())
        if path is not None:digest.update(json.dumps(path.to_dict(),sort_keys=True).encode())
        for values in (*volume.grid.edges,volume.velocity,volume.potential,volume.methanol,
                       data.fields['trajectory_time_s'],data.fields['native_trajectory_x_m'],data.fields['native_trajectory_y_m']):
            array=np.ascontiguousarray(values);digest.update(str(array.shape).encode());digest.update(array.tobytes())
        phase=getattr(volume,'phase',None)
        if phase is not None:
            digest.update(np.ascontiguousarray(phase.time_s).tobytes())
            digest.update(np.ascontiguousarray(phase.liquid_fraction).tobytes())
        self.source_id=digest.hexdigest()[:16]

    def set_charge_parameters(self,values):
        from .cell_charge import ChargeParameters
        if not isinstance(values,dict):raise ValueError('Charge parameters must be a JSON object')
        values=dict(values)
        if 'intracellular_conductivity_s_m' in values:
            values['cytosol_conductivity_s_m']=values.pop('intracellular_conductivity_s_m')
        if set(values)-set(ChargeParameters.__dataclass_fields__):raise ValueError('Unknown cell charge parameter')
        current=self.charge_parameters.to_dict();current.update(values)
        self.charge_parameters=ChargeParameters(**current).validate();self._charge_cache.clear()

    def event(self,event_id):
        from .cell_charge import charge_at_track
        value=super().event(event_id)
        value['charge_history']=_clean(charge_at_track(self,value['trajectory_index']))
        from .transport_view import add_interaction_event
        return _clean(add_interaction_event(self,value,value['event_id']))

    def frame(self,time_s,selected_id=None,max_cells=1200):
        from .cell_charge import charge_at_event
        value=super().frame(time_s,selected_id,max_cells)
        value['selected_charge']=None if selected_id is None else _clean(charge_at_event(self,selected_id,time_s))
        from .transport_view import add_bulk_deposits, apply_interaction_frame
        value=add_bulk_deposits(self,value,time_s,value['selected_id'],max_cells)
        return _clean(apply_interaction_frame(self,value,time_s,value['selected_id'],max_cells))

    def with_method(self,acquisition,parameters):
        """Rebuild timing only; preserve the native fields and Cartesian paths."""
        from types import SimpleNamespace
        proxy=SimpleNamespace(data=self.data,state_parameters=self.state_parameters,
            model=SimpleNamespace(acquisition=acquisition,parameters=parameters))
        updated=NativeSceneData(self.data,self.volume,self.native_particles,proxy,self.path)
        updated.reference_source=self.reference_source
        updated._xyz=self._xyz;updated._geometry=self._geometry
        updated._native_geometry=self._native_geometry;updated._flights=self._flights
        updated.charge_parameters=deepcopy(self.charge_parameters);updated._charge_cache=self._charge_cache
        if hasattr(self,'voltage_comparison'):updated.voltage_comparison=self.voltage_comparison
        if hasattr(self,'interaction_result'):
            # Timing changes cannot alter an already simulated continuous
            # source train. Preserve the study only for identical event IDs.
            ids=self.interaction_result.event_ids
            if (not len(ids) or int(np.max(ids))<len(updated.model.records['event_id'])) and all(
                   np.array_equal(self.model.records[key][ids],updated.model.records[key][ids],equal_nan=True)
                   for key in ('event_id','injection_s','template_id','nozzle_s')):
                updated.interaction_result=self.interaction_result
        return updated

    def _track_xyz(self,index):
        index=int(index)
        if index in self._xyz:return self._xyz[index]
        # The parent supplies exact saved bulk x/y, then its azimuthal extension
        # is replaced with the independently integrated Cartesian tip response.
        parent=super()._track_xyz(index)
        t,xyz,_,bulk=parent;xyz=xyz.copy();f=self.data.fields
        times=f['trajectory_time_s'][:,index]
        valid=np.isfinite(times)&np.isfinite(f['native_trajectory_x_m'][:,index])&np.isfinite(f['native_trajectory_y_m'][:,index])
        times=times[valid];keep=np.r_[np.diff(times)>1e-12,True]
        times=times[keep]
        x=f['native_trajectory_x_m'][valid,index][keep];y=f['native_trajectory_y_m'][valid,index][keep]
        col=self._bulk_columns.get(int(self.data.ids[index]))
        bulk_end=float(f['bulk_trajectory_time_s'][-1,col]) if col is not None else -np.inf
        tip=t>bulk_end+1e-12
        xyz[tip,0]=np.interp(t[tip],times,x);xyz[tip,1]=np.interp(t[tip],times,y)
        result=(t,xyz,float(np.arctan2(xyz[-1,1],xyz[-1,0])),bulk)
        self._xyz[index]=result
        return result

    def metadata(self):
        value=super().metadata();value['mode']='native_cartesian_tip'
        value['native_volume']=_clean(self.volume.diagnostics)
        value['native_volume_settings']=_clean(vars(self.volume.settings))
        value['native_transport']=_clean(self.native_particles.diagnostics)
        value['charge_parameters']=self.charge_parameters.to_dict()
        value['reference_source']=self.reference_source
        value['bulk_wall_adhesion']=dict(on_rate_s=self.data.config.supply.wall_capture_rate_s,
            capture_gap_um=self.data.config.supply.wall_capture_gap_um,
            scope='Uncalibrated conditional irreversible long-tube capture, separate from tip wall contact')
        if hasattr(self,'interaction_result'):
            from dataclasses import asdict
            value['interaction_study']=_clean(dict(parameters=asdict(self.interaction_result.parameters),
                diagnostics=self.interaction_result.diagnostics))
        if hasattr(self,'voltage_comparison'):
            value['voltage_comparison']=_clean(self.voltage_comparison.diagnostics)
        value['warnings']=['Quasi-steady background: '+w for w in self.volume.diagnostics.get('limits',[])]+[
            'Resolved tip velocity, gas, electric, methanol and salt fields are independently solved Cartesian 3-D fields.',
            'Upstream tubing uses segmented developed flow with route-projected gravity and 3-D cross-section particles. It is not full curved-tube 3-D CFD.',
            'The native solver holds the liquid/gas interface fixed. A dynamic 3-D Taylor cone and detached-droplet charge/shape evolution remain unresolved.',
            'Every cell preserves one event ID through upstream, resolved tip and acquisition timing; an assumed ion packet is not an intact cell.',
            'Conditional exterior carrier flight advances three coordinates through the native background; carrier formation and charge allocation remain assumed.',
        ]
        if hasattr(self,'interaction_result'):
            value['warnings'].extend([
                'Cell interactions are a local common-clock observation window with reused mobile resident prehistory; pre-existing adhesive deposits are not inferred.',
                'Local contacts, conditional bonds, unfinished residence and numerical censoring are distinct. Global theoretical acquisition rates retain the independent-response baseline.'])
        phase=getattr(self.volume,'phase',None)
        value['phase_times_s']=[] if phase is None else _clean(phase.time_s)
        if phase is not None:
            value['native_phase']=_clean(phase.diagnostics)
            value['native_phase_settings']=vars(phase.settings).copy()
            value['warnings']=[w for w in value['warnings'] if 'dynamic 3-D Taylor cone' not in w]
            value['warnings'].extend(phase.diagnostics.get('limits',[]))
            value['warnings'].append('The native moving phase has an independent short physical clock; it is not synchronized to the hundreds-of-seconds capillary residence time or proof of a steady spray.')
        return value

    def geometry(self,phase_frame=-1,field='methanol',**kwargs):
        import json
        from types import SimpleNamespace
        clock=kwargs.pop('data_clock','volume')
        if clock not in ('volume','phase'):raise ValueError('Field clock must be volume or phase')
        phase=getattr(self.volume,'phase',None)
        if clock=='phase' and phase is None:raise ValueError('No native moving-phase field has been computed')
        pf=int(np.clip(phase_frame if phase_frame>=0 else (len(phase.time_s)-1 if phase is not None else 0),
                       0,len(phase.time_s)-1 if phase is not None else 0))
        token=(id(phase),pf,field,clock,json.dumps(kwargs,sort_keys=True,default=str))
        if token in self._native_geometry:return self._native_geometry[token]
        # Coaxial quartz and the swept upstream pipe retain physical geometry.
        base=super().geometry(phase_frame)
        result={k:v for k,v in base.items() if k not in ('meshes','lines')}
        result.update(mode='native_cartesian_tip',meshes=[m for m in base['meshes']
                      if m.get('layer') in ('quartz','ms') or m.get('id')=='upstream_sample'],lines=[])
        try:
            from .fields import volume_geometry
        except ImportError:
            result['notes']=['Native volume is available; the independent volume renderer is unavailable.']
        else:
            kwargs=dict(kwargs);kwargs['include_context']=False
            kwargs.setdefault('slice_axis','both')
            selected_volume=self.volume
            if clock=='phase':
                selected_volume=SimpleNamespace(grid=phase.grid,
                    material=np.where(phase.material==2,2,np.where(phase.liquid_fraction[pf]>=.5,1,0)),
                    velocity=phase.velocity[pf],pressure=phase.pressure[pf],methanol=phase.methanol[pf],
                    salt_mol_m3=phase.salt_mol_m3[pf],potential=phase.potential[pf],electric=phase.electric[pf],
                    diagnostics=phase.diagnostics)
                if field=='electric_velocity':
                    comparison=getattr(self,'voltage_comparison',None)
                    if comparison is None:raise ValueError('Compute the 0 kV comparison before displaying voltage-driven velocity')
                    if any(not np.array_equal(a,b) for a,b in zip(phase.grid.edges,comparison.current_phase.grid.edges)):
                        raise ValueError('Phase comparison grid differs from the displayed phase grid')
                    selected_volume.delta_velocity=comparison.delta_phase_velocity
            extra=volume_geometry(selected_volume,self.data.config,self.path,field=field,**kwargs)
            extra['metadata'].update(data_clock=clock,field_time_s=None if clock=='volume' else float(phase.time_s[pf]))
            if clock=='phase' and field=='electric_velocity':extra['metadata']['field_time_s']=comparison.diagnostics['matched_phase_time_s']
            result['meshes'].extend(extra.get('meshes',[]));result['lines'].extend(extra.get('lines',[]))
            result['native_field']={k:extra[k] for k in ('scales','metadata') if k in extra}
        result.update(phase_time_s=None,phase_clock='unresolved_native_moving_interface',droplets=[])
        if phase is not None:
            from .phase_geometry import phase_geometry
            actual=phase_geometry(phase,pf,self.path,self.data.config.geometry.inner_tip)
            result['meshes'].extend(actual.pop('meshes'));result['lines'].extend(actual.pop('lines'))
            result.update(actual)
        result=_clean(result);self._native_geometry[token]=result
        if len(self._native_geometry)>12:self._native_geometry.pop(next(iter(self._native_geometry)))
        return result

    def flight(self,event_id):
        i=self._validate_event(event_id)
        if hasattr(self,'interaction_result') and i in self.interaction_result.event_ids:
            raise ValueError('This event has a separate interaction trajectory; the independent-response carrier flight cannot be reused')
        if i in self._flights:
            self._flights.move_to_end(i);return self._flights[i]
        from .native_flight import selected_native_flight
        k=int(self.model.records['template_id'][i]);_,xyz,angle,_=self._track_xyz(k)
        value=selected_native_flight(self,i,self.state_parameters)
        for row in value['history']:
            row['position']=self._global([row['x_m'],row['y_m'],row['z_m']]).tolist()
        value.update(conditional=True,source_id=self.source_id,
            coordinate_mode='native_cartesian_conditional_carrier',
            lip_position_m=self._global(xyz[-1]).tolist(),notes=[
                'Upstream, resolved tip and conditional exterior carrier retain computed 3-D positions and one source identity.',
                'Carrier radius and charge are assumed. No allocation of cells to native Taylor-cone droplets is claimed.',
                'Frozen native nozzle gas is added to approximate inlet suction; no shared nozzle/inlet momentum solve.',
                'Membrane validity is not viability; geometric aperture entry is not mass-spectral identification.'])
        value=_clean(value);self._flights[i]=value
        if len(self._flights)>8:self._flights.popitem(last=False)
        return value


def build_native_scene(base,path,volume,settings=None,progress=lambda value:None,config=None):
    """Build one continuous feed from the new fields without changing old data.

    ``config`` should be the actual flow operating point used to solve volume.
    With no config, volume's recorded inlet flow takes precedence over the
    reference source flow. ``settings`` may provide independent particle_count
    and saved_trajectories; defaults are 128/64 response samples.
    """
    from .version import solver_fingerprint
    volume.diagnostics.setdefault('model_fingerprint',solver_fingerprint())
    cfg=deepcopy(config or base.data.config);options=settings if isinstance(settings,dict) else {}
    cfg.cell.count=int(options.get('particle_count',min(cfg.cell.count,128)))
    cfg.numerics.saved_trajectories=int(options.get('saved_trajectories',min(cfg.numerics.saved_trajectories,64,cfg.cell.count)))
    q=volume.diagnostics.get('inlet_sample_flow_m3_s')
    if q is not None:cfg.fluid.sample_flow_ul_min=float(q)*60e9
    particles=simulate_native(cfg,volume,path,progress)
    cfg=particles.transport_config
    # Explicit reference meridian arrays satisfy the existing response loader;
    # native geometry uses the entire 3-D volume, never their revolution.
    xedges=volume.grid.edges[0];redges=xedges[xedges>=0]
    zedges=volume.grid.edges[2];r=(redges[:-1]+redges[1:])*.5;z=(zedges[:-1]+zedges[1:])*.5
    R,Z=np.meshgrid(r,z,indexing='ij');points=np.stack([R,np.zeros_like(R),Z],axis=-1)
    uv=volume.sample(points,'velocity');ev=volume.sample(points,'electric');g=cfg.geometry
    active=(Z<0)&((R<g.a)|((R>=g.b)&(R<g.c))|((Z>=g.inner_tip)&(R<g.c)))
    fields=dict(liquid_r_edges_m=redges,liquid_z_edges_m=zedges,liquid_active=active,
        velocity_r_m_s=uv[...,0],velocity_z_m_s=uv[...,2],pressure_pa=volume.sample(points,'pressure'),
        methanol_fraction=volume.sample(points,'methanol'),electric_r_edges_m=redges,electric_z_edges_m=zedges,
        potential_v=volume.sample(points,'potential'),field_r_v_m=ev[...,0],field_z_v_m=ev[...,2],
        trajectory_ids=particles.saved_ids,trajectory_time_s=particles.trajectory_time,
        trajectory_r_m=particles.trajectory_r,trajectory_z_m=particles.trajectory_z,
        trajectory_methanol_fraction=particles.trajectory_methanol,
        native_trajectory_x_m=particles.trajectory_x,native_trajectory_y_m=particles.trajectory_y,
        response_residence_time_s=particles.residence_time,response_exited=particles.exited)
    fields.update({f'bulk_trajectory_{key}':value for key,value in particles.bulk_history.items()})
    fields.update({f'cell_{key}':value for key,value in particles.stress.items()})
    summary=dict(particles=particles.diagnostics,native_volume=volume.diagnostics,
        warnings=list(volume.diagnostics.get('limits',[])),
        coupling=dict(converged=False,note='3-D finite iterations; inspect native residuals and coupling changes.'))
    data=NativeExplorerData(cfg,fields,summary,source=f'native3d derived from {base.data.source}')
    data.volume=volume
    return NativeSceneData(data,volume,particles,base,path)
