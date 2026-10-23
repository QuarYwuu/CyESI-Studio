"""Experiment state, background jobs and dimension-specific caches.

Axisymmetric and Cartesian solutions are stored separately and share the
continuous-event interface. Switching views selects the corresponding record.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import RLock, Thread

import numpy as np

from ..config import Config
from ..scene3d.data import SceneData, _clean
from ..scene3d.server import SceneService, ROOT, _number
from ..scene3d.server import SimulationCancelled
from ..visualization.explorer_data import ExplorerData
from ..workbench.compute import WorkbenchSolver, digest, physical_key, reusable_output, save_numeric


def dimension(scene):
    return '3d' if hasattr(scene, 'volume') else '2d'


def portable_source(value):
    """Relocate source paths beneath the current outputs directory.

    The recorded absolute path remains in the source metadata.
    """
    original=Path(str(value))
    if (original/'config.json').is_file():return original
    normalized=str(value).replace('\\','/')
    marker='/outputs/'
    suffix=normalized.split(marker,1)[1] if marker in normalized else (
        normalized[len('outputs/'):] if normalized.startswith('outputs/') else None)
    if suffix is not None:
        relative=Path(suffix)
        if not relative.is_absolute() and '..' not in relative.parts and ':' not in suffix:
            candidate=ROOT/'outputs'/relative
            if (candidate/'config.json').is_file():return candidate
    return original


def relocate_scene(scene):
    if not hasattr(scene,'volume'):return scene
    for owner,name in ((scene,'reference_source'),(scene.data,'source')):
        value=getattr(owner,name,'')
        relocated=portable_source(value)
        if str(relocated)!=str(value):
            setattr(owner,'archived_'+name,str(value))
            setattr(owner,name,str(relocated))
    return scene


class StudioAxisymmetricScene(SceneData):
    """The existing r/z result with the same charge/contact UI as native 3-D."""

    def __init__(self, data, base=None, acquisition=None, parameters=None, path=None):
        from ..scene3d.cell_charge import ChargeParameters
        super().__init__(data, acquisition or (base.model.acquisition if base else None),
                         parameters or (base.model.parameters if base else None),
                         base.state_parameters if base else None,path=path)
        self.charge_parameters = deepcopy(getattr(base, 'charge_parameters', ChargeParameters()))
        self._charge_cache = {}
        self._field_provider = None
        self.reference_source = getattr(base, 'reference_source', data.source)

    @property
    def field_provider(self):
        if self._field_provider is None:
            from ..particle.axisymmetric_interactions import AxisymmetricTipField
            self._field_provider = AxisymmetricTipField(self)
        return self._field_provider

    def charge_samples(self, points):
        return self.field_provider.charge_samples(np.asarray(points, float))

    def set_charge_parameters(self, values):
        from ..scene3d.cell_charge import ChargeParameters
        values = dict(values)
        if 'intracellular_conductivity_s_m' in values:
            values['cytosol_conductivity_s_m'] = values.pop('intracellular_conductivity_s_m')
        if set(values)-set(ChargeParameters.__dataclass_fields__):
            raise ValueError('Unknown cell charge parameter')
        self.charge_parameters = ChargeParameters(**(self.charge_parameters.to_dict() | values)).validate()
        self._charge_cache.clear()

    def event(self, event_id):
        from ..scene3d.cell_charge import charge_at_track
        from ..scene3d.transport_view import add_interaction_event
        value = super().event(event_id)
        value['charge_history'] = charge_at_track(self, value['trajectory_index'])
        return _clean(add_interaction_event(self, value, value['event_id']))

    def frame(self, time_s, selected_id=None, max_cells=1200):
        from ..scene3d.cell_charge import charge_at_event
        from ..scene3d.transport_view import add_bulk_deposits, apply_interaction_frame
        value = super().frame(time_s, selected_id, max_cells)
        value['selected_charge'] = None if selected_id is None else charge_at_event(self, selected_id, time_s)
        value = add_bulk_deposits(self, value, time_s, value['selected_id'], max_cells)
        return _clean(apply_interaction_frame(self, value, time_s, value['selected_id'], max_cells))

    def with_method(self, acquisition, parameters):
        result = StudioAxisymmetricScene(self.data, self, acquisition, parameters, self.path)
        result._xyz = self._xyz
        result._geometry = self._geometry
        result._flights = self._flights
        result._field_provider = self._field_provider
        result._charge_cache = self._charge_cache
        if hasattr(self, 'interaction_result'):
            ids = self.interaction_result.event_ids
            if (not len(ids) or int(np.max(ids)) < len(result.model.records['event_id'])) and all(
                    np.array_equal(self.model.records[key][ids], result.model.records[key][ids], equal_nan=True)
                    for key in ('event_id', 'injection_s', 'template_id', 'nozzle_s')):
                result.interaction_result = self.interaction_result
        return result

    def metadata(self):
        value = super().metadata()
        value['charge_parameters'] = self.charge_parameters.to_dict()
        value['bulk_wall_adhesion'] = dict(on_rate_s=self.data.config.supply.wall_capture_rate_s,
            capture_gap_um=self.data.config.supply.wall_capture_gap_um,
            scope='Uncalibrated conditional irreversible long-tube capture, separate from tip contact')
        if hasattr(self, 'interaction_result'):
            value['interaction_study'] = dict(parameters=asdict(self.interaction_result.parameters),
                                               diagnostics=self.interaction_result.diagnostics)
        return _clean(value)

    def flight(self, event_id):
        if hasattr(self, 'interaction_result') and int(event_id) in self.interaction_result.event_ids:
            raise ValueError('An interaction trajectory cannot reuse its independent-response exterior flight')
        return super().flight(event_id)


class StudioService(SceneService):
    """One asynchronous worker; separate field caches; one editable experiment."""

    def __init__(self, source=None, scene=None, output_root=None):
        from .startup import initial_source, studio_output_root
        self._scenes = {}
        self._saved_records = {}
        self._two_d = WorkbenchSolver(capacity=4)
        self._response_cache = OrderedDict()
        self.export_records = []
        self.export_lock = RLock()
        self.study_results = {}
        output_root = studio_output_root(output_root)
        source = initial_source(source, scene, output_root, self._two_d)
        super().__init__(source, scene, output_root)
        relocate_scene(self.scene)
        self.studio_root = self.output_root
        if not hasattr(self.scene, 'volume'):
            self.scene = StudioAxisymmetricScene(self.scene.data, self.scene)
            self._two_d.seed_saved(self.scene.data)
        self._scenes[dimension(self.scene)] = self.scene
        self._saved_records[dimension(self.scene)] = self.saved_record
        # When starting from a native checkpoint, its immutable reference may
        # also supply the original two-dimensional numerical record.
        if dimension(self.scene) == '3d':
            reference = Path(str(getattr(self.scene, 'reference_source', '')))
            if (reference/'fields.npz').is_file() and (reference/'config.json').is_file():
                data = ExplorerData.load(reference)
                if not any(key.startswith('native_trajectory') for key in data.fields):
                    other = StudioAxisymmetricScene(data, self.scene)
                    self._scenes['2d'] = other
                    self._saved_records['2d'] = str(reference)
                    self._two_d.seed_saved(data)

    def refresh_records(self):
        super().refresh_records()
        root = self.output_root
        rows = list(self.records)
        known = {row['source'] for row in rows}
        for folder in root.iterdir() if root.exists() else []:
            if (folder.is_dir() and (folder/'fields.npz').is_file() and
                    (folder/'config.json').is_file() and str(folder.resolve()) not in known):
                marker = folder/'record.json'
                if marker.exists():
                    import json
                    try:
                        if json.loads(marker.read_text(encoding='utf-8')).get('status') != 'complete':
                            continue
                    except (ValueError, OSError):
                        continue
                rows.append(dict(id='studio2d:'+folder.name, label=' 2D  · '+folder.name,
                    source=str(folder.resolve()), mode='axisymmetric',
                    modified=(folder/'fields.npz').stat().st_mtime))
        with self.lock:
            self.records = sorted(rows, key=lambda row: row['modified'], reverse=True)

    def start(self, kind, work):
        # Publish the scene and its dimension cache in one critical section.
        # Exclude cancelled jobs from the result cache.
        with self.lock:
            if self.job['state'] == 'running':
                raise RuntimeError('A numerical job is already running; the current scene remains available.')
            previous_record = self.saved_record
            self.job = dict(state='running',kind=kind,message='Starting',error=None)
            self.cancel_event.clear()
        def progress(value):
            if self.cancel_event.is_set():raise SimulationCancelled()
            with self.lock:self.job['message']=str(value)
        def run():
            try:
                result=work(progress)
                if result is not None:relocate_scene(result)
                if self.cancel_event.is_set():raise SimulationCancelled()
                if result is not None and not hasattr(result,'volume') and not isinstance(result,StudioAxisymmetricScene):
                    result=StudioAxisymmetricScene(result.data,result,path=result.path)
                if (result is not None and hasattr(result,'native_particles') and
                        kind in ('method','flight','compare_voltage','charge_parameters','state_parameters','interactions')):
                    progress(' Saving 3D checkpoint ')
                    record=self.save_native(result,result.path,result.volume,result.data.config)
                    with self.lock:self.saved_record=str(record)
                    self.refresh_records()
                if self.cancel_event.is_set():raise SimulationCancelled()
                with self.lock:
                    if result is not None:
                        self.scene=result
                        self.revision+=1
                        mode=dimension(result)
                        self._scenes[mode]=result
                        self._saved_records[mode]=self.saved_record
                        if kind == 'state_parameters' and hasattr(self,'geometry_cache'):
                            self.geometry_cache.clear()
                    self.job.update(state='complete',message='Completed',error=None)
            except Exception as exc:
                if not isinstance(exc,SimulationCancelled):
                    import traceback
                    traceback.print_exc()
                with self.lock:
                    self.saved_record=previous_record
                    self.job.update(state='cancelled' if isinstance(exc,SimulationCancelled) else 'failed',
                        message='Current scene retained',error=None if isinstance(exc,SimulationCancelled) else str(exc))
        Thread(target=run,name='CyESI-Studio-'+kind,daemon=True).start()
        return dict(accepted=True,kind=kind)

    def studio_metadata(self):
        value = super().metadata()
        with self.lock:
            scene = self.scene
            value['studio'] = dict(active_mode=dimension(scene),
                available_modes=sorted(self._scenes),
                scenes={name: dict(source=item.data.source, source_id=item.source_id,
                                   label=' Axisymmetric ' if name == '2d' else ' 3D Cartesian ',
                                   saved_record=self._saved_records.get(name))
                        for name, item in self._scenes.items()},
                shared_parameters=scene.data.config.to_dict(),
                cache={'2d': dict(self._two_d.cache.counts),
                       'axisymmetric_response_entries': len(self._response_cache)},
                workflow=['setup','fields','transport','interactions','ordering','spray','acquisition','compare','library'],
                scope='A view switch reuses its numerical record; a dimension change requires its own solve. Continuous arrivals are shared in meaning, not fabricated identical trajectories across different physical models.')
        return _clean(value)

    def metadata(self):
        return self.studio_metadata()

    def mode(self, body):
        mode = str(body.get('mode', '')).lower()
        if mode not in ('2d', '3d'):
            raise ValueError('mode must be 2d or 3d')
        with self.lock:
            target = self._scenes.get(mode)
            active = self.scene
        if target is None:
            raise ValueError(' No solution for this dimension ,  Run the '+(' 2D ' if mode == '2d' else ' 3D ')+' Solver .')
        def work(progress):
            progress(' Loading cached '+(' 2D ' if mode == '2d' else ' 3D ')+' Numerical record ')
            result = target
            if (target.model.acquisition.to_dict() != active.model.acquisition.to_dict() or
                    vars(target.model.parameters) != vars(active.model.parameters)):
                result = target.with_method(deepcopy(active.model.acquisition),deepcopy(active.model.parameters))
            if result.state_parameters.to_dict() != active.state_parameters.to_dict():
                if result is target:result=copy(target)
                result.state_parameters=deepcopy(active.state_parameters)
                result._flights=OrderedDict()
                result._geometry={}
                if hasattr(result,'_native_geometry'):result._native_geometry={}
                with self.lock:
                    if hasattr(self,'geometry_cache'):self.geometry_cache.clear()
            if result.charge_parameters.to_dict() != active.charge_parameters.to_dict():
                if result is target:result=copy(target)
                result.charge_parameters=deepcopy(active.charge_parameters)
                result._charge_cache={}
            with self.lock:
                self.saved_record = self._saved_records.get(mode) if result is target else None
            return result
        return self.start('mode', work)

    def _configuration(self, body):
        """Merge complete/partial Config sections and legacy flat controls."""
        scene, _ = self.snapshot()
        data = scene.data.config.to_dict()
        nested = body.get('config', {})
        if not isinstance(nested, dict):
            raise ValueError('config must be a JSON object')
        for section, values in nested.items():
            if section not in data or not isinstance(values, dict):
                raise ValueError('Unknown config section: '+str(section))
            if set(values)-set(data[section]):
                raise ValueError('Unknown '+section+' parameter')
            data[section].update(values)
        config = Config.from_dict(data)
        geometry = body.get('geometry', {})
        if not isinstance(geometry, dict) or set(geometry)-set(vars(config.geometry)):
            raise ValueError('Unknown tube geometry parameter')
        for name, value in geometry.items():
            setattr(config.geometry, name, value)
        mapping = {
            'length_cm': ('geometry','sample_length_mm',10.),
            'sample_pressure_mbar': ('supply','sample_pressure_mbar',1.),
            'sample_control': ('supply','sample_control',None),
            'sheath_flow_ul_min': ('fluid','sheath_flow_ul_min',1.),
            'sample_flow_ul_min': ('fluid','sample_flow_ul_min',1.),
            'voltage_kv': ('electrical','voltage_kv',1.),
            'quartz_zeta_mv': ('electrical','quartz_zeta_mv',1.),
            'sheath_conductivity_s_m': ('electrical','sheath_conductivity_s_m',1.),
            'contact_distance_mm': ('electrical','contact_distance_mm',1.),
            'spray_current_na': ('electrical','spray_current_na',None),
            'gas_pressure_kpa': ('fluid','gas_pressure_kpa',1.),
            'diameter_um': ('cell','diameter_um',1.),
            'cell_zeta_mv': ('cell','zeta_mv',1.),
            'concentration_per_ml': ('cell','concentration_per_ml',1.),
            'brownian_enabled': ('cell','brownian_enabled',None),
            'cells': ('cell','count',None),
            'bulk_wall_capture_rate_s': ('supply','wall_capture_rate_s',1.),
            'bulk_wall_capture_gap_um': ('supply','wall_capture_gap_um',1.),
        }
        for name, (section, attr, factor) in mapping.items():
            if name in body:
                value = body[name]
                if factor is not None:
                    if isinstance(value, bool):
                        raise ValueError(name+' must be numeric')
                    value = float(value)*factor
                setattr(getattr(config, section), attr, value)
        if 'phase_enabled' in body:
            if not isinstance(body['phase_enabled'], bool):
                raise ValueError('phase_enabled must be boolean')
            config.spray.enabled = body['phase_enabled']
        config.numerics.saved_trajectories = min(config.numerics.saved_trajectories, config.cell.count)
        return config.validate()

    def _path(self, config, body):
        from ..scene3d.path import TubePath
        route_keys = ('path_kind','points_mm','bend_radius_mm','up_fraction')
        if not any(name in body for name in route_keys):
            current, _ = self.snapshot()
            prior = current if current.path is not None else self._scenes.get('3d')
            existing = None if prior is None else prior.path
            if existing is not None:
                if (abs(existing.length_m-config.geometry.sample_length_mm*.001)<1e-12 and
                        np.linalg.norm(existing.centers_m[-1]-[0.,0.,config.geometry.inner_tip])<1e-12):
                    return existing
                if existing.metadata.get('kind') == 'up_bend_horizontal':
                    return TubePath.default(config,radius_mm=existing.metadata['bend_radius_m']*1000,
                                            up_fraction=existing.metadata['up_fraction'])
                return TubePath.from_points(existing.metadata.get('control_points_m',existing.centers_m),
                    length_m=config.geometry.sample_length_mm*.001,
                    tip_position=[0.,0.,config.geometry.inner_tip])
        if body.get('path_kind') == 'custom':
            points = np.asarray(body.get('points_mm'), float)*.001
            if points.ndim != 2 or points.shape[1] != 3 or not 2 <= len(points) <= 64:
                raise ValueError('Provide 2..64 finite [x,y,z] waypoints in millimetres')
            path = TubePath.from_points(points, length_m=config.geometry.sample_length_mm*.001,
                                       tip_position=[0.,0.,config.geometry.inner_tip])
            if np.linalg.norm(path.tangents[-1]-[0.,0.,1.]) > .02:
                raise ValueError('The terminal path must point toward +Z')
        else:
            path = TubePath.default(config,
                radius_mm=_number(body.get('bend_radius_mm',10.),'bend radius',.2,300.),
                up_fraction=_number(body.get('up_fraction',.5),'upward fraction',.01,.99))
        return path

    def run2d(self, body):
        scene, _ = self.snapshot()
        config = self._configuration(body)
        path = None
        # The axisymmetric local nozzle still supports the shared segmented
        # long-pipe transport. A requested route updates projected gravity.
        if any(name in body for name in ('path_kind','points_mm','bend_radius_mm','up_fraction')):
            path = self._path(config, body)
            config.supply.orientation = 'custom'
            config.supply.segments = path.segments_for_supply()
        key = digest(config.to_dict())
        def work(progress):
            cached = self._response_cache.get(key)
            base = self._scenes.get('2d')
            if cached is not None:
                progress(' Loading matching axisymmetric results ')
                data = cached
                with self.lock:self.saved_record = getattr(data,'studio_record',None)
            elif (base is not None and physical_key(base.data.config) == physical_key(config)
                    and base.data.config.events == config.events and reusable_output(base.data)):
                progress(' Reusing axisymmetric fields and particle responses ,  Updating concentration  /  Event rate ')
                data = base.data.with_concentration(config.cell.concentration_per_ml)
                # An in-memory concentration edit has not rewritten its source
                # archive. Do not describe that older record as this new run.
                with self.lock:
                    self.saved_record = (self._saved_records.get('2d') if data is base.data else None)
            else:
                progress(' Solving axisymmetric flow  /  Solvent  /  Ions  /  Finite spheres  /  Two-phase fields ')
                solved = self._two_d.run(config, progress)
                if self.cancel_event.is_set():
                    from ..scene3d.server import SimulationCancelled
                    raise SimulationCancelled()
                target = self.output_root/('axisymmetric_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                data = save_numeric(solved, target)
                data.studio_record = str(target)
                with self.lock:
                    self.saved_record = str(target)
                self.refresh_records()
            self._response_cache[key] = data
            self._response_cache.move_to_end(key)
            while len(self._response_cache) > 5:
                self._response_cache.popitem(last=False)
            return StudioAxisymmetricScene(data, scene,path=path)
        return self.start('axisymmetric2d', work)

    def interactions(self, body):
        scene, _ = self.snapshot()
        if hasattr(scene, 'volume'):
            return super().interactions(body)
        from ..particle.interactions import InteractionParameters, simulate_interactions
        if body.get('clear') is True:
            def clear(progress):
                result = copy(scene)
                if hasattr(result, 'interaction_result'):
                    del result.interaction_result
                return result
            return self.start('interactions', clear)
        if set(body)-set(InteractionParameters.__dataclass_fields__):
            raise ValueError('Unknown cell interaction parameter')
        parameters = InteractionParameters(**body).validate()
        if parameters.start_s+parameters.duration_s > scene.model.parameters.duration_s+1e-9:
            raise ValueError('Interaction window exceeds the continuous source record')
        def work(progress):
            progress(' Advancing finite spheres in the axisymmetric field  /  Binding states ')
            study = simulate_interactions(scene, parameters, progress=progress, field_provider=scene.field_provider)
            result = copy(scene)
            result.interaction_result = study
            study.diagnostics['saved_bulk_azimuth'] = bool('bulk_trajectory_x_m' in scene.data.fields)
            return result
        return self.start('interactions', work)

    def method(self, body):
        body = dict(body)
        if 'maximum_it_ms' in body:
            if 'max_injection_time_ms' in body and body['max_injection_time_ms'] != body['maximum_it_ms']:
                raise ValueError('Conflicting Maximum IT values')
            body['max_injection_time_ms'] = body.pop('maximum_it_ms')
        scene, _ = self.snapshot()
        acq = deepcopy(scene.model.acquisition)
        parameters = deepcopy(scene.model.parameters)
        nested = body.get('acquisition', {})
        if not isinstance(nested, dict) or set(nested)-set(vars(acq)):
            raise ValueError('Unknown acquisition parameter')
        for name, value in nested.items():
            setattr(acq, name, value)
        mapping = {'packet_width_ms':'signal_width_ms','max_injection_time_ms':'max_injection_time_ms',
                   'microscans':'microscans','overhead_ms':'overhead_ms',
                   'signal_delivery_fraction':'signal_delivery_fraction',
                   'extra_record_delay_ms':'extra_record_delay_ms'}
        for name, target in mapping.items():
            if name in body:
                setattr(acq, target, body[name])
        if 'max_injection_time_ms' in body or 'max_injection_time_ms' in nested:
            acq.actual_injection_times_ms = [min(x, acq.max_injection_time_ms) for x in acq.actual_injection_times_ms]
            parameters.actual_it_ms = min(parameters.actual_it_ms, acq.max_injection_time_ms)
        for name in ('resolution','actual_it_ms','duration_s','startup','packet_delay_ms'):
            if name in body:
                setattr(parameters, name, body[name])
        acq.validate()
        parameters.validate(acq)
        def work(progress):
            progress(' Updating  Q Exactive  Acquisition gates ,  Injection times and event outcomes ;  Current physical fields ')
            return scene.with_method(acq, parameters)
        return self.start('method', work)

    def state_parameters(self, body):
        """Update a conditional lipid-carrier/inlet scenario, never CFD inputs."""
        from ..cell_state.parameters import StateParameters
        if not isinstance(body, dict):
            raise ValueError('State parameters must be a JSON object')
        options = body.get('state_parameters',body)
        if not isinstance(options, dict) or set(options)-set(StateParameters.__dataclass_fields__):
            raise ValueError('Unknown conditional lipid-carrier parameter')
        if 'state_parameters' in body and set(body)-{'state_parameters'}:
            raise ValueError('Pass state_parameters or its fields directly, not both')
        scene, _ = self.snapshot()
        values = scene.state_parameters.to_dict()
        values.update(options)
        parameters = StateParameters(**values).validate()
        def work(progress):
            progress(' Updating lipid carriers  /  MS inlet settings ;  Updating transfer with current internal fields ')
            result = copy(scene)
            result.state_parameters = deepcopy(parameters)
            result._flights = OrderedDict()
            # Inlet aperture geometry belongs to these scenario parameters.
            # A fresh geometry cache is necessary even when CFD is reused.
            result._geometry = {}
            if hasattr(result,'_native_geometry'):
                result._native_geometry = {}
            return result
        return self.start('state_parameters',work)

    def run3d(self, body):
        """Native solver with full Config merge and dependency-aware reuse."""
        active, _ = self.snapshot()
        config = self._configuration(body)
        path = self._path(config, body)
        config.supply.orientation = 'custom'
        config.supply.segments = path.segments_for_supply()
        from ..scene3d.volume import VolumeSettings
        from ..scene3d.phase import PhaseSettings
        prior = self._scenes.get('3d', active)
        volume_options = body.get('volume', {})
        phase_options = body.get('phase', {})
        if not isinstance(volume_options, dict) or set(volume_options)-set(VolumeSettings.__dataclass_fields__):
            raise ValueError('Unknown 3-D field settings')
        if not isinstance(phase_options, dict) or set(phase_options)-set(PhaseSettings.__dataclass_fields__):
            raise ValueError('Unknown 3-D phase settings')
        defaults = vars(prior.volume.settings) if hasattr(prior,'volume') else dict(flow_steps=16,coupling_iterations=3,axial_cells=48)
        settings = VolumeSettings(**(defaults | volume_options)).validate()
        if 'external_radius_mm' in body.get('geometry', {}):
            settings.gas_extent_mm = config.geometry.external_radius_mm
        old_phase = getattr(getattr(prior,'volume',None),'phase',None)
        phase_settings = PhaseSettings(**((vars(old_phase.settings) if old_phase is not None else {}) | phase_options)).validate()
        phase_enabled = config.spray.enabled
        current_mode = body.get('current_mode',getattr(getattr(prior,'volume',None),'diagnostics',{}).get('current_mode','specified'))
        if current_mode not in ('specified','scaling'):
            raise ValueError('current_mode must be specified or scaling')
        if config.electrical.contact_distance_mm < abs(config.geometry.liquid_start)*1000:
            raise ValueError('The contact electrode must remain upstream of the resolved tip inlet')
        config.validate()
        def work(progress):
            from ..scene3d.cache import reuse_concentration, reuse_fields
            from ..scene3d.volume import solve_volume
            from ..scene3d.native import build_native_scene
            from ..scene3d.current import estimate_current
            from ..fluid.supply import estimate_flow, resistance, hydrostatic_assist_pa
            reused = reuse_concentration(prior,config,path,settings,phase_settings,phase_enabled,current_mode)
            if reused is not None:
                progress(' Reusing Cartesian fields  /  Two-phase  /  Finite-sphere responses ,  Updating continuous source ')
                # Preserve this workspace's active acquisition settings when
                # the cached native solution was last viewed under another method.
                if reused.model.acquisition != active.model.acquisition or reused.model.parameters != active.model.parameters:
                    reused = reused.with_method(active.model.acquisition,active.model.parameters)
                if reused is not prior:
                    record = self.save_native(reused,path,reused.volume,reused.data.config)
                    with self.lock:self.saved_record = str(record)
                    self.refresh_records()
                return reused
            volume = reuse_fields(prior,config,path,settings,phase_settings,phase_enabled,current_mode)
            if volume is not None:
                config.fluid.sample_flow_ul_min = prior.data.config.fluid.sample_flow_ul_min
                config.electrical.spray_current_na = prior.data.config.electrical.spray_current_na
                progress(' Reusing 3D flow  /  Electric field  /  Two-phase ,  Updating particle responses ')
            else:
                supply = dict(control=config.supply.sample_control,
                    feed_pressure_mbar=config.supply.sample_pressure_mbar,
                    hydrostatic_assist_pa=hydrostatic_assist_pa(config))
                def solve_at_flow():
                    current = estimate_current(config)
                    if current_mode == 'scaling':
                        config.electrical.spray_current_na = current['predicted_current_a']*1e9
                    return solve_volume(config,settings,progress),current
                if config.supply.sample_control == 'pressure':
                    config.fluid.sample_flow_ul_min = estimate_flow(config)
                    if config.fluid.sample_flow_ul_min <= 0:
                        raise ValueError('Sample pressure cannot overcome the route hydrostatic head')
                    drive = config.supply.sample_pressure_mbar*100+hydrostatic_assist_pa(config)
                    previous = None
                    history = []
                    for iteration in range(config.supply.pressure_iterations):
                        progress(f' 3D pressure-flow coupling  {iteration+1} · Q={config.fluid.sample_flow_ul_min:.6g} µL/min')
                        volume,current = solve_at_flow()
                        xx,yy = np.meshgrid(*volume.grid.centers[:2],indexing='ij')
                        inlet = (np.hypot(xx,yy)<config.geometry.a)&(volume.material[:,:,0] == 1)
                        if not inlet.any():raise ValueError('Native sample inlet is unresolved on this mesh')
                        weights = volume.grid.widths[0][:,None]*volume.grid.widths[1][None,:]
                        tip_pressure = float(np.average(volume.pressure[:,:,0][inlet],weights=weights[inlet]))
                        resolved_length = config.geometry.inner_tip-float(volume.grid.edges[2][0])
                        bulk_length = path.length_m-resolved_length
                        if bulk_length < -1e-10:raise ValueError('Resolved tip exceeds physical sample length')
                        residual = resistance(config,max(0.,bulk_length))*config.fluid.qs+tip_pressure-drive
                        q = config.fluid.sample_flow_ul_min
                        history.append(dict(sample_flow_ul_min=q,tip_inlet_pressure_pa=tip_pressure,pressure_residual_pa=float(residual)))
                        if not np.isfinite(residual):raise ValueError('Nonfinite native pressure closure')
                        if abs(residual) <= config.supply.pressure_tolerance_pa:break
                        slope = resistance(config)/60e9
                        if previous is not None and abs(q-previous[0])>1e-12:
                            secant = (residual-previous[1])/(q-previous[0])
                            if np.isfinite(secant) and secant>0:slope=secant
                        next_q = q-residual/slope
                        previous = q,residual
                        if next_q <= 0:
                            if q<1e-6:raise ValueError('Tip backpressure exceeds the sample pressure')
                            next_q=q*.1
                        config.fluid.sample_flow_ul_min=next_q
                    else:
                        raise RuntimeError(f'Native pressure closure did not converge: {residual:.4g} Pa')
                    supply.update(tip_inlet_pressure_pa=tip_pressure,pressure_residual_pa=float(residual),converged=True,
                                  iterations=history,scope='Developed upstream resistance plus native tip pressure; momentum convergence is separate')
                else:
                    volume,current=solve_at_flow()
                supply['actual_sample_flow_ul_min']=config.fluid.sample_flow_ul_min
                volume.diagnostics.update(supply=supply,current_estimate=current,current_mode=current_mode,
                    electrical_input_scope=dict(user_reported_current_range_na=[200.,1000.],
                        scenario_spray_current_na=config.electrical.spray_current_na,
                        current_scope='Editable specified or scaling scenario; a measured current is not inferred',
                        supporting_conductivity_s_m=config.electrical.sheath_conductivity_s_m))
                if phase_enabled:
                    from ..scene3d.phase import solve_phase
                    progress(' Computing charged interface transient ')
                    volume.phase=solve_phase(volume,config,settings=phase_settings,progress=progress)
            result=build_native_scene(active,path,volume,
                settings=dict(particle_count=config.cell.count,saved_trajectories=config.numerics.saved_trajectories),
                progress=progress,config=config)
            result.reference_source=getattr(active,'reference_source',active.data.source)
            record=self.save_native(result,path,volume,config)
            with self.lock:self.saved_record=str(record)
            self.refresh_records()
            return result
        return self.start('native3d',work)
