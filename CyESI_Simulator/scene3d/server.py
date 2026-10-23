"""Loopback HTTP server for the 3D workspace.

A background worker runs numerical jobs. Browser assets are served locally.
"""
from __future__ import annotations

from copy import copy, deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock, Thread, Event
from urllib.parse import parse_qs, urlsplit
import gzip
import json
import math
import secrets
import traceback
import webbrowser

import numpy as np

from .data import SceneData, _clean
from .path import TubePath
from ..visualization.explorer_data import ExplorerData, find_results

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name('static')
FIELDS = {'methanol', 'speed', 'pressure', 'potential', 'electric', 'salt', 'gas_speed', 'electric_velocity'}


class SimulationCancelled(Exception):
    pass


def catalog(root=ROOT):
    outputs = Path(root)/'outputs'
    rows = []
    for p in outputs.iterdir() if outputs.exists() else []:
        if (p.is_dir() and p.name not in {'verification', 'scene3d', 'size_studies', 'length_studies'}
                and not (p/'verification_only.json').exists()
                and (p/'fields.npz').is_file() and (p/'config.json').is_file()):
            rows.append(dict(id=p.name, label=p.name, source=str(p.resolve()),
                             modified=(p/'fields.npz').stat().st_mtime))
    from .storage import catalog_native
    rows.extend(catalog_native(outputs/'scene3d'))
    return sorted(rows, key=lambda row: row['modified'], reverse=True)


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(name+' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


class SceneService:
    def __init__(self, source=None, scene=None, output_root=None):
        self.lock = RLock()
        self.cancel_event=Event()
        self.token = secrets.token_urlsafe(32)
        self.output_root = Path(output_root or ROOT/'outputs'/'scene3d')
        self.records = catalog()
        self.source = Path(source or find_results() or '')
        if scene is None:
            from .storage import MANIFEST, latest_compatible, load_scene
            cached = self.source if (self.source/MANIFEST).is_file() else (
                latest_compatible(self.output_root, self.source) if source is None else None)
            if cached is not None:
                scene = load_scene(cached)
            else:
                if not (self.source/'fields.npz').is_file():
                    raise ValueError('No saved numerical response. Run the existing simulator first, or pass --results.')
                from ..acquisition.q_exactive import AcquisitionParameters
                scene = SceneData(ExplorerData.load(self.source),
                                  AcquisitionParameters(signal_width_ms=20.))
        self.scene = scene
        self.revision = 1
        self.job = dict(state='idle', kind=None, message='Ready', error=None)
        self.saved_record = getattr(scene, 'checkpoint_source', None)
        self.refresh_records()

    def refresh_records(self):
        from .storage import catalog_native
        rows = catalog()
        known = {row['source'] for row in rows}
        rows.extend(row for row in catalog_native(self.output_root) if row['source'] not in known)
        with self.lock:
            self.records = sorted(rows, key=lambda row: row['modified'], reverse=True)

    def snapshot(self):
        with self.lock:
            return self.scene, self.revision

    def metadata(self):
        scene, revision = self.snapshot()
        value = scene.metadata()
        with self.lock:
            value.update(revision=revision, job=dict(self.job), session_token=self.token,
                         saved_record=self.saved_record)
        if hasattr(scene, 'volume'):
            value['current_estimate'] = scene.volume.diagnostics.get('current_estimate')
            value['current_mode'] = scene.volume.diagnostics.get('current_mode', 'specified')
        return _clean(value)

    def start(self, kind, work):
        with self.lock:
            if self.job['state'] == 'running':
                raise RuntimeError('A numerical job is already running; the previous scene remains available.')
            self.job = dict(state='running', kind=kind, message='Starting', error=None)
            self.cancel_event.clear()

        def progress(value):
            if self.cancel_event.is_set():raise SimulationCancelled()
            with self.lock:
                self.job['message'] = str(value)

        def run():
            try:
                result = work(progress)
                if self.cancel_event.is_set():raise SimulationCancelled()
                if result is not None and hasattr(result,'native_particles') and kind in ('method','flight','compare_voltage','charge_parameters','interactions'):
                    progress('Saving the updated independent native checkpoint')
                    record=self.save_native(result,result.path,result.volume,result.data.config)
                    with self.lock:self.saved_record=str(record)
                    self.refresh_records()
                with self.lock:
                    if result is not None:
                        self.scene = result
                        self.revision += 1
                    self.job.update(state='complete', message='Completed', error=None)
            except Exception as exc:
                if not isinstance(exc,SimulationCancelled):traceback.print_exc()
                with self.lock:
                    self.job.update(state='cancelled' if isinstance(exc,SimulationCancelled) else 'failed',
                        message='Current scene retained',error=None if isinstance(exc,SimulationCancelled) else str(exc))
        Thread(target=run, name='CyESI-3D-'+kind, daemon=True).start()
        return dict(accepted=True, kind=kind)

    def load(self, body):
        target = next((r for r in self.records if r['id'] == body.get('id')), None)
        if target is None:
            raise ValueError('Unknown source record')
        def work(progress):
            if target.get('mode') == 'native' or target['id'].startswith('native:'):
                from .storage import load_scene
                result = load_scene(target['source'])
                with self.lock:
                    self.saved_record = target['source']
                return result
            from ..acquisition.q_exactive import AcquisitionParameters
            with self.lock:self.saved_record=None
            return SceneData(ExplorerData.load(target['source']),
                             AcquisitionParameters(signal_width_ms=20.))
        return self.start('load', work)

    def method(self, body):
        scene, _ = self.snapshot()
        raw_resolution = _number(body.get('resolution', 17500), 'resolution', 17500, 140000)
        resolution = int(raw_resolution)
        if raw_resolution != resolution:
            raise ValueError('Resolution must be an integer')
        if resolution not in (17500, 35000, 70000, 140000):
            raise ValueError('Unsupported Q Exactive resolution')
        actual_it = _number(body.get('actual_it_ms', 30.), 'actual IT', .001,
                            scene.model.acquisition.max_injection_time_ms)
        width = _number(body.get('packet_width_ms', 20.), 'signal width', 0., 5000.)

        def work(progress):
            from ..visualization.process_data import ProcessParameters
            progress('Reclassifying the same cell-origin events against collection windows')
            acq = deepcopy(scene.model.acquisition); acq.signal_width_ms = width
            parameters = deepcopy(scene.model.parameters)
            parameters.resolution = resolution; parameters.actual_it_ms = actual_it
            if hasattr(scene, 'with_method'):
                return scene.with_method(acq, parameters)
            return SceneData(scene.data, acq, parameters, scene.state_parameters, scene.path)
        return self.start('method', work)

    def run3d(self, body):
        """Resolve the route's pressure feed, then solve a new independent record."""
        scene, _ = self.snapshot()
        config = deepcopy(scene.data.config)
        geometry = body.get('geometry', {})
        geometry_names = {'inner_id_um','inner_od_um','sheath_id_um','sheath_od_um',
            'gas_id_um','gas_od_um','tip_resolved_length_mm','inner_recess_mm',
            'gas_recess_mm','gas_supply_length_mm','ms_gap_mm','external_radius_mm'}
        if not isinstance(geometry, dict) or set(geometry)-geometry_names:
            raise ValueError('Unknown tube geometry parameter')
        for name, value in geometry.items():
            low = 0. if name in ('inner_recess_mm','gas_recess_mm') else .001
            setattr(config.geometry, name, _number(value, name, low, 10000.))
        if 'contact_distance_mm' in body:
            config.electrical.contact_distance_mm = _number(body['contact_distance_mm'], 'contact distance', .001, 10000.)
        config.supply.sample_control = body.get('sample_control', 'pressure')
        # The user reported a 0.2--1 uA operating range. This midpoint is a
        # scenario, not a measured current for any individual simulation.
        config.electrical.spray_current_na = 600.
        config.electrical.sheath_conductivity_s_m = .005
        current_mode = body.get('current_mode', 'specified')
        if current_mode not in ('specified', 'scaling'):
            raise ValueError('current_mode must be specified or scaling')
        # Parameter overrides belong to the new scene only.
        mapping = {
            'length_cm': (config.geometry, 'sample_length_mm', 1., 300., 10.),
            'sample_pressure_mbar': (config.supply, 'sample_pressure_mbar', .01, 30000., 1.),
            'sheath_flow_ul_min': (config.fluid, 'sheath_flow_ul_min', .01, 200., 1.),
            'sample_flow_ul_min': (config.fluid, 'sample_flow_ul_min', .001, 100., 1.),
            'voltage_kv': (config.electrical, 'voltage_kv', 0., 10., 1.),
            'quartz_zeta_mv': (config.electrical, 'quartz_zeta_mv', -200., 200., 1.),
            'sheath_conductivity_s_m': (config.electrical, 'sheath_conductivity_s_m', 1e-8, 10., 1.),
            'gas_pressure_kpa': (config.fluid, 'gas_pressure_kpa', 0., 10., 1.),
            'diameter_um': (config.cell, 'diameter_um', .1, 40., 1.),
            'cell_zeta_mv': (config.cell, 'zeta_mv', -100., 100., 1.),
            'concentration_per_ml': (config.cell, 'concentration_per_ml', 0., 1e9, 1.),
            'bulk_wall_capture_rate_s': (config.supply, 'wall_capture_rate_s', 0., 1e6, 1.),
            'bulk_wall_capture_gap_um': (config.supply, 'wall_capture_gap_um', .0001, 10., 1.),
        }
        for name, (obj, attr, low, high, factor) in mapping.items():
            if name in body:
                setattr(obj, attr, _number(body[name], name, low, high)*factor)
        if 'spray_current_na' in body:
            value = body['spray_current_na']
            config.electrical.spray_current_na = None if value is None else _number(value, 'spray current', 0., 1e5)
        if 'brownian_enabled' in body:
            if body['brownian_enabled'] is not None and not isinstance(body['brownian_enabled'], bool):
                raise ValueError('Brownian control must be true, false or null (automatic)')
            config.cell.brownian_enabled = body['brownian_enabled']
        count = _number(body.get('cells', 512), 'transport samples', 32, 10000)
        if count != int(count):
            raise ValueError('Transport sample count must be an integer')
        config.cell.count = int(count)
        config.numerics.saved_trajectories = min(config.numerics.saved_trajectories, config.cell.count)
        if body.get('path_kind') == 'custom':
            points = np.asarray(body.get('points_mm'), float)*.001
            if points.ndim != 2 or points.shape[1] != 3 or not 2 <= len(points) <= 64:
                raise ValueError('Provide 2..64 finite [x,y,z] waypoints in millimetres')
            path = TubePath.from_points(points, length_m=config.geometry.sample_length_mm*.001,
                                       tip_position=[0., 0., config.geometry.inner_tip])
            if np.linalg.norm(path.tangents[-1]-[0., 0., 1.]) > .02:
                raise ValueError('The last two path points must point along +Z toward the coaxial tip')
        else:
            path = TubePath.default(config,
                radius_mm=_number(body.get('bend_radius_mm', 10.), 'bend radius', .2, 300.),
                up_fraction=_number(body.get('up_fraction', .5), 'upward straight fraction', .01, .99))
        config.supply.orientation = 'custom'; config.supply.segments = path.segments_for_supply()
        from .volume import VolumeSettings
        allowed = set(VolumeSettings.__dataclass_fields__)
        supplied = body.get('volume', {})
        if not isinstance(supplied, dict) or set(supplied)-allowed:
            raise ValueError('Unknown 3-D field settings')
        settings = VolumeSettings(**({'flow_steps': 16, 'coupling_iterations': 3,
                                      'axial_cells': 48} | supplied)).validate()
        if 'external_radius_mm' in geometry:
            settings.gas_extent_mm = config.geometry.external_radius_mm
        if config.electrical.contact_distance_mm < abs(config.geometry.liquid_start)*1000:
            raise ValueError('The contact electrode must be upstream of the resolved 3-D inlet; increase contact distance or shorten the resolved tip')
        phase_enabled = body.get('phase_enabled', True)
        if not isinstance(phase_enabled, bool):
            raise ValueError('phase_enabled must be true or false')
        from .phase import PhaseSettings
        phase_options = body.get('phase', {})
        if not isinstance(phase_options, dict) or set(phase_options)-set(PhaseSettings.__dataclass_fields__):
            raise ValueError('Unknown 3-D phase settings')
        phase_settings = PhaseSettings(**phase_options).validate()
        config.validate()

        def work(progress):
            from .cache import reuse_concentration
            reused=reuse_concentration(scene,config,path,settings,phase_settings,phase_enabled,current_mode)
            if reused is not None:
                progress('Reusing identical physical fields and particle responses; updating continuous feed only')
                if reused is not scene:
                    record=self.save_native(reused,path,reused.volume,reused.data.config)
                    with self.lock:self.saved_record=str(record)
                    self.refresh_records()
                return reused
            from .volume import solve_volume
            from .native import build_native_scene
            from .current import estimate_current
            from ..fluid.supply import estimate_flow, resistance, hydrostatic_assist_pa
            from .cache import reuse_fields
            cached_volume=reuse_fields(scene,config,path,settings,phase_settings,phase_enabled,current_mode)
            if cached_volume is not None:
                config.fluid.sample_flow_ul_min=scene.data.config.fluid.sample_flow_ul_min
                config.electrical.spray_current_na=scene.data.config.electrical.spray_current_na
                progress('Reusing dilute fluid, electric and phase fields; updating finite-sphere transport')
                result=build_native_scene(scene,path,cached_volume,
                    settings={'particle_count':config.cell.count,'saved_trajectories':config.numerics.saved_trajectories},
                    progress=progress,config=config)
                record=self.save_native(result,path,cached_volume,config)
                with self.lock:self.saved_record=str(record)
                self.refresh_records()
                return result
            current_estimate = None
            def update_current():
                nonlocal current_estimate
                current_estimate = estimate_current(config)
                if current_mode == 'scaling':
                    config.electrical.spray_current_na = current_estimate['predicted_current_a']*1e9
            supply = {'control': config.supply.sample_control,
                      'feed_pressure_mbar': config.supply.sample_pressure_mbar,
                      'hydrostatic_assist_pa': hydrostatic_assist_pa(config)}
            if config.supply.sample_control == 'pressure':
                q = estimate_flow(config)
                if q <= 0:
                    raise ValueError('Sample pressure cannot overcome the upward hydrostatic head')
                config.fluid.sample_flow_ul_min = q
            progress('Solving actual Cartesian 3-D fields for the new route')
            volume = None
            if config.supply.sample_control == 'pressure':
                # p_tip already includes the prescribed outlet pressure. Keep
                # that offset in p_tip once, rather than subtracting it again.
                drive = config.supply.sample_pressure_mbar*100+hydrostatic_assist_pa(config)
                history = []
                previous = None
                for iteration in range(config.supply.pressure_iterations):
                    progress(f'3-D pressure closure {iteration+1}: Q={config.fluid.sample_flow_ul_min:.6g} uL/min')
                    update_current()
                    volume = solve_volume(config, settings, progress)
                    xx, yy = np.meshgrid(*volume.grid.centers[:2], indexing='ij')
                    inlet = (np.hypot(xx, yy)<config.geometry.a) & (volume.material[:, :, 0] == 1)
                    if not inlet.any():
                        raise ValueError('Native sample inlet is not resolved by this grid')
                    weights = volume.grid.widths[0][:, None]*volume.grid.widths[1][None, :]
                    tip_pressure = float(np.average(volume.pressure[:, :, 0][inlet], weights=weights[inlet]))
                    resolved_length = config.geometry.inner_tip-float(volume.grid.edges[2][0])
                    bulk_length = path.length_m-resolved_length
                    if bulk_length < -1e-10:
                        raise ValueError('Resolved inner tube is longer than the physical sample tube')
                    bulk_resistance = resistance(config, max(0., bulk_length))
                    q_now = config.fluid.sample_flow_ul_min
                    residual = bulk_resistance*config.fluid.qs+tip_pressure-drive
                    history.append(dict(sample_flow_ul_min=q_now, tip_inlet_pressure_pa=tip_pressure,
                                        pressure_residual_pa=float(residual)))
                    if not math.isfinite(residual):
                        raise ValueError('Native pressure closure produced a non-finite pressure')
                    if abs(residual) <= config.supply.pressure_tolerance_pa:
                        break
                    slope = resistance(config)/60e9
                    if previous is not None and abs(q_now-previous[0]) > 1e-12:
                        secant = (residual-previous[1])/(q_now-previous[0])
                        if math.isfinite(secant) and secant > 0:
                            slope = secant
                    next_q = q_now-residual/slope
                    previous = q_now, residual
                    if next_q <= 0:
                        if q_now < 1e-6:
                            raise ValueError('Resolved tip backpressure exceeds the sample supply pressure')
                        next_q = q_now*.1
                    config.fluid.sample_flow_ul_min = next_q
                else:
                    raise RuntimeError(f'Native pressure closure did not converge; residual {residual:.4g} Pa')
                supply.update(tip_inlet_pressure_pa=tip_pressure, pressure_residual_pa=float(residual),
                              converged=True, iterations=history,
                              scope='Developed upstream resistance + iterated native 3-D tip pressure; finite-volume momentum convergence is separate')
            else:
                update_current()
                volume = solve_volume(config, settings, progress)
            supply['actual_sample_flow_ul_min'] = config.fluid.sample_flow_ul_min
            volume.diagnostics['supply'] = supply
            volume.diagnostics['current_estimate'] = current_estimate
            volume.diagnostics['current_mode'] = current_mode
            volume.diagnostics['electrical_input_scope'] = dict(
                user_reported_current_range_na=[200., 1000.],
                scenario_spray_current_na=config.electrical.spray_current_na,
                current_scope=('cone-jet scaling scenario; not a measured current or a free-charge-conservation solution'
                    if current_mode == 'scaling' else 'open-circuit limiting case: zero imposed spray current'
                    if not config.electrical.spray_current_na else
                    'representative specified scenario, not a measured current for this numerical run'),
                supporting_conductivity_s_m=config.electrical.sheath_conductivity_s_m,
                supporting_conductivity_scope='editable unmeasured assumption; requires experimental calibration')
            if phase_enabled:
                from .phase import solve_phase
                progress('Advancing a short 3-D free-surface research transient')
                volume.phase = solve_phase(volume, config, settings=phase_settings, progress=progress)
            result = build_native_scene(scene, path, volume,
                settings={'particle_count': config.cell.count,
                          'saved_trajectories': config.numerics.saved_trajectories},
                progress=progress, config=config)
            result.reference_source = getattr(scene, 'reference_source', scene.data.source)
            progress('Saving an independent 3-D numerical record')
            record = self.save_native(result, path, volume, config)
            with self.lock:
                self.saved_record = str(record)
            self.refresh_records()
            return result
        return self.start('native3d', work)

    def interactions(self, body):
        """Run a local population study with common-clock events and fixed fields.

        Finite-window outcomes are separate from full-duration acquisition rates.
        """
        scene, _ = self.snapshot()
        if not hasattr(scene, 'volume'):
            raise ValueError('Compute native 3-D fields before studying cell interactions')
        from .interactions import InteractionParameters, simulate_interactions
        if body.get('clear') is True:
            def clear(progress):
                progress('Removing the local interaction overlay; retaining all baseline fields')
                result = copy(scene)
                if hasattr(result, 'interaction_result'):
                    del result.interaction_result
                return result
            return self.start('interactions', clear)
        if set(body)-set(InteractionParameters.__dataclass_fields__):
            raise ValueError('Unknown cell interaction parameter')
        parameters = InteractionParameters(**body).validate()
        if parameters.start_s+parameters.duration_s > scene.model.parameters.duration_s+1e-9:
            raise ValueError('Interaction observation window exceeds the continuous-feed record')
        def work(progress):
            progress('Advancing actual continuous event IDs on a shared interaction clock')
            study = simulate_interactions(scene, parameters, progress=progress)
            result = copy(scene)
            result.interaction_result = study
            return result
        return self.start('interactions', work)

    def compare_voltage(self, body):
        scene, _ = self.snapshot()
        if not hasattr(scene, 'volume'):
            raise ValueError('Run the native 3-D solver before comparing voltage-driven flow')
        def work(progress):
            from .comparison import solve_voltage_comparison
            result = solve_voltage_comparison(scene.data.config, scene.volume.settings,
                current_volume=scene.volume, current_phase=getattr(scene.volume, 'phase', None),
                progress=progress)
            with self.lock:
                scene.voltage_comparison = result
                scene.volume.delta_velocity = result.delta_velocity
            return scene
        return self.start('compare_voltage', work)

    def save_native(self, scene, path, volume, config):
        from .storage import save_scene
        target = self.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return save_scene(scene, target)


class Handler(BaseHTTPRequestHandler):
    server_version = 'CyESI3D/1'

    @property
    def service(self):
        return self.server.service

    def log_message(self, format, *args):
        pass

    def _same_origin(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {'http://'+host for host in allowed}

    def _send(self, status, value, content_type='application/json; charset=utf-8'):
        data = value if isinstance(value, bytes) else json.dumps(_clean(value), ensure_ascii=False, allow_nan=False).encode('utf-8')
        compressed = len(data)>2048 and 'gzip' in self.headers.get('Accept-Encoding', '')
        if compressed:
            data = gzip.compress(data, compresslevel=2)
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        if compressed:
            self.send_header('Content-Encoding', 'gzip')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._same_origin():
            return self._send(403, {'error': 'Local same-origin access only'})
        url = urlsplit(self.path); query = parse_qs(url.query)
        one = lambda key, default=None: query.get(key, [default])[0]
        try:
            if url.path == '/api/meta':
                return self._send(200, self.service.metadata())
            if url.path == '/api/records':
                return self._send(200, self.service.records)
            scene, revision = self.service.snapshot()
            if url.path == '/api/frame':
                value = scene.frame(_number(one('t', 0.), 'time', 0., scene.model.parameters.duration_s),
                                    None if one('selected') is None else one('selected'))
                value['revision'] = revision
                return self._send(200, value)
            if url.path == '/api/event':
                event_id = one('id')
                if one('next') is not None or one('trip') is not None:
                    records = scene.model.records
                    nozzle = np.asarray(records['nozzle_s'], float)
                    injection = np.asarray(records['injection_s'], float)
                    if one('next') is not None:
                        after = _number(one('next'), 'next event time', 0., scene.model.parameters.duration_s)
                        indices = np.flatnonzero(np.isfinite(nozzle) & (nozzle >= after+.001))
                        if not len(indices):
                            raise ValueError('No later resolved nozzle event in this observation interval')
                        event_id = int(indices[np.argmin(nozzle[indices])])
                    else:
                        indices = np.flatnonzero(np.isfinite(nozzle) & (injection >= 0.) & (nozzle < scene.model.parameters.duration_s))
                        if not len(indices):
                            raise ValueError('No complete entrance-to-nozzle trip in this observation interval')
                        event_id = int(indices[np.argmin(injection[indices])])
                return self._send(200, scene.event(event_id))
            if url.path == '/api/flight':
                event_id = scene._validate_event(one('id'))
                # Return the stored solution; numerical jobs run through the job endpoints.
                value = scene._flights.get(event_id)
                if value is None:
                    return self._send(404, {'error': 'Selected conditional flight has not been computed'})
                return self._send(200, value)
            if url.path == '/api/interactions':
                study = getattr(scene, 'interaction_result', None)
                if study is None:
                    return self._send(404, {'error': 'No continuous interaction study has been computed'})
                value = dict(parameters=study.parameters.to_dict(), diagnostics=study.diagnostics,
                             source_id=scene.source_id, revision=revision)
                if one('id') is not None:
                    event = scene.event(one('id'))
                    value['event_history'] = event.get('interaction_history')
                    if value['event_history'] is None:
                        raise ValueError('The selected event does not participate in this observation window')
                return self._send(200, value)
            if url.path == '/api/geometry':
                field = one('field', 'methanol')
                if field not in FIELDS:
                    raise ValueError('Unknown scalar field')
                phase = int(_number(one('phase', -1), 'phase frame', -1, 10000))
                if hasattr(scene, 'volume'):
                    value = scene.geometry(field=field, phase_frame=phase, data_clock=one('clock', 'volume'),
                        slice_axis=one('slice_axis', 'both'),
                        slice_position_m=_number(one('slice_position_m', 0.), 'slice position', -10., 10.))
                else:
                    value = scene.geometry(phase_frame=phase)
                value['revision'] = revision
                return self._send(200, value)
            if url.path == '/api/slice':
                from .slices import scene_slice
                value = scene_slice(scene, field=one('field', 'methanol'), axis=one('axis', 'x'),
                    position_m=_number(one('position', 0.), 'slice position', -10., 10.),
                    clock=one('clock', 'volume'),
                    phase_frame=int(_number(one('phase', -1), 'phase frame', -1, 10000)),
                    time_s=None if one('t') is None else _number(one('t'), 'slice time', 0., scene.model.parameters.duration_s))
                value['revision'] = revision
                return self._send(200, value)
            names = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js',
                     '/renderer.js': 'renderer.js', '/slices.js': 'slices.js', '/style.css': 'style.css'}
            if url.path in names:
                filename = names[url.path]
                mime = 'text/html; charset=utf-8' if filename.endswith('.html') else (
                    'text/css; charset=utf-8' if filename.endswith('.css') else 'text/javascript; charset=utf-8')
                return self._send(200, (STATIC/filename).read_bytes(), mime)
            return self._send(404, {'error': 'Not found'})
        except (ValueError, TypeError, KeyError) as exc:
            return self._send(400, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            return self._send(500, {'error': str(exc)})

    def do_POST(self):
        if not self._same_origin() or not secrets.compare_digest(self.headers.get('X-CyESI-Session', ''), self.service.token):
            return self._send(403, {'error': 'A local scene session is required'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536:
                raise ValueError('JSON request body must be 1..65536 bytes')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            route = urlsplit(self.path).path
            if route == '/api/load':
                result = self.service.load(body)
            elif route == '/api/cancel':
                self.service.cancel_event.set()
                result={'accepted':True,'message':'Stopping at the next solver checkpoint'}
            elif route == '/api/method':
                result = self.service.method(body)
            elif route == '/api/run3d':
                result = self.service.run3d(body)
            elif route == '/api/compare-voltage':
                result = self.service.compare_voltage(body)
            elif route == '/api/interactions':
                result = self.service.interactions(body)
            elif route == '/api/charge-parameters':
                scene, _ = self.service.snapshot()
                if not hasattr(scene,'set_charge_parameters'):
                    raise ValueError('Run the native 3-D solver before updating cell charging parameters')
                # Validate before starting the worker, then atomically replace
                # the diagnostic parameters without modifying transport forces.
                from .cell_charge import ChargeParameters
                options=dict(body)
                if 'intracellular_conductivity_s_m' in options:
                    options['cytosol_conductivity_s_m']=options.pop('intracellular_conductivity_s_m')
                if set(options)-set(ChargeParameters.__dataclass_fields__):
                    raise ValueError('Unknown cell charge parameter')
                ChargeParameters(**(scene.charge_parameters.to_dict()|options)).validate()
                def work(progress):
                    progress('Updating conditional cell charge response; reusing all flow and particle fields')
                    scene.set_charge_parameters(body)
                    return scene
                result=self.service.start('charge_parameters',work)
            elif route == '/api/flight':
                scene, _ = self.service.snapshot(); event_id = scene._validate_event(body.get('id'))
                def work(progress):
                    progress('Computing the selected conditional lipid-carrier scenario')
                    scene.flight(event_id)
                    return scene
                result = self.service.start('flight', work)
            else:
                return self._send(404, {'error': 'Not found'})
            self._send(202, result)
        except RuntimeError as exc:
            self._send(409, {'error': str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            self._send(400, {'error': str(exc)})
        except Exception as exc:
            traceback.print_exc(); self._send(500, {'error': str(exc)})


def create_server(source=None, scene=None, port=0, output_root=None):
    server = ThreadingHTTPServer(('127.0.0.1', int(port)), Handler)
    server.daemon_threads = True
    try:
        server.service = SceneService(source, scene, output_root)
    except Exception:
        server.server_close()
        raise
    return server


def launch(source=None, port=0, open_browser=True, auto3d=True):
    server = create_server(source=source, port=port)
    url = f'http://127.0.0.1:{server.server_port}'
    print('CyESI independent 3-D workspace: '+url, flush=True)
    print('Existing numerical records are read-only. Close this terminal to stop.', flush=True)
    if auto3d and not hasattr(server.service.scene, 'volume'):
        server.service.run3d({})
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
