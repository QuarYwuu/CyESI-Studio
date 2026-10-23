"""Versioned, pickle-free 3D checkpoints.

Loading restores fields, trajectories, event histories and acquisition records.
The manifest is written last to mark a completed record.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import _clean

FORMAT = 'cyesi-native-scene'
VERSION = 1
MANIFEST = 'native_manifest.json'


def _json_write(path, value):
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2,
                                  allow_nan=False), encoding='utf-8')
    partial.replace(path)


def _arrays_write(path, values):
    arrays = {key: np.asarray(value) for key, value in values.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Native checkpoints contain numerical arrays only; object arrays are forbidden')
    partial = path.with_suffix('.partial.npz')
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(source):
    """Small read-only fingerprint for automatic reuse of an unchanged source."""
    path = Path(str(source))
    if not (path/'config.json').is_file() or not (path/'fields.npz').is_file():
        return None
    fields = (path/'fields.npz').stat()
    return dict(source=str(path.resolve()), config_sha256=_sha(path/'config.json'),
                fields_size=fields.st_size, fields_mtime_ns=fields.st_mtime_ns,
                summary_sha256=_sha(path/'summary.json') if (path/'summary.json').is_file() else None)


def _save_volume(directory, prefix, volume):
    names = ('material', 'velocity', 'pressure', 'methanol', 'salt_mol_m3', 'potential', 'electric')
    values = {name: getattr(volume, name) for name in names}
    values.update({axis+'_edges_m': edge for axis, edge in zip('xyz', volume.grid.edges)})
    values.update({'face_flux_'+axis: value for axis, value in zip('xyz', volume.face_flux)})
    if hasattr(volume, 'delta_velocity'):
        values['delta_velocity'] = volume.delta_velocity
    _arrays_write(directory/(prefix+'.npz'), values)
    _json_write(directory/(prefix+'.json'), dict(settings=vars(volume.settings), diagnostics=volume.diagnostics))


def _load_volume(directory, prefix):
    from .volume import CartesianGrid, VolumeResult, VolumeSettings
    arrays = _read_arrays(directory/(prefix+'.npz'))
    meta = _read_json(directory/(prefix+'.json'))
    grid = CartesianGrid(tuple(arrays.pop(axis+'_edges_m') for axis in 'xyz'))
    flux = tuple(arrays.pop('face_flux_'+axis) for axis in 'xyz')
    delta = arrays.pop('delta_velocity', None)
    result = VolumeResult(grid=grid, face_flux=flux, diagnostics=meta['diagnostics'],
                          settings=VolumeSettings(**meta['settings']).validate(), **arrays)
    if delta is not None:
        result.delta_velocity = delta
    return result


def _save_phase(directory, prefix, phase):
    if phase is None:
        return False
    values = {key: value for key, value in vars(phase).items() if isinstance(value, np.ndarray)}
    values.update({axis+'_edges_m': edge for axis, edge in zip('xyz', phase.grid.edges)})
    _arrays_write(directory/(prefix+'.npz'), values)
    _json_write(directory/(prefix+'.json'), dict(settings=vars(phase.settings), diagnostics=phase.diagnostics))
    return True


def _load_phase(directory, prefix):
    from .phase import PhaseResult, PhaseSettings
    from .volume import CartesianGrid
    values = _read_arrays(directory/(prefix+'.npz'))
    meta = _read_json(directory/(prefix+'.json'))
    grid = CartesianGrid(tuple(values.pop(axis+'_edges_m') for axis in 'xyz'))
    return PhaseResult(grid=grid, settings=PhaseSettings(**meta['settings']).validate(),
                       diagnostics=meta['diagnostics'], **values)


def _save_process(directory, model):
    # Explicit known state prevents future arbitrary attributes from silently
    # becoming executable or requiring pickle at reload.
    scalar_names = ('row', 'rates', 'source_rate', 'start_z', 'stage_names', 'has_bulk',
                    'ledger', 'scans', 'default_event', 'theory')
    array_names = ('delays', 'exited', 'stage_z', 'crossing', 'projection_direction',
                   'cohort', 'onsets', 'delivered')
    meta = {name: getattr(model, name) for name in scalar_names}
    arrays = {name: getattr(model, name) for name in array_names}
    tracks = []
    for index, track in enumerate(model.tracks):
        scalar = {}
        for name, value in track.items():
            if isinstance(value, np.ndarray):
                arrays[f'track_{index}__{name}'] = value
            else:
                scalar[name] = value
        tracks.append(scalar)
        arrays[f'local_y_{index}'] = model.local_y[index]
    meta.update(tracks=tracks, acquisition=model.acquisition.to_dict(),
                parameters=asdict(model.parameters))
    _arrays_write(directory/'process_arrays.npz', arrays)
    _arrays_write(directory/'events.npz', model.records)
    _json_write(directory/'process.json', meta)


def _load_process(directory, data):
    from ..acquisition.q_exactive import AcquisitionParameters
    from ..visualization.process_data import ProcessModel, ProcessParameters
    meta = _read_json(directory/'process.json')
    arrays = _read_arrays(directory/'process_arrays.npz')
    model = ProcessModel.__new__(ProcessModel)
    model.data = data
    model.acquisition = AcquisitionParameters(**meta.pop('acquisition')).validate()
    model.parameters = ProcessParameters(**meta.pop('parameters')).validate(model.acquisition)
    track_meta = meta.pop('tracks')
    for name, value in meta.items():
        setattr(model, name, value)
    model.tracks = []
    model.local_y = []
    for index, scalar in enumerate(track_meta):
        track = dict(scalar)
        prefix = f'track_{index}__'
        for name in list(arrays):
            if name.startswith(prefix):
                track[name[len(prefix):]] = arrays.pop(name)
        model.tracks.append(track)
        model.local_y.append(arrays.pop(f'local_y_{index}'))
    for name, value in arrays.items():
        setattr(model, name, value)
    model.records = _read_arrays(directory/'events.npz')
    model.by_id = {row['event_id']: row for row in model.ledger}
    model.flight = None
    model.stream = None  # Saved feed is authoritative; changing method builds a new model explicitly.
    ids = model.records.get('event_id', np.array([]))
    if not np.array_equal(ids, np.arange(len(ids))) or any(len(value) != len(ids) for value in model.records.values()):
        raise ValueError('Invalid continuous-feed identity arrays in native record')
    if len(model.records['template_id']) and (np.min(model.records['template_id']) < 0 or
            np.max(model.records['template_id']) >= len(model.tracks)):
        raise ValueError('Native feed references an unavailable particle response')
    return model


def save_scene(scene, directory):
    """Save a complete independent native record, refusing existing targets."""
    if not hasattr(scene, 'volume') or not hasattr(scene, 'native_particles'):
        raise ValueError('Only genuinely native 3-D scenes can use this checkpoint format')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    scene.data.config.save(directory/'config.json')
    _arrays_write(directory/'response_fields.npz', scene.data.fields)
    _save_volume(directory, 'volume', scene.volume)
    _save_process(directory, scene.model)
    _json_write(directory/'path.json', None if scene.path is None else scene.path.to_dict())
    _json_write(directory/'scene.json', dict(source_id=scene.source_id, data_source=scene.data.source,
        reference_source=scene.reference_source, summary=scene.data.summary,
        particle_diagnostics=scene.native_particles.diagnostics,
        charge_parameters=scene.charge_parameters.to_dict(),
        state_parameters=scene.state_parameters.to_dict(),
        flights={str(key): value for key, value in scene._flights.items()}))
    phase = getattr(scene.volume, 'phase', None)
    phase_saved = _save_phase(directory, 'phase3d', phase)
    comparison = getattr(scene, 'voltage_comparison', None)
    if comparison is not None:
        _save_volume(directory, 'zero_volume', comparison.zero_volume)
        _save_phase(directory, 'zero_phase', comparison.zero_phase)
        _save_phase(directory, 'comparison_phase', comparison.current_phase)
        _arrays_write(directory/'comparison.npz', {name: getattr(comparison, name) for name in
            ('delta_velocity', 'delta_phase_velocity', 'delta_liquid_fraction')})
        _json_write(directory/'comparison.json', comparison.diagnostics)
    interactions = getattr(scene, 'interaction_result', None)
    if interactions is not None:
        metadata, arrays = interactions.to_storage()
        _json_write(directory/'interactions.json', metadata)
        _arrays_write(directory/'interactions.npz', arrays)
    files = {p.name: _sha(p) for p in directory.iterdir() if p.is_file()}
    _json_write(directory/MANIFEST, dict(format=FORMAT, version=VERSION, complete=True,
        model_fingerprint=scene.volume.diagnostics.get('model_fingerprint','legacy_unversioned'),
        source_id=scene.source_id, phase=phase_saved, comparison=comparison is not None,
        interactions=interactions is not None,
        reference_fingerprint=source_fingerprint(scene.reference_source), files=files))
    return directory


def manifest(directory):
    directory = Path(directory)
    value = _read_json(directory/MANIFEST)
    if value.get('format') != FORMAT or value.get('version') != VERSION or value.get('complete') is not True:
        raise ValueError('Unsupported or incomplete native 3-D checkpoint')
    required = {'config.json', 'response_fields.npz', 'volume.npz', 'volume.json',
                'process_arrays.npz', 'process.json', 'events.npz', 'path.json', 'scene.json'}
    if not required <= set(value.get('files', {})):
        raise ValueError('Native 3-D checkpoint is missing required files')
    if any(Path(name).name != name or not (directory/name).is_file() for name in value['files']):
        raise ValueError('Native checkpoint files are missing or use invalid names')
    return value


def load_scene(directory, verify=True):
    """Restore native fields and event identities without running any solver."""
    from ..config import Config
    from ..cell_state.parameters import StateParameters
    from .native import NativeExplorerData, NativeSceneData
    from .path import TubePath
    directory = Path(directory).resolve()
    saved = manifest(directory)
    if verify and any(_sha(directory/name) != digest for name, digest in saved['files'].items()):
        raise ValueError('Native checkpoint integrity check failed; a file is incomplete or changed')
    meta = _read_json(directory/'scene.json')
    config = Config.load(directory/'config.json')
    volume = _load_volume(directory, 'volume')
    if saved['phase']:
        volume.phase = _load_phase(directory, 'phase3d')
    data = NativeExplorerData(config, _read_arrays(directory/'response_fields.npz'),
                              meta['summary'], source=meta['data_source'])
    data.volume = volume
    path_data = _read_json(directory/'path.json')
    path = None
    if path_data is not None:
        path = TubePath(*(np.asarray(path_data[name], float) for name in
            ('station_m', 'centers_m', 'tangents', 'normals', 'binormals')), path_data['metadata'])
        if (np.any(np.diff(path.station_m) <= 0) or any(value.shape != (len(path.station_m), 3)
                for value in (path.centers_m, path.tangents, path.normals, path.binormals))):
            raise ValueError('Invalid saved 3-D centreline')
    scene = NativeSceneData.__new__(NativeSceneData)
    scene.data = data
    scene.volume = volume
    scene.native_particles = SimpleNamespace(diagnostics=meta['particle_diagnostics'])
    scene.reference_source = meta['reference_source']
    scene.state_parameters = StateParameters(**meta['state_parameters']).validate()
    scene.model = _load_process(directory, data)
    scene.path = path
    scene.source_id = meta['source_id']
    scene._xyz, scene._geometry, scene._native_geometry = {}, {}, {}
    from .cell_charge import ChargeParameters
    scene.charge_parameters=ChargeParameters(**meta.get('charge_parameters',{})).validate()
    scene._charge_cache={}
    scene._flights = OrderedDict((int(key), value) for key, value in meta.get('flights', {}).items())
    scene._bulk_columns = {int(key): index for index, key in enumerate(data.fields.get('bulk_trajectory_ids', []))}
    if saved['comparison']:
        from .comparison import VoltageComparison
        delta = _read_arrays(directory/'comparison.npz')
        scene.voltage_comparison = VoltageComparison(volume, _load_volume(directory, 'zero_volume'),
            _load_phase(directory, 'comparison_phase'), _load_phase(directory, 'zero_phase'),
            diagnostics=_read_json(directory/'comparison.json'), **delta)
        volume.delta_velocity = scene.voltage_comparison.delta_velocity
    if saved.get('interactions'):
        from .interactions import InteractionResult
        scene.interaction_result = InteractionResult.from_storage(
            _read_json(directory/'interactions.json'), _read_arrays(directory/'interactions.npz'))
    scene.checkpoint_source = str(directory)
    return scene


def catalog_native(root):
    rows = []
    root = Path(root)
    for directory in root.iterdir() if root.is_dir() else []:
        if not directory.is_dir() or not (directory/MANIFEST).is_file():
            continue
        try:
            meta = manifest(directory)
        except (ValueError, KeyError, OSError):
            continue
        rows.append(dict(id='native:'+directory.name, label='3D · '+directory.name,
                         source=str(directory.resolve()), mode='native', source_id=meta['source_id'],
                         modified=(directory/MANIFEST).stat().st_mtime))
    return sorted(rows, key=lambda row: row['modified'], reverse=True)


def latest_compatible(root, source):
    from .version import solver_fingerprint
    wanted = source_fingerprint(source)
    if wanted is None:
        return None
    for row in catalog_native(root):
        meta=manifest(row['source'])
        if meta.get('reference_fingerprint') == wanted and meta.get('model_fingerprint')==solver_fingerprint():
            return Path(row['source'])
    return None
