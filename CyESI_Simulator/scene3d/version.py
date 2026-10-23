"""Invalidate physical caches when the implemented equations change."""
from functools import lru_cache
from pathlib import Path
import hashlib


@lru_cache(maxsize=1)
def solver_fingerprint():
    root=Path(__file__).resolve().parents[1]
    names=['scene3d/volume.py','scene3d/phase.py','scene3d/particles.py','scene3d/path.py',
        'scene3d/native.py','scene3d/native_flight.py','scene3d/cell_charge.py','scene3d/current.py',
        'scene3d/server.py','particle/long_tube.py','particle/forces.py','config.py',
        'scene3d/interactions.py','scene3d/transport_view.py',
        'particle/interactions.py','particle/axisymmetric_interactions.py',
        'cell_state/membrane.py','cell_state/flight.py','cell_state/inlet.py','cell_state/transport.py']
    digest=hashlib.sha256()
    for name in names:
        path=root/name
        if path.is_file():digest.update(name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()
