"""Create a fresh, editable 2-D preview when no numerical record exists."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from ..config import Config
from ..visualization.explorer_data import find_results
from ..workbench.compute import save_numeric


def studio_output_root(output_root=None):
    """Keep generated files in a writable workspace, outside the package."""
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    base = Path(os.environ.get('CYESI_OUTPUT_DIR', str(Path.cwd()/'outputs')))
    return base.expanduser().resolve()/'studio'


def _latest_axisymmetric(root):
    records = []
    for path in root.iterdir() if root.is_dir() else []:
        if not path.is_dir() or not (path/'fields.npz').is_file() or not (path/'config.json').is_file():
            continue
        if (path/'native_manifest.json').exists():
            continue
        marker = path/'record.json'
        if marker.exists():
            try:
                if json.loads(marker.read_text(encoding='utf-8')).get('status') != 'complete':
                    continue
            except (ValueError, OSError):
                continue
        records.append(path)
    return max(records, key=lambda path: (path/'fields.npz').stat().st_mtime) if records else None


def initial_source(source, scene, output_root, solver):
    """Retain explicit inputs, reuse local work, or solve a new startup preview."""
    if source is not None or scene is not None:
        return source
    existing = _latest_axisymmetric(output_root) or find_results()
    if existing is not None:
        return existing

    config = Config()
    config.cell.count = 64
    config.numerics.saved_trajectories = 64
    config.numerics.radial_cells = 24
    config.numerics.upstream_cells = 24
    config.numerics.mixing_cells = 40
    config.numerics.workers = 1
    config.numerics.particle_backend = 'numpy'
    config.spray.enabled = False
    config.validate()

    print('CyESI Studio: no saved results found. Computing a fresh 2-D startup preview.', flush=True)
    print('Preview: 64 cell responses, reduced mesh, transient spray disabled. All settings remain editable.', flush=True)

    def progress(message):
        message = str(message).replace('\u8ba1\u7b97 \u00b7 ', 'Solving: ').replace('\u590d\u7528\u7f13\u5b58 \u00b7 ', 'Reusing: ')
        print('  '+message, flush=True)

    result = solver.run(config, progress)
    target = output_root/('axisymmetric_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    save_numeric(result, target)
    print('Startup preview ready. Generated files: '+str(target), flush=True)
    return target
