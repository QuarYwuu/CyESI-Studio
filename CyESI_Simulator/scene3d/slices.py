"""Cell-constant cuts through Cartesian finite-volume fields.

Each cut uses its containing slab and material mask. Coordinates remain
physical, without the axial display compression.
"""
from __future__ import annotations

from types import SimpleNamespace
import math
import numpy as np

from .fields import _field, _scale


def flat_slice(volume, field="methanol", axis="x", position_m=0.):
    """Return row-major scalar/material arrays and physical in-plane edges.

    Longitudinal cuts place z horizontally; a z cut places x horizontally.
    ``values[row][column]`` belongs to the corresponding physical edge cell.
    Masked values are JSON null, while material codes remain available to draw
    quartz separately from a gas/liquid field which was intentionally masked.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("Slice axis must be x, y or z")
    position = float(position_m)
    if not math.isfinite(position):
        raise ValueError("Slice position must be finite")
    k = "xyz".index(axis)
    normal_edges = volume.grid.edges[k]
    if not normal_edges[0] <= position <= normal_edges[-1]:
        raise ValueError("Slice lies outside the native 3-D volume")
    index = int(np.clip(np.searchsorted(normal_edges, position, side="right")-1,
                        0, len(normal_edges)-2))
    horizontal, vertical = ((2, 1) if k == 0 else (2, 0) if k == 1 else (0, 1))
    remaining = [j for j in range(3) if j != k]
    values, active = _field(volume, field)
    scale = _scale(values, active, field)

    def plane(array):
        cut = np.take(array, index, axis=k)
        # Reorder each physical plane to [vertical, horizontal].
        return np.transpose(cut, (remaining.index(vertical), remaining.index(horizontal)))

    selected, enabled = plane(values), plane(active)
    material = plane(volume.material)
    rows = [[float(v) if ok else None for v, ok in zip(row, mask)]
            for row, mask in zip(selected, enabled)]
    shape = list(selected.shape)
    vector = volume.electric if field in ("potential", "electric") else (
        volume.delta_velocity if field == "electric_velocity" else volume.velocity)
    vectors = []
    stride = tuple(max(1, int(np.ceil(n/16))) for n in shape)
    vector_plane = np.take(vector, index, axis=k)
    vector_plane = np.transpose(vector_plane,
        (remaining.index(vertical), remaining.index(horizontal), 2))
    for row in range(0, shape[0], stride[0]):
        for column in range(0, shape[1], stride[1]):
            v = vector_plane[row, column]
            if not enabled[row, column] or material[row, column] == 2 or not np.isfinite(v).all():
                continue
            vectors.append([float(volume.grid.centers[horizontal][column]),
                float(volume.grid.centers[vertical][row]), float(v[horizontal]), float(v[vertical])])
    return {
        "mode": "native_cartesian_flat_cut", "axis": axis,
        "position_m": position, "plane_index": index,
        "cell_slab_m": [float(normal_edges[index]), float(normal_edges[index+1])],
        "axis_ranges_m": {a: [float(e[0]), float(e[-1])]
                          for a, e in zip("xyz", volume.grid.edges)},
        "horizontal_axis": "xyz"[horizontal], "vertical_axis": "xyz"[vertical],
        "horizontal_edges_m": volume.grid.edges[horizontal].tolist(),
        "vertical_edges_m": volume.grid.edges[vertical].tolist(),
        "shape": shape, "values": rows, "material": material.astype(int).tolist(),
        "active_cells": int(np.count_nonzero(enabled)), "scale": scale,
        "vectors": vectors, "vector_unit": "V/m" if field in ("potential", "electric") else "m/s",
        "material_codes": {"gas": 0, "liquid": 1, "quartz": 2},
        "coordinate_system": "Native Cartesian tip coordinates; middle lip z = 0; metres",
        "sampling": "Containing finite-volume cell slab; cell-constant values; no interpolation across material masks",
        "notes": [
            "This is a direct cut of the solved three-dimensional Cartesian volume, not an axisymmetric reconstruction.",
            "The plane position is exact; field values are constant within the reported finite-volume slab.",
            "Cell outlines are true sphere-plane intersections at the main transport clock, with no display magnification.",
        ],
    }


def scene_slice(scene, field="methanol", axis="x", position_m=0., clock="volume",
                phase_frame=-1, time_s=None):
    """Select the same physical field/clock as the native three-dimensional view."""
    if not hasattr(scene, "volume"):
        raise ValueError("A native three-dimensional result is required for Cartesian cuts")
    if clock not in ("volume", "phase"):
        raise ValueError("Field clock must be volume or phase")
    volume = scene.volume
    field_time = None
    phase = getattr(volume, "phase", None)
    selected_frame = None
    if clock == "phase":
        if phase is None or not len(phase.time_s):
            raise ValueError("No native moving-phase field has been computed")
        selected_frame = int(np.clip(phase_frame if phase_frame >= 0 else len(phase.time_s)-1,
                                     0, len(phase.time_s)-1))
        volume = SimpleNamespace(grid=phase.grid,
            material=np.where(phase.material == 2, 2, np.where(phase.liquid_fraction[selected_frame] >= .5, 1, 0)),
            velocity=phase.velocity[selected_frame], pressure=phase.pressure[selected_frame],
            methanol=phase.methanol[selected_frame], salt_mol_m3=phase.salt_mol_m3[selected_frame],
            potential=phase.potential[selected_frame], electric=phase.electric[selected_frame])
        field_time = float(phase.time_s[selected_frame])
        if field == "electric_velocity":
            comparison = getattr(scene, "voltage_comparison", None)
            if comparison is None:
                raise ValueError("Compute the 0 kV comparison before displaying voltage-driven velocity")
            if any(not np.array_equal(a, b) for a, b in zip(phase.grid.edges, comparison.current_phase.grid.edges)):
                raise ValueError("Phase comparison grid differs from the displayed phase grid")
            volume.delta_velocity = comparison.delta_phase_velocity
            field_time = float(comparison.diagnostics["matched_phase_time_s"])
    result = flat_slice(volume, field, axis, position_m)
    result.update(source_id=scene.source_id, data_clock=clock, field_time_s=field_time,
                  phase_frame=selected_frame, transport_time_s=None if time_s is None else float(time_s))
    if clock == "phase":
        result["notes"].append("The moving-phase field clock is independent of the steady-feed transport clock; they must not be interpreted as one synchronized transient.")
    if time_s is not None:
        cells = scene.frame(float(time_s))["cells"]
        result["cell_intersections"] = sphere_intersections(cells, axis, float(position_m))
    return result


def sphere_intersections(cells, axis, position_m):
    """Intersect finite spheres carrying *native* local positions with a plane.

    Off-plane cells are not projected into the cut.  Missing local coordinates
    are skipped rather than inverting a bent-pipe display transform incorrectly.
    """
    if axis not in ("x", "y", "z") or not math.isfinite(float(position_m)):
        raise ValueError("Invalid sphere cut plane")
    normal = "xyz".index(axis)
    horizontal, vertical = ((2, 1) if normal == 0 else (2, 0) if normal == 1 else (0, 1))
    result = []
    for cell in cells:
        xyz = cell.get("local_position_m")
        radius = float(cell.get("radius_m", 0.))
        if xyz is None or len(xyz) != 3 or radius <= 0 or not np.isfinite([*xyz, radius]).all():
            continue
        distance = float(xyz[normal])-float(position_m)
        if abs(distance) > radius:
            continue
        result.append({"id": cell["id"], "horizontal_m": float(xyz[horizontal]),
                       "vertical_m": float(xyz[vertical]),
                       "radius_m": math.sqrt(max(0., radius*radius-distance*distance)),
                       "sphere_radius_m": radius, "distance_to_plane_m": distance,
                       "selected": bool(cell.get("selected", False)),
                       "interaction_status": cell.get("interaction_status", "mobile"),
                       "color": cell.get("color", [.3, .9, .8, 1.])})
    return result
