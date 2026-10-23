"""Native Cartesian-volume geometry for the separate offline 3-D viewer.

This adapter draws finite-volume values at their actual three-dimensional
coordinates. It never revolves a 2-D field, invents a Taylor cone, or extrapolates
an unresolved field into the upstream tubing. Slice patches are cell-constant
values, matching the solver's finite-volume storage, with solids masked.
"""
from __future__ import annotations

import math
import numpy as np


FIELD_INFO = {
    "methanol": (" Methanol volume fraction ", "fraction", "liquid"),
    "speed": (" Liquid velocity ", "mm/s", "liquid"),
    "pressure": (" Pressure ", "Pa", "fluid"),
    "potential": (" Potential ", "V", "all"),
    "electric": (" Electric field magnitude ", "V/m", "all"),
    "salt": (" Salt concentration ", "mM", "liquid"),
    "gas_speed": (" Gas velocity ", "m/s", "gas"),
    "electric_velocity": (" Voltage-induced velocity change  |Δu|", "mm/s", "liquid"),
}


def _map_local(points, path, config):
    points = np.asarray(points, dtype=float)
    if path is None:
        return points
    station = path.length_m+points[..., 2]-config.geometry.inner_tip
    return path.map(station, points[..., 0], points[..., 1])


def _palette(fraction, name="sequential"):
    fraction = np.asarray(fraction, float)
    # Matplotlib-free, transparent and fixed scale across every slice. These
    # interpolated colour knots carry no extra field or concentration content.
    if name == "methanol":
        knots = np.array([[.05, .38, .82], [.1, .73, .72], [.94, .68, .12], [.91, .28, .12]])
    elif name == "diverging":
        knots = np.array([[.16, .35, .75], [.9, .94, .98], [.83, .2, .19]])
    else:
        knots = np.array([[.12, .16, .38], [.08, .45, .68], [.12, .72, .61], [.97, .86, .27]])
    stops = np.linspace(0, 1, len(knots))
    return np.stack([np.interp(np.clip(fraction, 0, 1), stops, knots[:, j]) for j in range(3)], axis=-1)


def _field(volume, name):
    if name not in FIELD_INFO:
        raise ValueError(f"Unknown native 3-D field: {name}")
    if name == "electric_velocity":
        if not hasattr(volume, "delta_velocity"):
            raise ValueError(' Compute  0 kV  Compare ;  Velocity difference requires two voltage solutions ')
        values = np.linalg.norm(volume.delta_velocity, axis=-1)*1000
    elif name == "speed":
        values = np.linalg.norm(volume.velocity, axis=-1)*1000
    elif name == "gas_speed":
        values = np.linalg.norm(volume.velocity, axis=-1)
    elif name == "electric":
        values = np.linalg.norm(volume.electric, axis=-1)
    elif name == "salt":
        # 1 mol/m³ equals 1 mmol/L.
        values = volume.salt_mol_m3
    else:
        values = getattr(volume, name)
    category = FIELD_INFO[name][2]
    active = {"liquid": volume.material == 1, "gas": volume.material == 0,
              "fluid": volume.material != 2, "all": np.ones(volume.material.shape, bool)}[category]
    active &= np.isfinite(values)
    return np.asarray(values), active


def _scale(values, active, name):
    valid = values[active]
    low, high = (float(np.min(valid)), float(np.max(valid))) if len(valid) else (0., 1.)
    if name == "methanol":
        low, high = 0., 1.
    elif name in ("speed", "gas_speed", "electric", "salt", "electric_velocity"):
        low = 0.
    elif name == "pressure" and low < 0 < high:
        low, high = -max(abs(low), abs(high)), max(abs(low), abs(high))
    if high <= low:
        high = low+max(abs(low)*1e-6, 1.)
    return {"field": name, "label": FIELD_INFO[name][0], "unit": FIELD_INFO[name][1],
            "min": low, "max": high, "palette": "methanol" if name == "methanol" else
            "diverging" if name == "pressure" and low < 0 else "sequential",
            "data_min": float(np.min(valid)) if len(valid) else None,
            "data_max": float(np.max(valid)) if len(valid) else None,
            "storage": "3-D finite-volume cell-centred values; cell-constant cut patches"}


def scalar_slice(volume, config, field="methanol", axis="x", position_m=0., path=None, scale=None):
    """Return one true Cartesian cut plane, clipping out inactive material.

    The requested plane lies at position_m. Its finite-volume values are from
    the containing cell slab; actual slab bounds are returned as metadata.
    No interpolation across quartz, detached phases, or masked holes is used.
    """
    if axis not in ("x", "y", "z"):
        raise ValueError("Slice axis must be x, y or z")
    position = float(position_m)
    k = "xyz".index(axis)
    edges = volume.grid.edges[k]
    if not math.isfinite(position) or position < edges[0] or position > edges[-1]:
        raise ValueError("Slice lies outside the native 3-D volume")
    values, active = _field(volume, field)
    scale = _scale(values, active, field) if scale is None else scale
    index = min(max(int(np.searchsorted(edges, position, side="right")-1), 0), len(edges)-2)
    remaining = [j for j in range(3) if j != k]
    plane_values = np.take(values, index, axis=k)
    plane_active = np.take(active, index, axis=k)
    first, second = np.nonzero(plane_active)
    if not len(first):
        return None, {"axis": axis, "position_m": position, "active_cells": 0}
    # Each active cell contributes an independent quad, so a narrow quartz wall
    # cannot be visually bridged by an interpolated triangle from either side.
    a, b = remaining
    aa = volume.grid.edges[a]
    bb = volume.grid.edges[b]
    p = np.zeros((len(first), 4, 3))
    p[..., k] = position
    p[:, :, a] = np.c_[aa[first], aa[first+1], aa[first+1], aa[first]]
    p[:, :, b] = np.c_[bb[second], bb[second], bb[second+1], bb[second+1]]
    selected = plane_values[first, second]
    c = _palette((selected-scale["min"])/(scale["max"]-scale["min"]), scale["palette"])
    colors = np.repeat(np.c_[c, np.ones(len(c))], 4, axis=0)
    indices = (np.arange(len(first))[:, None]*4+np.array([0, 1, 2, 0, 2, 3])[None, :]).reshape(-1)
    mesh = {"id": f"native-{field}-{axis}-{position:.8g}", "layer": "fields",
            "positions": _map_local(p.reshape(-1, 3), path, config).reshape(-1).tolist(),
            "indices": indices.tolist(), "colors": colors.reshape(-1).tolist(),
            "opacity": .88, "doubleSided": True, "unlit": True}
    return mesh, {"axis": axis, "position_m": position, "active_cells": len(first),
                  "cell_slab_m": [float(edges[index]), float(edges[index+1])],
                  "native_plane_index": index, "coordinates": "Cartesian cell-face quads, no revolution"}


def _arrow(start, vector, length):
    magnitude = np.linalg.norm(vector)
    if not math.isfinite(magnitude) or magnitude <= 1e-30 or length <= 0:
        return []
    direction = vector/magnitude
    end = start+length*direction
    reference = np.array([1., 0., 0.]) if abs(direction[2]) > .8 else np.array([0., 0., 1.])
    side = np.cross(direction, reference)
    side /= max(np.linalg.norm(side), 1e-30)
    base = end-length*.25*direction
    return [start, end, end, base+side*length*.1, end, base-side*length*.1]


def vector_arrows(volume, config, vector="gas", path=None, maximum=96):
    """Native sampled vectors with an explicit linear visual arrow scale."""
    if vector not in ("gas", "liquid", "electric", "electric_velocity"):
        raise ValueError("Vector must be gas, liquid or electric")
    if vector == "electric_velocity" and not hasattr(volume, "delta_velocity"):
        raise ValueError('Voltage comparison has not been computed')
    data = volume.delta_velocity if vector == "electric_velocity" else volume.electric if vector == "electric" else volume.velocity
    active = (volume.material == (0 if vector == "gas" else 1)) if vector != "electric" else (volume.material != 2)
    magnitude = np.linalg.norm(data, axis=-1)
    active &= np.isfinite(magnitude) & (magnitude > 1e-20)
    eligible = np.argwhere(active)
    if not len(eligible):
        return [], {"vector": vector, "count": 0, "length_m_per_unit": 0.}
    maximum = int(np.clip(maximum, 1, 512))
    if len(eligible) > maximum:
        eligible = eligible[np.unique(np.linspace(0, len(eligible)-1, maximum).astype(int))]
    # Use the global finite-data maximum, identical across camera presets.
    max_value = float(np.max(magnitude[active]))
    scale = 100e-6/max_value
    positions = []
    for ix, iy, iz in eligible:
        start = np.array([volume.grid.centers[0][ix], volume.grid.centers[1][iy], volume.grid.centers[2][iz]])
        positions.extend(_arrow(start, data[ix, iy, iz], float(magnitude[ix, iy, iz])*scale))
    if not positions:
        return [], {"vector": vector, "count": 0, "length_m_per_unit": scale}
    colors = {"gas": [.35, .83, 1., 1.], "liquid": [.2, .9, .74, 1.], "electric": [1., .77, .26, 1.],
              "electric_velocity": [1., .42, .69, 1.]}
    line = {"id": f"native-{vector}-vectors", "layer": "electric" if vector == "electric" else "gas" if vector == "gas" else "flow",
            "positions": _map_local(positions, path, config).reshape(-1).tolist(), "color": colors[vector],
            "mode": "segments", "width": 1.2, "opacity": .9}
    return [line], {"vector": vector, "count": len(positions)//6,
                    "length_m_per_unit": scale, "maximum_native_magnitude": max_value,
                    "length_rule": "linear magnitude, global maximum = 100 micrometres before the display lens"}


def _annular_surface(inner, outer, z0, z1, config, path, name, color, count=64):
    """Context geometry from specified physical quartz dimensions, not fields."""
    z = np.linspace(z0, z1, count)
    theta = np.linspace(0, 2*math.pi, 49)
    profile = [(inner, v) for v in z]+[(outer, z1)]+[(outer, v) for v in z[::-1]]+[(inner, z0)]
    positions = []
    for radius, station_z in profile:
        positions.extend(np.c_[radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), station_z)])
    indices = []
    for j in range(len(profile)-1):
        for i in range(len(theta)-1):
            a = j*len(theta)+i
            b = a+len(theta)
            indices.extend([a, b, a+1, a+1, b, b+1])
    return {"id": name, "layer": "quartz", "positions": _map_local(positions, path, config).reshape(-1).tolist(),
            "indices": indices, "color": [*color, 1.], "opacity": .13, "doubleSided": True}


def context_geometry(config, path=None):
    g = config.geometry
    meshes, lines = [], []
    if path is not None:
        meshes.append(path.sweep(g.b, id="native-inner-quartz-outer", layer="quartz",
                                 color=[.66, .83, .94, 1.], opacity=.13))
        meshes.append(path.sweep(g.a, id="native-inner-quartz-bore", layer="quartz",
                                 color=[.77, .9, .96, 1.], opacity=.1))
    else:
        meshes.append(_annular_surface(g.a, g.b, g.inner_tip-g.sample_length_mm*.001,
                                        g.inner_tip, config, path, "native-inner-quartz", [.66, .83, .94], 160))
    upstream = min(g.liquid_start, -config.electrical.contact_distance_mm*.001)
    meshes.append(_annular_surface(g.c, g.d, upstream, 0., config, path, "native-middle-quartz", [.66, .83, .94]))
    meshes.append(_annular_surface(g.gas_id_um*.5e-6, g.gas_od_um*.5e-6,
                                   -g.gas_supply_length_mm*.001, -g.gas_recess_mm*.001,
                                   config, path, "native-gas-quartz", [.62, .72, .87]))
    # Aperture marker only: the real heated inlet CFD is not solved here.
    theta = np.linspace(0, 2*math.pi, 96)
    aperture = np.c_[.0003*np.cos(theta), .0003*np.sin(theta), np.full(len(theta), g.ms_gap_mm*.001)]
    lines.append({"id": "native-ms-aperture", "layer": "inlet", "positions": _map_local(aperture, path, config).reshape(-1).tolist(),
                  "color": [.74, .83, .93, 1.], "width": 2})
    return meshes, lines


def volume_geometry(volume, config, path=None, field="methanol", slice_axis="x",
                    slice_position_m=0., include_context=True, include_vectors=True):
    """Build renderer-native geometry for one independently solved 3-D volume.

    slice_axis='both' draws x=position and y=position; 'none' hides scalar cuts.
    Returned bounds and scales are physical SI metadata, not camera settings.
    """
    if slice_axis not in ("x", "y", "z", "both", "none"):
        raise ValueError("Slice axis must be x, y, z, both or none")
    values, active = _field(volume, field)
    scale = _scale(values, active, field)
    meshes, lines = context_geometry(config, path) if include_context else ([], [])
    slices = []
    for axis in (["x", "y"] if slice_axis == "both" else [] if slice_axis == "none" else [slice_axis]):
        mesh, meta = scalar_slice(volume, config, field, axis, slice_position_m, path, scale)
        if mesh is not None:
            meshes.append(mesh)
        slices.append(meta)
    vectors = []
    if include_vectors:
        for name in (["gas", "liquid", "electric", "electric_velocity"] if field == "electric_velocity" else ["gas", "liquid", "electric"]):
            new_lines, meta = vector_arrows(volume, config, name, path)
            lines.extend(new_lines)
            vectors.append(meta)
    corners = np.array(np.meshgrid(*[[e[0], e[-1]] for e in volume.grid.edges], indexing="ij")).reshape(3, -1).T
    corners = _map_local(corners, path, config)
    if path is not None:
        corners = np.r_[corners, path.centers_m]
    return {"meshes": meshes, "lines": lines, "scales": [scale],
            "metadata": {"mode": "native_cartesian_volume", "dimensions": 3,
                         "grid_shape": list(volume.grid.shape), "field": field,
                         "available_fields": [{"name": name, "label": info[0], "unit": info[1]} for name, info in FIELD_INFO.items()],
                         "slices": slices, "vectors": vectors,
                         "bounds_m": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
                         "quartz": "specified physical dimensions; CFD uses its explicitly reported staircase mesh",
                         "free_surface": "unresolved in this fixed-domain native volume; no fabricated cone or droplets",
                         "upstream": "geometry only outside resolved volume; dynamics require the separately coupled long-tube transport",
                         "diagnostics": volume.diagnostics}}
