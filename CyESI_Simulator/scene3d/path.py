"""Capillary centrelines and local gravity frames.

Coordinates and arclength s use SI metres, with s measured from the sample
entrance to the inner-capillary lip. The path supplies geometry and gravity
projection; curved-pipe secondary flow and bend impacts are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicSpline


def _unit(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-30)


def _transport_frames(tangents):
    """Rotation-minimising frames; no Frenet-frame flip at straight sections."""
    tangents = _unit(tangents)
    normal = np.zeros_like(tangents)
    # At an upward inlet, +global Z is a stable transverse initial direction.
    seed = np.array([0., 0., 1.])
    if abs(np.dot(seed, tangents[0])) > .95:
        seed = np.array([0., -1., 0.])
    normal[0] = _unit(seed - np.dot(seed, tangents[0])*tangents[0])
    for index in range(1, len(tangents)):
        before, after = tangents[index-1:index+1]
        axis = np.cross(before, after)
        sine = np.linalg.norm(axis)
        cosine = np.clip(np.dot(before, after), -1., 1.)
        if sine > 1e-12:
            axis /= sine
            candidate = (normal[index-1]*cosine
                         + np.cross(axis, normal[index-1])*sine
                         + axis*np.dot(axis, normal[index-1])*(1-cosine))
        elif cosine < 0:
            raise ValueError("Path reverses by 180 degrees; insert intermediate control points")
        else:
            candidate = normal[index-1]
        normal[index] = _unit(candidate-np.dot(candidate, after)*after)
    # N cross B = T, with B = T cross N.
    return normal, _unit(np.cross(tangents, normal))


@dataclass
class TubePath:
    """Continuous centreline and orthonormal transverse frames.

    The default path rises, bends and becomes horizontal. from_points accepts
    [x, y, z] control points in metres. length_m scales the path about its endpoint.
    """
    station_m: np.ndarray
    centers_m: np.ndarray
    tangents: np.ndarray
    normals: np.ndarray
    binormals: np.ndarray
    metadata: dict

    @property
    def length_m(self):
        return float(self.station_m[-1])

    @classmethod
    def default(cls, config, radius_mm=10., up_fraction=.5):
        length = float(config.geometry.sample_length_mm)*1e-3
        radius = float(radius_mm)*1e-3
        fraction = float(up_fraction)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Bend radius must be finite and positive")
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Upward fraction must lie strictly between 0 and 1")
        arc = math.pi*radius/2
        if arc >= length:
            raise ValueError("A quarter bend must be shorter than the capillary")
        up = (length-arc)*fraction
        horizontal = length-arc-up
        tip = float(config.geometry.inner_tip)
        # Include both analytic joins and resolve the bend independently of the
        # long straight arms. Tail samples retain detail near the resolved tip.
        stations = np.unique(np.r_[np.linspace(0, up, 100),
                                     np.linspace(up, up+arc, 121),
                                     np.linspace(up+arc, length, 100),
                                     np.linspace(max(up+arc, length-.012), length, 100)])
        centers = np.zeros((len(stations), 3))
        theta = np.clip((stations-up)/radius, 0, math.pi/2)
        vertical = stations <= up
        bend = (stations > up) & (stations < up+arc)
        straight = stations >= up+arc
        centers[vertical, 1] = -radius-up+stations[vertical]
        centers[vertical, 2] = tip-horizontal-radius
        centers[bend, 1] = -radius+radius*np.sin(theta[bend])
        centers[bend, 2] = tip-horizontal-radius+radius*(1-np.cos(theta[bend]))
        centers[straight, 2] = tip-horizontal+stations[straight]-up-arc
        tangents = np.c_[np.zeros(len(theta)), np.cos(theta), np.sin(theta)]
        normals = np.c_[np.zeros(len(theta)), -np.sin(theta), np.cos(theta)]
        binormals = np.tile([1., 0., 0.], (len(stations), 1))
        return cls(stations, centers, tangents, normals, binormals, {
            "kind": "up_bend_horizontal", "length_m": length,
            "bend_radius_m": radius, "up_fraction": fraction,
            "up_length_m": up, "bend_length_m": arc,
            "horizontal_length_m": horizontal,
            "control_points_m": [centers[0].tolist(), centers[vertical][-1].tolist(),
                                  centers[straight][0].tolist(), centers[-1].tolist()],
            "shape_source": "editable assumed route; not an instrument measurement",
            "flow_scope": "segmented developed flow; Dean flow and bend impact unresolved",
        })

    @classmethod
    def from_points(cls, points, length_m=None, tip_position=None, samples=1025):
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError("Path needs at least two finite [x_m,y_m,z_m] control points")
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(distances <= 1e-10):
            raise ValueError("Adjacent path control points must be distinct")
        if not isinstance(samples, int) or not 65 <= samples <= 8193:
            raise ValueError("Path samples must be an integer between 65 and 8193")
        parameter = np.r_[0., np.cumsum(distances)]
        first = (points[1]-points[0])/distances[0]
        last = (points[-1]-points[-2])/distances[-1]
        spline = CubicSpline(parameter, points, axis=0,
                             bc_type=((1, first), (1, last)))
        dense_t = np.unique(np.r_[np.linspace(0, parameter[-1], max(8193, samples*8)), parameter])
        dense = spline(dense_t)
        dense_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        measured = float(dense_s[-1])
        wanted = measured if length_m is None else float(length_m)
        if not math.isfinite(wanted) or wanted <= 0:
            raise ValueError("Path length must be finite and positive")
        scale = wanted/measured
        anchor = points[-1] if tip_position is None else np.asarray(tip_position, dtype=float)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("Tip position must be a finite [x_m,y_m,z_m] point")
        stations = np.linspace(0, wanted, samples)
        # Additional stations preserve all user control points and their frames.
        control_s = np.interp(parameter, dense_t, dense_s)*scale
        stations = np.unique(np.r_[stations, control_s])
        t = np.interp(stations/scale, dense_s, dense_t)
        centers = (spline(t)-points[-1])*scale+anchor
        tangents = _unit(spline(t, 1))
        if np.any(np.linalg.norm(spline(t, 1), axis=1) < 1e-8):
            raise ValueError("Path contains a cusp; adjust the control points")
        normal, binormal = _transport_frames(tangents)
        return cls(stations, centers, tangents, normal, binormal, {
            "kind": "editable_control_points", "length_m": wanted,
            "unscaled_length_m": measured, "length_scale": scale,
            "control_points_m": points.tolist(), "tip_position_m": anchor.tolist(),
            "interpolation": "clamped cubic spline, arclength resampling, parallel transport frames",
            "shape_source": "user-editable route; curve interpolation is a geometric assumption",
            "flow_scope": "segmented developed flow; Dean flow and bend impact unresolved",
        })

    def sample(self, station):
        """Return (centre, tangent, normal, binormal), broadcasting station shape.

        Outside the capillary, extend the terminal straight frame. This gives a
        continuous coordinate map for tip, emitted fluid and MS inlet geometry.
        """
        station = np.asarray(station, dtype=float)
        if not np.isfinite(station).all():
            raise ValueError("Station must be finite")
        flat = station.reshape(-1)
        clipped = np.clip(flat, 0, self.length_m)
        def interpolate(values):
            return np.stack([np.interp(clipped, self.station_m, values[:, j]) for j in range(3)], axis=-1)
        if self.metadata.get("kind") == "up_bend_horizontal":
            # Analytic evaluation keeps the quarter-circle tangent continuous;
            # display/serialization sampling does not become a polyline model.
            up = self.metadata["up_length_m"]
            arc = self.metadata["bend_length_m"]
            horizontal = self.metadata["horizontal_length_m"]
            radius = self.metadata["bend_radius_m"]
            tip = self.centers_m[-1, 2]
            theta = np.clip((clipped-up)/radius, 0, math.pi/2)
            vertical = clipped <= up
            bend = (clipped > up) & (clipped < up+arc)
            straight = clipped >= up+arc
            center = np.zeros((len(flat), 3))
            center[vertical, 1] = -radius-up+clipped[vertical]
            center[vertical, 2] = tip-horizontal-radius
            center[bend, 1] = -radius+radius*np.sin(theta[bend])
            center[bend, 2] = tip-horizontal-radius+radius*(1-np.cos(theta[bend]))
            center[straight, 2] = tip-horizontal+clipped[straight]-up-arc
            tangent = np.c_[np.zeros(len(flat)), np.cos(theta), np.sin(theta)]
            normal = np.c_[np.zeros(len(flat)), -np.sin(theta), np.cos(theta)]
            binormal = np.tile([1., 0., 0.], (len(flat), 1))
        else:
            center = interpolate(self.centers_m)
            tangent = _unit(interpolate(self.tangents))
            normal = interpolate(self.normals)
            normal = _unit(normal-np.sum(normal*tangent, axis=1, keepdims=True)*tangent)
            binormal = _unit(np.cross(tangent, normal))
        low = flat < 0
        high = flat > self.length_m
        center[low] += flat[low, None]*self.tangents[0]
        center[high] += (flat[high, None]-self.length_m)*self.tangents[-1]
        shape = station.shape+(3,)
        return tuple(value.reshape(shape) for value in (center, tangent, normal, binormal))

    def map(self, station, x=0., y=0.):
        """Map local transverse (x,y) and arclength to physical global xyz."""
        station, x, y = np.broadcast_arrays(station, x, y)
        center, _, normal, binormal = self.sample(station)
        return center+np.asarray(x)[..., None]*normal+np.asarray(y)[..., None]*binormal

    def segments_for_supply(self, n=64):
        """Approximate local gravity for the existing segmented transport model.

        Segment axial gravity uses the exact endpoint height drop, preserving
        hydrostatic pressure. Transverse gravity uses the midpoint frame. The
        curved-pipe secondary flow is not introduced by this projection.
        """
        if not isinstance(n, int) or not 2 <= n <= 4096:
            raise ValueError("Segment count must be an integer between 2 and 4096")
        station = np.linspace(0, self.length_m, n+1)
        # Exact analytic transition stations improve default bend resolution.
        if self.metadata["kind"] == "up_bend_horizontal":
            a = self.metadata["up_length_m"]
            b = a+self.metadata["bend_length_m"]
            station = np.unique(np.r_[station, np.linspace(a, b, max(17, n//2))])
        centers, *_ = self.sample(station)
        widths = np.diff(station)
        _, _, normal, binormal = self.sample((station[:-1]+station[1:])/2)
        sine = np.clip(-np.diff(centers[:, 1])/widths, -1., 1.)
        angles = np.rad2deg(np.arcsin(sine))
        roll = np.rad2deg(np.arctan2(-binormal[:, 1], -normal[:, 1]))
        result = [{"fraction": float(width/self.length_m),
                   "angle_deg": float(angle), "roll_deg": float(rotation)}
                  for width, angle, rotation in zip(widths, angles, roll)]
        result[-1]["fraction"] += 1-sum(row["fraction"] for row in result)
        return result

    def sweep(self, radius_m, segments=48, stations=None, **attributes):
        """Swept circular surface with flat SI arrays for the WebGL renderer.

        station_m records each vertex's path station for an optional continuous
        display lens. It does not change the physical coordinates or geometry.
        """
        radius = float(radius_m)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Tube radius must be finite and positive")
        if not isinstance(segments, int) or not 8 <= segments <= 128:
            raise ValueError("Radial segments must be an integer between 8 and 128")
        stations = self.station_m if stations is None else np.asarray(stations, dtype=float)
        if stations.ndim != 1 or len(stations) < 2 or np.any(np.diff(stations) <= 0):
            raise ValueError("Sweep stations must increase strictly")
        angle = np.linspace(0, 2*math.pi, segments+1)
        positions = self.map(stations[:, None], radius*np.cos(angle)[None, :], radius*np.sin(angle)[None, :])
        j, i = np.meshgrid(np.arange(len(stations)-1), np.arange(segments), indexing="ij")
        a = j*(segments+1)+i
        b = a+segments+1
        indices = np.stack((a, b, a+1, a+1, b, b+1), axis=-1).reshape(-1)
        return {"positions": positions.reshape(-1).tolist(), "indices": indices.tolist(),
                "station_m": np.repeat(stations, segments+1).tolist(), **attributes}

    def to_dict(self):
        return {"station_m": self.station_m.tolist(), "centers_m": self.centers_m.tolist(),
                "tangents": self.tangents.tolist(), "normals": self.normals.tolist(),
                "binormals": self.binormals.tolist(), "metadata": self.metadata.copy()}
