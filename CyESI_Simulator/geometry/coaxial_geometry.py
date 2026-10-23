"""Axisymmetric domains: middle tube lip is z=0; positive z points to MS."""
from enum import IntEnum
import numpy as np


class Domain(IntEnum):
    SAMPLE = 1
    SHEATH = 2
    NITROGEN = 3
    QUARTZ = 4
    AIR = 5


def classify(r, z, geometry):
    r, z = np.broadcast_arrays(r, z)
    g = geometry
    out = np.full(r.shape, Domain.AIR, dtype=np.int8)
    before_inner = z < g.inner_tip
    before_sheath = z < 0
    before_gas = z < -g.gas_recess_mm * 1e-3
    out[(r < g.c) & before_sheath] = Domain.SHEATH
    out[(r < g.a) & before_inner] = Domain.SAMPLE
    out[(r >= g.d) & (r < g.gas_id_um * 0.5e-6) & before_gas] = Domain.NITROGEN
    out[((r >= g.a) & (r <= g.b) & before_inner)
        | ((r >= g.c) & (r <= g.d) & before_sheath)
        | ((r >= g.gas_id_um * 0.5e-6) & (r <= g.gas_od_um * 0.5e-6) & before_gas)] = Domain.QUARTZ
    return out


def liquid_mask(r, z, g):
    return (r < g.c) & ~((z < g.inner_tip) & (r > g.a) & (r < g.b))


def wall_distance(r, z, g):
    """Distance to nearest liquid wall and unit normal pointing into fluid.

    Inner tube solid is a semi-infinite rectangular cross-section. Distance
    includes its lip corners; finite sphere clearance uses distance minus R.
    """
    r, z = np.broadcast_arrays(r, z)
    dr = r - np.clip(r, g.a, g.b)
    dz = np.maximum(z - g.inner_tip, 0.)
    inner_dist = np.hypot(dr, dz)
    inside = (r >= g.a) & (r <= g.b) & (z < g.inner_tip)
    inner_dist = np.where(inside, -np.minimum(r - g.a, g.b - r), inner_dist)
    denom = np.maximum(np.hypot(dr, dz), 1e-30)
    nr, nz = dr / denom, dz / denom
    outer_dist = g.c - r
    outer_nearest = outer_dist < inner_dist
    return (np.where(outer_nearest, outer_dist, inner_dist),
            np.where(outer_nearest, -1., nr), np.where(outer_nearest, 0., nz))
