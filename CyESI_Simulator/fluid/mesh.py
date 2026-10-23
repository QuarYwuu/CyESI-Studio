from dataclasses import dataclass
import numpy as np
from skfem import MeshTri


@dataclass
class LiquidMesh:
    mesh: MeshTri
    radial_edges: np.ndarray
    axial_edges: np.ndarray
    scale: float


def liquid_mesh(config):
    g, n = config.geometry, config.numerics
    # Include all solid/fluid radial interfaces exactly.
    widths = np.diff([0., g.a, g.b, g.c])
    counts = np.maximum(3, np.rint(n.radial_cells * widths / g.c).astype(int))
    re = np.unique(np.concatenate([np.linspace(a, b, k + 1) for a, b, k in
                                  zip([0, g.a, g.b], [g.a, g.b, g.c], counts)]))
    # Geometrically finer spacing near the recessed inner tip.
    length = g.inner_tip - g.liquid_start
    t = np.linspace(0, 1, n.upstream_cells + 1)
    upstream = g.inner_tip - length * (1 - t) ** 1.6
    downstream = np.linspace(g.inner_tip, 0., n.mixing_cells + 1)
    ze = np.unique(np.r_[upstream, downstream])
    scale = g.c
    mesh = MeshTri.init_tensor(re / scale, ze / scale)
    mid = mesh.p[:, mesh.t].mean(axis=1) * scale
    solid = (mid[1] < g.inner_tip) & (mid[0] > g.a) & (mid[0] < g.b)
    mesh = mesh.remove_elements(np.flatnonzero(solid))
    return LiquidMesh(mesh, re, ze, scale)
