"""Steady methanol advection-diffusion: cell-centred, bounded upwind FV.

The sample is methanol-free 140 mM ammonium formate by default, not PBS.
This is extracellular exposure; no calibrated membrane rupture kinetics.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from ..geometry.coaxial_geometry import liquid_mask
from ..numerics import metric, elliptic


def filled(values, active):
    indices = distance_transform_edt(~active, return_distances=False, return_indices=True)
    return values[tuple(indices)]


@dataclass
class SolventResult:
    radial_edges: np.ndarray
    axial_edges: np.ndarray
    concentration: np.ndarray
    active: np.ndarray
    velocity_r: np.ndarray
    velocity_z: np.ndarray
    pressure: np.ndarray
    radial_flux: np.ndarray
    axial_flux: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        self.r, self.z, *_ = metric(self.radial_edges, self.axial_edges)
        self._interpolators = {}

    def sample(self, r, z, field="concentration"):
        if field not in self._interpolators:
            self._interpolators[field] = RegularGridInterpolator((self.r, self.z),
                filled(getattr(self, field), self.active), bounds_error=False, fill_value=None)
        r, z = np.broadcast_arrays(r, z)
        # Constant extension over the half cells next to symmetry/inlet/outlet.
        pts = np.c_[np.clip(r.ravel(), self.r[0], self.r[-1]), np.clip(z.ravel(), self.z[0], self.z[-1])]
        return self._interpolators[field](pts).reshape(r.shape)


def _evaluate(flow, r, z, active, pressure=False):
    r, z = np.broadcast_arrays(r, z)
    shape = r.shape
    out = np.zeros(shape if pressure else (2,)+shape)
    good = np.flatnonzero(active)
    for indices in np.array_split(good, max(1, int(np.ceil(len(good)/1500)))):
        if not len(indices): continue
        result = flow.pressure(r.ravel()[indices], z.ravel()[indices]) if pressure else flow.velocity(r.ravel()[indices], z.ravel()[indices])
        if pressure: out.ravel()[indices] = result
        else: out.reshape(2, -1)[:, indices] = result
    return out


def solve(config, flow):
    g = config.geometry
    re, ze = flow.mesh_data.radial_edges, flow.mesh_data.axial_edges
    r, z, aa, ra, volumes = metric(re, ze)
    R, Z = np.meshgrid(r, z, indexing="ij")
    active = liquid_mask(R, Z, g)
    vel = _evaluate(flow, R, Z, active)
    pressure = _evaluate(flow, R, Z, active, pressure=True)
    # Axial face integrals use two-point Gaussian quadrature (exact for r*u_z P2).
    qz = np.zeros((len(r), len(ze)))
    valid_z = np.zeros_like(qz, dtype=bool)
    valid_z[:, 1:-1] = active[:, :-1] & active[:, 1:]
    valid_z[:, 0] = active[:, 0]; valid_z[:, -1] = active[:, -1]
    for xi in [-1/np.sqrt(3), 1/np.sqrt(3)]:
        Rf, Zf = np.meshgrid(r+xi*np.diff(re)/2, ze, indexing="ij")
        qz += _evaluate(flow, Rf, Zf, valid_z)[1]*np.pi*Rf*np.diff(re)[:, None]
    # Reconstruct a conservative FV flux from section-integrated FEM flux.
    # This small projection preserves the exact supplied sample/sheath flow rates.
    correction = []
    for j, zj in enumerate(ze):
        groups = [(r < g.a, config.fluid.qs), (r > g.b, config.fluid.qh)] if zj <= g.inner_tip+1e-12 else [(np.ones(len(r), bool), config.fluid.qs+config.fluid.qh)]
        for mask, target in groups:
            mask = mask & valid_z[:, j]
            measured = qz[mask, j].sum()
            if measured <= 0: raise RuntimeError("Non-positive section flux")
            correction.append(abs(measured/target-1))
            qz[mask, j] *= target/measured
    qr = np.zeros((len(re), len(z)))
    qr[1:] = -np.cumsum(np.diff(qz, axis=1), axis=0)
    # Algebraic cancellation may leave roundoff at solid walls.
    qr[1:-1] *= active[:-1] & active[1:]
    qr[-1] = 0.
    D = config.chemistry.diffusivity_m2_s
    cm = config.chemistry.sheath_methanol_fraction
    cin = np.where(r < g.a, 0., cm)
    C, balance = scalar_transport(re, ze, active, qr, qz, D, cin)
    div = np.diff(qr, axis=0)+np.diff(qz, axis=1)
    diagnostics = {"method": "steady axisymmetric conservative upwind advection-diffusion FV",
        "methanol_flux_relative_error": float(balance),
        "max_fv_cell_mass_imbalance_m3_s": float(np.max(np.abs(div[active]))),
        "max_fem_flux_projection_relative_correction": float(max(correction)),
        "max_axial_cell_peclet": float(np.max(np.abs(vel[1])*np.diff(ze)[None, :]/D)),
        "diffusivity_m2_s": D, "mixing_model": "neutral methanol; electromigration charge number zero",
        "viscosity_model": "log interpolation between pure inlet viscosities; no excess viscosity"}
    return SolventResult(re, ze, C, active, vel[0], vel[1], pressure, qr, qz, diagnostics)


def scalar_transport(re, ze, active, qr, qz, D, cin):
    r, z, aa, _, _ = metric(re, ze)
    A, rhs = elliptic(re, ze, D, active, {"z0": cin})
    nr, nz = active.shape
    ids = np.arange(nr*nz).reshape(nr, nz)
    rows, cols, vals = [], [], []
    # Add conservative upwind advection across each internal face.
    for axis, flux in [(0, qr[1:-1]), (1, qz[:, 1:-1])]:
        lo = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        hi = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        low, high = ids[lo].ravel(), ids[hi].ravel()
        F = flux.ravel()
        source = np.where(F >= 0, low, high)
        rows.extend([low, high]); cols.extend([source, source]); vals.extend([F, -F])
    rows.append(ids[:, -1]); cols.append(ids[:, -1]); vals.append(qz[:, -1])
    adv = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=A.shape).tocsr()
    A = A+adv
    rhs[ids[:, 0]] += qz[:, 0]*cin
    C = spsolve(A, rhs).reshape(nr, nz)
    C[~active] = 0
    if C[active].min() < -1e-7 or C[active].max() > np.max(cin)+1e-7:
        raise RuntimeError("Solvent concentration violates physical bounds")
    total_in = float(np.sum(qz[:, 0]*cin))
    diff_in = D*aa/(z[0]-ze[0])*(cin-C[:, 0])*active[:, 0]
    total_out = float(np.sum(qz[:, -1]*C[:, -1]))
    balance = abs(total_out-total_in-diff_in.sum())/max(total_in, 1e-30)
    return C, balance
