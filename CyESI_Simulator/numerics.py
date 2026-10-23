"""Conservative axisymmetric finite-volume operators on orthogonal grids."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def metric(re, ze):
    rc, zc = (re[1:]+re[:-1])/2, (ze[1:]+ze[:-1])/2
    axial_area = np.pi*np.diff(re**2)
    radial_area = 2*np.pi*re[:, None]*np.diff(ze)[None, :]
    volume = axial_area[:, None]*np.diff(ze)[None, :]
    return rc, zc, axial_area, radial_area, volume


def elliptic(re, ze, coefficient, active=None, boundary=None):
    """Integrated -div(k grad), zero flux at omitted boundaries/inactive cells.

    boundary maps r0/r1/z0/z1 to Dirichlet values at boundary faces.
    Returns matrix and boundary RHS, before internal Dirichlet elimination.
    """
    nr, nz = len(re)-1, len(ze)-1
    ids = np.arange(nr*nz).reshape(nr, nz)
    rc, zc, aa, ra, _ = metric(re, ze)
    k = np.broadcast_to(coefficient, (nr, nz))
    active = np.ones((nr, nz), bool) if active is None else active
    rows, cols, vals = [], [], []
    diagonal = np.zeros((nr, nz))
    rhs = np.zeros((nr, nz))
    for axis in [0, 1]:
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        conductance = 2*k[left]*k[right]/np.maximum(k[left]+k[right], 1e-300)
        conductance *= (ra[1:-1]/np.diff(rc)[:, None] if axis == 0 else aa[:, None]/np.diff(zc)[None, :])
        conductance *= active[left] & active[right]
        diagonal[left] += conductance
        diagonal[right] += conductance
        rows.extend([ids[left].ravel(), ids[right].ravel()])
        cols.extend([ids[right].ravel(), ids[left].ravel()])
        vals.extend([-conductance.ravel(), -conductance.ravel()])
    for side, values in (boundary or {}).items():
        if side == "r0": sl, area, dist = (0, slice(None)), ra[0], rc[0]-re[0]
        elif side == "r1": sl, area, dist = (-1, slice(None)), ra[-1], re[-1]-rc[-1]
        elif side == "z0": sl, area, dist = (slice(None), 0), aa, zc[0]-ze[0]
        elif side == "z1": sl, area, dist = (slice(None), -1), aa, ze[-1]-zc[-1]
        else: raise ValueError(side)
        conductance = k[sl]*area/dist*active[sl]
        diagonal[sl] += conductance
        rhs[sl] += conductance*values
    diagonal[~active] = 1.
    rows.append(ids.ravel()); cols.append(ids.ravel()); vals.append(diagonal.ravel())
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(nr*nz, nr*nz)).tocsr()
    return A, rhs.ravel()


def dirichlet_solve(A, rhs, fixed, values):
    fixed = np.asarray(fixed).ravel()
    values = np.broadcast_to(values, fixed.shape).copy()
    free = ~fixed
    out = values.copy()
    b = rhs[free] - A[free][:, fixed]@values[fixed]
    out[free] = spsolve(A[free][:, free], b)
    residual = np.linalg.norm((A@out-rhs)[free])/max(np.linalg.norm(rhs[free]), np.linalg.norm(b), 1e-30)
    return out, float(residual)
