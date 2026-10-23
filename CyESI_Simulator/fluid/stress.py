"""Axisymmetric Newtonian strain invariants, including hoop and extension."""
import numpy as np


def axisymmetric_stress(r, z, ur, uz, viscosity):
    dur_dr, dur_dz = np.gradient(ur, r, z, edge_order=2)
    duz_dr, duz_dz = np.gradient(uz, r, z, edge_order=2)
    hoop = np.divide(ur, np.asarray(r)[:, None], out=dur_dr.copy(),
                     where=np.asarray(r)[:, None] > 1e-20)
    drz = .5*(dur_dz+duz_dr)
    invariant = np.sqrt(2*(dur_dr**2+hoop**2+duz_dz**2+2*drz**2))
    # tau_eq = sqrt((2 mu D):(2 mu D)/2) = mu sqrt(2 D:D).
    return {"equivalent_stress_pa": viscosity*invariant,
            "shear_stress_rz_pa": 2*viscosity*drz,
            "axial_extension_stress_pa": 2*viscosity*duz_dz,
            "strain_rate_s": invariant,
            "divergence_s": dur_dr+hoop+duz_dz}
