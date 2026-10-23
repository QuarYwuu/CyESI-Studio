"""Identify detached, mesh-resolved liquid components; no assigned droplet size."""
import numpy as np
from scipy.ndimage import label


def identify(phi, volumes, radial_centers, axial_centers):
    labels, count = label(phi >= .5)
    inlet_labels = set(labels[:, 0])
    result = []
    for i in range(1, count+1):
        if i in inlet_labels: continue
        mask = labels == i
        volume = float(np.sum(volumes[mask]*phi[mask]))
        weight = np.where(mask, volumes*phi, 0)
        result.append({"volume_m3": volume, "equivalent_radius_um": (3*volume/(4*np.pi))**(1/3)*1e6,
            "centroid_z_um": float(np.sum(weight*axial_centers[None, :])/max(volume, 1e-30))*1e6,
            "cells": int(mask.sum()), "touches_outer_boundary": bool(mask[-1].any() or mask[:, -1].any())})
    return result
