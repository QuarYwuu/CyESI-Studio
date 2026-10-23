"""Liquid-inventory transport on the same physical/compression phase fluxes."""
import numpy as np


def transport_inventory(inventory,concentration,fr,fz,inlet,volume,dt):
    """Transport phi*C without moving liquid solute in the gas volume flux."""
    ar=np.zeros_like(fr);az=np.zeros_like(fz)
    ar[1:-1]=fr[1:-1]*np.where(fr[1:-1]>=0,concentration[:-1],concentration[1:])
    az[:,1:-1]=fz[:,1:-1]*np.where(fz[:,1:-1]>=0,concentration[:,:-1],concentration[:,1:])
    ar[-1]=np.maximum(fr[-1],0)*concentration[-1]
    az[:,0]=fz[:,0]*np.where(fz[:,0]>=0,inlet,concentration[:,0])
    az[:,-1]=np.maximum(fz[:,-1],0)*concentration[:,-1]
    new=inventory-dt*(np.diff(ar,axis=0)+np.diff(az,axis=1))/volume
    boundary=float(dt*(ar[-1].sum()+az[:,-1].sum()-az[:,0].sum()))
    return new,boundary


def recover_concentration(inventory,phi):
    return np.divide(inventory,phi,out=np.zeros_like(inventory),where=phi>1e-14)
