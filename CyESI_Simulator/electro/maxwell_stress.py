import numpy as np
from .electrostatics import EPS0


def stress_tensor(electric_field, permittivity=EPS0):
    E = np.asarray(electric_field)
    return permittivity*(np.einsum('i...,j...->ij...', E, E)
                         - .5*np.sum(E*E, axis=0)*np.eye(E.shape[0]).reshape((E.shape[0], E.shape[0])+(1,)*(E.ndim-1)))


def conducting_normal_pressure(field_magnitude):
    return .5*EPS0*np.asarray(field_magnitude)**2
