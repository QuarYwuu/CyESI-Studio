"""Finite-radius rigid sphere force closures; SI units.

These are dilute, unresolved-particle closures, not a resolved cell membrane
or immersed-boundary calculation. Saffman is a low-slip, far-wall asymptotic
law. The lubrication correction is damped at two radii from the surface.
"""
import numpy as np

KB = 1.380649e-23


def drag_coefficient(viscosity, radius):
    return 6*np.pi*np.asarray(viscosity)*radius


def drag_force(viscosity, radius, fluid_velocity, cell_velocity):
    return drag_coefficient(viscosity, radius)*(np.asarray(fluid_velocity)-np.asarray(cell_velocity))


def saffman_lift(rho, mu, radius, slip_z, shear):
    return 6.46*np.asarray(rho)*radius**2*np.sqrt(mu/rho*np.abs(shear))*slip_z*np.sign(shear)


def lubrication_factors(radius, surface_gap):
    h = np.maximum(surface_gap, radius*1e-4)
    normal = 1+np.maximum(radius/h-.5, 0.)
    parallel = 1+8/15*np.log(np.maximum(2*radius/h, 1.))
    return normal, parallel


def brownian_diffusivity(temperature, viscosity, radius):
    return KB*temperature/drag_coefficient(viscosity, radius)


def electrophoretic_mobility(permittivity, zeta_v, viscosity):
    return permittivity*zeta_v/np.asarray(viscosity)


def dielectrophoretic_force(permittivity, radius, cm_factor, grad_e_squared):
    return 2*np.pi*permittivity*radius**3*cm_factor*np.asarray(grad_e_squared)


def exponential_step(position, velocity, target, relaxation_time, dt):
    """Exact frozen-force integration; remains stable for dt >> m/(6 pi mu R)."""
    decay = np.exp(-dt/relaxation_time)
    displacement = target*dt+(velocity-target)*relaxation_time*(1-decay)
    return position+displacement, target+(velocity-target)*decay


def sphere_linear_step(vr, vz, ur, uz, extra_r, extra_z, normal_r, normal_z,
                       normal_drag, parallel_drag, lift_ratio, tau, dt):
    """Exact frozen-flow 2x2 drag/lift dynamics, including anisotropic wall drag.

    Saffman force is linear in (u_z-v_z). Integrating it as a frozen external
    force creates an artificial O(dt) slip when dt >> tau. Here it is part of
    the matrix exponential, so the terminal velocity and transient are both
    independent of a one-step velocity lag.
    """
    difference = normal_drag-parallel_drag
    a = parallel_drag+difference*normal_r**2
    c = difference*normal_r*normal_z
    b = c+lift_ratio
    d = parallel_drag+difference*normal_z**2
    # Tangential wall correction multiplies relative (u_t-v_t) drag.
    # Squeeze lubrication resists absolute normal motion towards the wall.
    # Multiplying only v_t would spuriously slow an entrained sphere in a
    # developed straight pipe and then generate an artificial Saffman drift.
    un = ur*normal_r+uz*normal_z
    fluid_r = parallel_drag*ur+(1-parallel_drag)*un*normal_r
    fluid_z = parallel_drag*uz+(1-parallel_drag)*un*normal_z
    rhs_r, rhs_z = fluid_r+extra_r+lift_ratio*uz, fluid_z+extra_z
    determinant = a*d-b*c
    target_r = (d*rhs_r-b*rhs_z)/determinant
    target_z = (a*rhs_z-c*rhs_r)/determinant
    delta_r, delta_z = vr-target_r, vz-target_z
    trace = (a+d)/2
    discriminant = ((a-d)/2)**2+b*c
    h = dt/tau
    delta = np.sqrt(np.maximum(discriminant, 0.))
    eplus, eminus = np.exp(-(trace+delta)*h), np.exp(-(trace-delta)*h)
    c0 = (eplus+eminus)/2
    c1 = np.divide(eplus-eminus, 2*delta, out=-h*np.exp(-trace*h), where=delta>1e-10)
    oscillatory = discriminant < 0
    if np.any(oscillatory):
        omega = np.sqrt(np.maximum(-discriminant, 1e-300))
        decay = np.exp(-trace*h)
        c0 = np.where(oscillatory, decay*np.cos(omega*h), c0)
        c1 = np.where(oscillatory, -decay*np.sin(omega*h)/omega, c1)
    ev_r = c0*delta_r+c1*((a-trace)*delta_r+b*delta_z)
    ev_z = c0*delta_z+c1*(c*delta_r+(d-trace)*delta_z)
    diff_r, diff_z = delta_r-ev_r, delta_z-ev_z
    dr = target_r*dt+tau*(d*diff_r-b*diff_z)/determinant
    dz = target_z*dt+tau*(-c*diff_r+a*diff_z)/determinant
    return dr, dz, target_r+ev_r, target_z+ev_z
