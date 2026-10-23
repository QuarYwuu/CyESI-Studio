"""Pressure units, segmented tube orientation and hydraulic network closure.

Gauge pressure is relative to atmosphere at the spray. Hydrostatic pressure is
included once, over the actual tube path, not in the axisymmetric tip solver.
"""
from __future__ import annotations
import math
import numpy as np

PRESSURE_UNITS = {"mbar": 100., "kPa": 1000., "bar": 1e5,
                  "Pa": 1., "psi": 6894.757293168}
GRAVITY = 9.80665
ORIENTATIONS = {
    "horizontal": [(1., 0.)], "down": [(1., 90.)], "up": [(1., -90.)],
    "horizontal_down": [(.5, 0.), (.5, 90.)],
    "down_horizontal": [(.5, 90.), (.5, 0.)]}


def convert_pressure(value, source="mbar", target="kPa"):
    if not math.isfinite(float(value)):
        raise ValueError("Pressure must be finite")
    return float(value)*PRESSURE_UNITS[source]/PRESSURE_UNITS[target]


def segments(config):
    """Return path segments with local gravity (x,y,z) and end arclength."""
    s = config.supply
    if s.orientation == "legacy":
        # Preserves explicitly supplied historical axial gravity, no transverse gravity.
        raw = [(1., config.cell.gravity_z, 0., 0.)]
    else:
        rows = s.segments if s.orientation == "custom" else [
            {"fraction": f, "angle_deg": a} for f, a in ORIENTATIONS[s.orientation]]
        raw = []
        for row in rows:
            angle, roll = np.deg2rad([row["angle_deg"], row.get("roll_deg", 0.)])
            raw.append((row["fraction"], GRAVITY*np.sin(angle),
                        GRAVITY*np.cos(angle)*np.cos(roll),
                        GRAVITY*np.cos(angle)*np.sin(roll)))
    position = 0.
    result = []
    for fraction, gz, gx, gy in raw:
        if not s.gravity_enabled: gz = gx = gy = 0.
        length = fraction*config.geometry.sample_length_mm*1e-3
        result.append({"start_m": position, "end_m": position+length,
                       "gx": float(gx), "gy": float(gy), "gz": float(gz)})
        position += length
    return result


def hydrostatic_assist_pa(config):
    rho = config.fluid.sample_density
    return rho*(sum((s["end_m"]-s["start_m"])*s["gz"] for s in segments(config))
                + GRAVITY*config.supply.reservoir_head_mm*1e-3)


def resistance(config, length_m=None):
    g = config.geometry
    length = g.sample_length_mm*1e-3 if length_m is None else length_m
    return 8*config.fluid.sample_viscosity*length/(np.pi*g.a**4)


def estimate_flow(config, extra_tip_pressure_pa=0.):
    """Fast developed-flow preview; full run additionally solves tip backpressure."""
    pressure = (config.supply.sample_pressure_mbar*100 + hydrostatic_assist_pa(config)
                - config.supply.outlet_pressure_pa - extra_tip_pressure_pa)
    return pressure/resistance(config)*60e9


def infer_pressure(config, target_ul_min=None, extra_tip_pressure_pa=0.):
    q = config.fluid.sample_flow_ul_min if target_ul_min is None else target_ul_min
    return (resistance(config)*q/60e9 + extra_tip_pressure_pa
            + config.supply.outlet_pressure_pa-hydrostatic_assist_pa(config))/100


def inlet_pressure(flow, config):
    # Area-mean static pressure; Gauss integration stays inside finite elements.
    nodes, weights = np.polynomial.legendre.leggauss(20)
    r = (nodes+1)*config.geometry.a/2
    z = np.full_like(r, config.geometry.liquid_start+1e-12)
    return float(np.sum(weights*flow.pressure(r, z)*r)/np.sum(weights*r))


def pressure_solution(config, solve_fixed_flow, progress):
    """Secant solve R_bulk Q + p_FEM,in(Q) + p_out = p_feed + rho g dz.

    The sheath flow stays prescribed. Every trial recomputes the coupled FEM,
    species and ionic fields, so EOF and sample/sheath backpressure participate.
    A reverse-flow operating point is rejected rather than made positive.
    """
    s = config.supply
    driving = s.sample_pressure_mbar*100+hydrostatic_assist_pa(config)-s.outlet_pressure_pa
    if driving <= 0:
        raise ValueError(" Sample pressure is below the hydrostatic head / Outlet back pressure ,  Stagnant or reversed flow ;  Transport model requires forward flow .")
    bulk_r = resistance(config, config.geometry.bulk_length)
    full_r = resistance(config)
    q = max(driving/full_r*60e9, 1e-6)
    history = []
    previous = None
    for iteration in range(s.pressure_iterations):
        config.fluid.sample_flow_ul_min = float(q)
        progress(f"Pressure-controlled sample {iteration+1}: Q={q:.6g} uL/min; sheath fixed")
        flow, solvent, coupling = solve_fixed_flow(config, progress)
        p_tip = inlet_pressure(flow, config)
        residual = bulk_r*config.fluid.qs + p_tip-driving
        history.append({"sample_flow_ul_min": float(q), "tip_inlet_pressure_pa": p_tip,
                        "pressure_residual_pa": float(residual)})
        if abs(residual) < s.pressure_tolerance_pa:
            coupling["supply"] = {"control": "pressure", "feed_pressure_mbar": s.sample_pressure_mbar,
                "actual_sample_flow_ul_min": float(q), "hydrostatic_assist_pa": hydrostatic_assist_pa(config),
                "external_backpressure_pa": s.outlet_pressure_pa, "iterations": history,
                "pressure_residual_pa": float(residual), "converged": True,
                "backpressure_scope": "prescribed external offset + self-consistent internal FEM; transient spray feedback unresolved"}
            flow.diagnostics["supply"] = coupling["supply"]
            return flow, solvent, coupling
        slope = full_r/60e9
        if previous is not None and abs(q-previous[0]) > 1e-12:
            candidate = (residual-previous[1])/(q-previous[0])
            if candidate > 0: slope = candidate
        next_q = q-residual/slope
        previous = (q, residual)
        if next_q <= 0:
            if q < 1e-5:
                raise ValueError(" Sheath back pressure exceeds sample supply pressure ,  No forward sample-flow solution .")
            next_q = q*.05
        q = next_q
    raise RuntimeError(f"Pressure / flow closure did not converge; last residual {residual:.4g} Pa")
