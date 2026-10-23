"""Numerical balances, convergence checks and model-scope summaries.

Checks use recorded solver diagnostics. Missing diagnostics remain unavailable.
Screening tolerances are numerical criteria, not experimental error bounds.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import binomtest


REFERENCES = [
    {"title": "NASA CFD Verification and Validation",
     "url": "https://www.grc.nasa.gov/www/wind/valid/tutorial/overview.html"},
    {"title": "SciPy exact binomial confidence interval",
     "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html"},
]


def _number(value):
    """Discard malformed/nonfinite archive values instead of emitting NaN JSON."""
    if value is None or isinstance(value, (bool, np.bool_, str)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _finite(values):
    try:
        return bool(np.isfinite(values).all())
    except (TypeError, ValueError):
        return False


def _ratio(value, denominator):
    if value is None or denominator in (None, 0.):
        return None
    numerator = _number(value)
    return math.nan if numerator is None else numerator / denominator


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    return value


def assess_model(data):
    """Return numerical checks as JSON-compatible values.

    Statuses distinguish passed checks, warnings, failures, missing diagnostics
    and model assumptions. Exact 95% intervals cover finite independent response
    sampling, excluding input uncertainty and model discrepancy.
    """
    c = data.config
    f = data.fields
    summary = data.summary or {}
    flow = summary.get("flow") or {}
    solvent = summary.get("solvent") or {}
    electric = summary.get("electric") or {}
    particles = summary.get("particles") or {}
    coupling = summary.get("coupling") or {}
    events = summary.get("events") or {}
    spray = summary.get("spray") or {}
    checks = []

    def add(key, label, stage, status, value=None, unit="", detail="", **extra):
        checks.append(dict(key=key, label=label, stage=stage, status=status,
                           value=value, unit=unit, detail=detail, **extra))

    def residual(key, label, stage, value, tolerance, detail, signed=False):
        number = _number(value)
        if value is None:
            status = "missing"
        elif number is None or (not signed and number < 0):
            status = "failed"
            detail += "  Recorded error must be finite " + ("." if signed else " And non-negative .")
        else:
            status = "pass" if abs(number) <= tolerance else "warning"
        add(key, label, stage, status, number, detail=detail,
            screening_tolerance=tolerance)

    # Raw fields must be finite only within their own physical domains. Trajectory
    # histories may contain documented padding; these are checked separately.
    active = np.asarray(f.get("liquid_active", []), bool)
    failures = []
    required = ("velocity_r_m_s", "velocity_z_m_s", "pressure_pa", "methanol_fraction")
    for name in required:
        values = f.get(name)
        if values is None or np.shape(values) != active.shape or not active.any():
            failures.append(name + ": missing/dimensions")
        elif not _finite(np.asarray(values)[active]):
            failures.append(name + ": nonfinite")
    for name in ("potential_v", "field_r_v_m", "field_z_v_m"):
        values = f.get(name)
        if values is None or not np.size(values) or not _finite(values):
            failures.append(name + ": missing/nonfinite")
    add("finite_fields", " Velocity ,  Pressure ,  Species and electric-field data ", " Data ", "failed" if failures else "pass",
        detail="; ".join(failures) if failures else " Finite values in active liquid and electric-field arrays ;  Quartz cells excluded from liquid checks .")

    C = np.asarray(f.get("methanol_fraction", []))
    if C.shape == active.shape and active.any() and _finite(C[active]):
        bounds = [float(C[active].min()), float(C[active].max())]
        bounded = bounds[0] >= -1e-7 and bounds[1] <= c.chemistry.sheath_methanol_fraction + 1e-7
        add("methanol_bounds", " Methanol fraction bounds ", " Solvent ", "pass" if bounded else "failed", bounds,
            detail=" Non-reactive ,  Internal transport preserves inlet concentration bounds ;  Allowed  1e-7  Floating-point tolerance .")
    else:
        add("methanol_bounds", " Methanol fraction bounds ", " Solvent ", "missing")

    residual("flow_residual", " Flow equation residual ", " Fluid ", flow.get("algebraic_relative_residual"),
             1e-6, "1e-6  Numerical screening tolerance ;  Mesh convergence and experimental validation are separate checks .")
    residual("solvent_balance", " Methanol flux balance ", " Solvent ", solvent.get("methanol_flux_relative_error"),
             1e-6, " Recorded inlet advection / Diffusion and outlet fluxes ;  Based on solver fluxes .")
    total_q = c.fluid.qs + c.fluid.qh
    residual("liquid_balance", " Local volumetric flux balance  /  Total flow ", " Fluid ",
             _ratio(solvent.get("max_fv_cell_mass_imbalance_m3_s"), total_q), 1e-6,
             " Conservation check  FV  Carrier flow ; FEM  To  FV  Correction reported separately .")
    residual("flux_projection", "FEM  To  FV  Maximum flux correction ", " Mesh ",
             solvent.get("max_fem_flux_projection_relative_correction"), .05,
             " Solver setting  5%  Screening threshold ;  Post-correction balance ,  Refine the tip mesh to assess the original field .")
    pe = _number(solvent.get("max_axial_cell_peclet"))
    add("cell_peclet", " Maximum axial cell size  Peclet  Number ", " Mesh ",
        "missing" if solvent.get("max_axial_cell_peclet") is None else
        ("failed" if pe is None or pe < 0 else ("warning" if pe > 2 else "pass")), pe,
        detail=" Bounded first-order upwind scheme ,  With  Pe  Axial numerical diffusion approximately  |u|Δz/2;  Focusing and exposure require their own convergence study .")
    converged = coupling.get("converged")
    add("coupling", " Fluid — Solvent — Electric-field iterations ", " Coupled ",
        "missing" if converged is None else
        ("failed" if not isinstance(converged, (bool, np.bool_)) else ("pass" if converged else "warning")),
        _number(coupling.get("iterations")), " Events ",
        " Recorded convergence flag ;  Unconverged trajectories ,  Species and pressure from the current iteration .")
    concentration_change = _number(coupling.get("max_concentration_change"))
    field_change = _number(coupling.get("max_field_relative_change"))
    coupling_tolerance = _number(coupling.get("tolerance"))
    change_values = (concentration_change, field_change, coupling_tolerance)
    if all(x is not None for x in change_values) and isinstance(converged, (bool, np.bool_)):
        consistent = (min(concentration_change, field_change) >= 0 and coupling_tolerance > 0
                      and bool(converged) == (max(concentration_change, field_change) < coupling_tolerance))
        add("coupling_flag_consistency", " Coupling flag and final update ", " Coupled ", "pass" if consistent else "failed",
            {"concentration_change": concentration_change, "field_relative_change": field_change,
             "requested_tolerance": coupling_tolerance},
            detail=" Consistency of the convergence flag with the final update ;  Additional checks for flow ,  Mesh and material properties .")
    if c.supply.sample_control == "pressure":
        supply = coupling.get("supply") or flow.get("supply") or {}
        residual("pressure_closure", " Pressure-supply residual ", " Fluid supply ", supply.get("pressure_residual_pa"),
                 c.supply.pressure_tolerance_pa, " Specified pressure ,  Hydrostatic pressure ,  Capillary resistance and tip  FEM  Back pressure .", signed=True)
        checks[-1]["unit"] = "Pa"
    residual("electric_residual", " Potential equation residual ", " Electric field ", electric.get("relative_residual"),
             1e-6, " Discrete steady-field balance ;  Moving-interface free charge and RC Circuit checks in the spray model .")
    ionic = electric.get("ionic") or {}
    if c.electrical.liquid_model == "resistive":
        residual("ionic_residual", " Ionic-current residual ", " Electric field ", ionic.get("relative_current_residual"),
                 1e-6, " Electroneutral binary electrolyte ;  Electrode reactions ,  Space-charge emission requires a separate closure .")
        current = _number(ionic.get("current_a"))
        residual("ionic_balance", " Local current balance  /  Specified current ", " Electric field ",
                 _ratio(ionic.get("max_cell_current_imbalance_a"), None if current is None else abs(current)),
                 1e-6, " Relative balance undefined at zero current ,  Potential residual remains available .")
        add("spray_current", " Spray-current boundary ", " Electric field ", "assumption", current, "A",
            " Blank current uses the open-circuit limit ;  Specified current boundary ,  Emission-current calibration is external to this model .")
    else:
        add("liquid_electrical_model", " Internal DC model ", " Electric field ", "assumption", c.electrical.liquid_model,
            detail=" Zero internal DC field in the ideal-conductor limit ;  Dielectric-liquid sensitivity model ,  Conductive electrolyte requires the resistive model .")

    n = _number(particles.get("count"))
    exited = _number(particles.get("exited"))
    wall = _number(particles.get("wall_contacts"))
    censored = _number(particles.get("censored_or_reverse"))
    valid_particle_counts = False
    if all(x is not None for x in (n, exited, wall, censored)):
        valid = n > 0 and all(0 <= x <= n and x.is_integer() for x in (n, exited, wall, censored))
        valid = valid and exited + wall + censored == n
        valid_particle_counts = valid
        add("particle_accounting", " Transport state accounting ", " Cell ", "pass" if valid else "failed",
            {"sampled": int(n), "exited": int(exited), "wall": int(wall), "censored": int(censored)},
            detail=" Exclusive states and conserved event counts . Wall capture follows the specified binding kinetics ;  Time-censored trajectories are reported separately .")
        if valid:
            ci = binomtest(int(exited), int(n)).proportion_ci(method="exact")
            add("transport_sampling_interval", " Outlet fraction  95%  Sampling interval ", " Statistics ", "pass", [float(ci.low), float(ci.high)],
                detail="Clopper-Pearson  Exact binomial interval ,  Independent transport samples within the integration interval ;  Excludes post-censoring outcomes ,  Material properties ,  Wall ,  Mesh and biological model errors .")
            add("transport_censoring_bounds", " Censoring interval for the outlet fraction ", " Cell ", "warning" if censored else "pass",
                [exited / n, (exited + censored) / n],
                detail=" Lower bound: observed exits ;  Upper bound: all unresolved trajectories eventually exit ,  Wall-capture states retained . Censoring bound ,  Model-based interval .")
    else:
        keys = ("count", "exited", "wall_contacts", "censored_or_reverse")
        corrupted = any(particles.get(key) is not None and _number(particles[key]) is None for key in keys)
        add("particle_accounting", " Transport state accounting ", " Cell ", "failed" if corrupted else "missing",
            detail=" Counts must be finite ,  Non-negative integers ;  Missing and invalid values are distinguished .")
    add("transport_censoring", " Censored or reversed trajectories ", " Cell ",
        "missing" if censored is None else ("warning" if censored else "pass"), censored, " Particles ",
        " Finite integration duration affects the observed outlet fraction ;  Extend integration or inspect reverse flow ,  Report censored outcomes separately .")
    saved = len(f.get("trajectory_ids", []))
    response_times = np.asarray(f.get("response_residence_time_s", []))
    pool = response_times.size
    response_exited = np.asarray(f.get("response_exited", []))
    response_ok = pool > 0 and response_times.shape == (pool,) and response_exited.shape == (pool,)
    if response_ok:
        response_ok = np.isin(response_exited, [0, 1]).all()
    if response_ok:
        exit_times = response_times[response_exited.astype(bool)]
        response_ok = _finite(exit_times) and bool(np.all(exit_times >= 0))
    if response_ok and valid_particle_counts:
        response_ok = pool == n and int(np.sum(response_exited)) == exited
    add("response_coverage", " Saved trajectories  /  Response count ", " Statistics ",
        "missing" if not pool else ("pass" if response_ok else "failed"),
        {"saved_tracks": saved, "full_responses": pool},
        detail=" Playback uses saved spatial histories ;  History lengths and exit counts must match the event ledger . Missing historical arrays ,  Load the source  cells.csv  Or save a new record .")
    if c.supply.orientation == "legacy":
        corrected = all(particles.get(key) for key in ("bulk_wall_boundary", "bulk_time_censoring"))
        add("legacy_transport_revision", " History  legacy  Capillary transport version ", " Source ", "pass" if corrected else "warning",
            detail=" History  legacy  Earlier transport version used different censoring and wall-contact rules . Historical boundary policy retained for comparison ,  Recompute particle responses ;  Flow fields remain reusable ."
            if not corrected else " Current  legacy  Finite-size wall exclusion ,  Binding and capillary time censoring .")
    fraction = c.cell.concentration_per_ml * 1e6 * (math.pi / 6) * (c.cell.diameter_um * 1e-6) ** 3
    add("dilute_fraction", " Inlet cell volume fraction ", " Applicability ", "warning" if fraction > .01 else "pass",
        fraction * 100, "%", "φ=C·πd³/6; 1%  Dilute-suspension screening threshold ,  Model-specific criterion . Cell collisions ,  Aggregation and two-way feedback omitted from the independent model .")
    add("confinement", " Cell diameter  /  Sample ID ", " Applicability ", "assumption",
        c.cell.diameter_um / c.geometry.inner_id_um,
        detail=" Finite-size exclusion and near-wall drag ,  Background velocity sampled at sphere centres ;  Excludes  Faxén  Velocity-curvature correction ,  Particle-scale surface flow and deformation are reduced models . High confinement requires additional validation .")
    add("wall_adhesion", " Quartz binding rate ", " Boundary ", "assumption", c.supply.wall_capture_rate_s, "s⁻¹",
        " Zero rate gives non-adhesive wall exclusion ;  Wall recovery depends on calibrated capture kinetics .")
    add("mechanical_damage", " Shear-damage coefficient ", " Cell ", "assumption", c.supply.shear_damage_coefficient,
        detail=" Damage model disabled when unspecified ;  Specified constitutive relation  hazard,  Shear dose and biological viability are distinct .")

    expected_rate = c.cell.concentration_per_ml * c.fluid.sample_flow_ul_min / 60000
    observed_rate = _number(events.get("rate_cells_s"))
    matches = observed_rate is not None and math.isclose(expected_rate, observed_rate, rel_tol=1e-9, abs_tol=1e-12)
    add("source_rate", " Continuous-source concentration — Flow consistency ", " Events ",
        "missing" if observed_rate is None else ("pass" if matches else "failed"), expected_rate, " Particles /s",
        "λ=C( Particles /mL)·Q(μL/min)/60000;  Uses the solved flow rate ,  Pressure control determines the resulting flow .",
        recorded_value=observed_rate)
    ledger_errors = []
    rows = events.get("summaries") or []
    for row in rows:
        values = [_number(row.get(k)) for k in ("initial_in_pipe", "injected", "nozzle_exits", "upstream_in_transit", "transport_unresolved")]
        raw_values = [row.get(k) for k in ("initial_in_pipe", "injected", "nozzle_exits", "upstream_in_transit", "transport_unresolved")]
        if any(raw is not None and value is None for raw, value in zip(raw_values, values)):
            ledger_errors.append("invalid")
        elif any(x is None for x in values):
            ledger_errors.append("missing")
        elif any(x < 0 or not x.is_integer() for x in values):
            ledger_errors.append("invalid")
        else:
            a, b, d, e, g = values
            ledger_errors.append(a + b - d - e - g)
    bad = any(x != 0 for x in ledger_errors if x != "missing")
    add("event_accounting", " Observation-window cell balance ", " Events ",
        "failed" if bad else ("missing" if not rows or "missing" in ledger_errors else "pass"), ledger_errors,
        " Particles ", (" Observation events generated on demand ;  Event ledger not included ,  Open the process / Or acquisition view for the event ledger ."
                 if events.get("finite_window_pending") else
                 " Initial population  +  Injected in window  =  Exited  +  Remaining in transit  +  Pending continuation ;  Finite-window Poisson variability ."))

    if c.spray.enabled:
        spdata = data.spray
        completed = spray.get("completed")
        elapsed = _number(spray.get("elapsed_s"))
        requested = _number(spray.get("requested_s"))
        if completed is None or spray.get("elapsed_s") is None or spray.get("requested_s") is None:
            horizon_status = "missing"
        elif not isinstance(completed, (bool, np.bool_)) or elapsed is None or requested is None or elapsed < 0 or requested <= 0:
            horizon_status = "failed"
        elif bool(completed) != (elapsed >= requested * (1 - 1e-10)):
            horizon_status = "failed"
        else:
            horizon_status = "pass" if completed else "warning"
        add("spray_horizon", " Interface integration duration ", " Electrospray ",
            horizon_status,
            {"elapsed_ms": None if elapsed is None else elapsed * 1000,
             "requested_ms": None if requested is None else requested * 1000},
            detail=" Target integration duration  Taylor  Cone ; max_steps  Truncated integration reported explicitly .")
        residual("spray_mass", " Two-phase liquid-volume balance ", " Electrospray ", spray.get("max_relative_liquid_volume_error"),
                 1e-6, " Volume balance from interface updates ;  Separate checks for charge ,  Solvent and energy balances .")
        phi = None if spdata is None else spdata.get("liquid_fraction")
        if phi is not None and np.size(phi):
            bounded = _finite(phi) and np.min(phi) >= -1e-7 and np.max(phi) <= 1 + 1e-7
            add("phase_bounds", " Phase-fraction bounds ", " Electrospray ", "pass" if bounded else "failed",
                None if not _finite(phi) else [float(np.min(phi)), float(np.max(phi))],
                detail="0  To  1  Droplet resolution depends on mesh spacing .")
        else:
            add("phase_bounds", " Phase-fraction bounds ", " Electrospray ", "missing")
        electric_coupling=spray.get('electric_coupling')
        if electric_coupling:
            residual('spray_charge_balance',' Global free-charge balance ',' Electrospray ',
                abs(electric_coupling.get('global_charge_balance_error_c',float('nan'))),1e-18,
                ' Charge advection and implicit  Ohmic  Current ;  Conductivity calibration is a separate requirement .')
            residual('spray_circuit_balance',' Upstream RC Electrical-circuit residual ',' Electrospray ',
                electric_coupling.get('max_circuit_balance_error_a'),1e-12,
                ' Conduction and displacement current ;  Specified effective conductivity ,  Transient supply-current prediction .')
            detail=' Axisymmetric free-charge advection ,  Implicit Ohmic Relaxation , RC Total supply and Maxwell Feedback ;  Conductivity model ,  Ions PNP,  Electrode reactions and steady-jet validation remain separate .'
        else:detail=' No free-charge history in this record ;  Specified potential retained Maxwell Approximation . Run the updated model to create a new record .'
        add('spray_charge_closure',' Detached-droplet charge and current balance ',' Closure ','assumption',detail=detail)
    add("material_properties", " Mixture properties and methanol transport ", " Material properties ", "assumption",
        detail=" Methanol is electrically neutral ,  Zero valence ;  Ion migration and electroosmosis couple electric fields to flow . Constant diffusivity ,  Logarithmic viscosity mixing ,  Permittivity and electrolyte conductivity require calibration .")
    add("geometry_closure", " Quartz outlets and capillary geometry ", " Boundary ", "assumption",
        detail=" Inner ,  Middle ,  Outer-capillary recess included ;  Sharp edges ,  Contact angle / Hysteresis ,  Outlet rounding and bends /Dean  Vortical flow omitted .")
    from .capillary_transport import assess_capillary
    capillary=None
    try:
        capillary=assess_capillary(data)
    except (ValueError,KeyError,IndexError,TypeError) as error:
        add('long_pipe_history',' Capillary history availability ',' Data ','failed',detail=str(error))
    if capillary is not None:
        end=capillary['sections'][-1] if capillary['sections'] else None
        add('long_pipe_fields',' Capillary and tip coupling domains ',' Closure ','assumption',
            detail=' Fully developed pressure-driven capillary flow ,  Gravity / Buoyancy ,  Slip lift and wall resistance ;  Electrophoresis , DEP, EOF Species coupling in the resolved tip . Full-capillary electrical boundary ,  Non-axisymmetric flow ,  Bend secondary flow omitted from the baseline model .')
        add('long_pipe_transverse_stream',' Capillary cell stream :  Axis offset and transverse width ',' Focusing ',
            'missing' if end is None else ('warning' if capillary['near_wall_cluster_in_model'] else 'assumption'),
            None if end is None else dict(centroid_offset_um=end['centroid_offset_um'],spread_r95_um=end['spread_r95_um'],saved_crossings=end['count']),
            detail=' A narrow population can be offset toward the lower wall ,  Centroid and axis-based focusing differ . Independent rigid-sphere reference / Collisions ,  Stream width and single-file ordering are distinct ;  Cross-section uses saved station crossings .')
        add('long_pipe_brownian_width',' Thermal dispersion over long residence times ',' Focusing ',
            'warning' if capillary['brownian_width_warning'] else 'assumption',
            dict(brownian_enabled=capillary['bulk_brownian_enabled'],free_1d_rms_um=capillary['free_1d_brownian_rms_um']),
            detail=' Size-threshold disabled 8 μm Thermal motion broadens deterministic near-wall streams . Free diffusion RMS Order-of-magnitude reference ;  Near-wall fluctuations require mobility-dependent noise and thermal drift ,  Separate free-space RMS Diffusion and stream-width measures .')
        add('axial_ordering',' Radial focusing and axial arrivals ',' Events ','assumption',
            dict(independent_poisson_gap_cv=1.,mean_input_gap_s=capillary['mean_poisson_gap_s']),
            detail=' Poisson inlet arrivals + Independent transit delays preserve stochastic steady arrivals . Axial hydrodynamic coupling ,  Collisions / Exclusion and aggregation omitted from the independent reference ;  Focusing and unique-source acquisition use different metrics .')
        add('focusing_threshold',' Outlet focusing definition ',' Focusing ','assumption',c.cell.focusing_radius_um,'μm',
            ' Fraction of outlet sphere centres within the radial threshold ;  Threshold-based radial metric . Single-phase focusing preserves liquid continuity .')
    add("ms_capture_closure", " Inlet state and detection efficiency ", " Closure ", "assumption",
        detail=" Intact-cell transport ,  Single-source  Full MS  Acquisition rate ,  Lipid-carrier transfer metrics ;  Inlet aspiration requires a separate boundary model  CFD,  Post-lysis mass partition and ion-detection calibration .")
    add("mesh_validation", " Mesh / Time-step and experimental validation ", " Validation ", "missing",
        detail=" Residual and balance checks are separate from mesh ,  Time-step convergence and experimental calibration ;  Predictive accuracy requires validation .")
    counts = {status: sum(x["status"] == status for x in checks)
              for status in ("pass", "warning", "failed", "missing", "assumption")}
    status = next((s for s in ("failed", "warning", "missing", "assumption") if counts[s]), "pass")
    return _json_safe({"schema_version": 1, "status": status, "counts": counts,
            "summary": f" Pass  {counts['pass']};  Review  {counts['warning']};  Inconsistent  {counts['failed']};  Evidence pending  {counts['missing']};  Conditions / Not included  {counts['assumption']}",
            "source": str(data.source), "checks": checks, "references": [dict(x) for x in REFERENCES],
            "capillary_transport": capillary,
            "scope_note": " Numerical checks and model assumptions ,  Validation status of the coupled  CyESI→ MS prediction ."})
