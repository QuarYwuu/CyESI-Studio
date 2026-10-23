"""Module orchestration; no GUI imports, plotting or hidden parameter changes."""
from dataclasses import dataclass
import numpy as np
from .config import Config
from .fluid.navier_stokes import solve as solve_flow, annular_gas
from .chemistry.solvent_transport import solve as solve_solvent
from .electro.electrostatics import solve as solve_electric
from .electro.ionic_transport import solve as solve_ionic
from .particle.cell_transport import simulate as track_cells
from .particle.events import simulate_events
from .spray.level_set import simulate as solve_spray


@dataclass
class SimulationResult:
    config: Config
    flow: object
    solvent: object
    electric: object
    particles: object
    particles_zero_voltage: object
    events: dict
    spray: object
    coupling: dict
    warnings: list
    scan: list | None = None


def _fixed_flow_solution(config, progress=lambda x: None, initial_state=None):
    solvent, ionic = initial_state if initial_state is not None else (None, None)
    error = np.inf; field_error=0.
    for iteration in range(config.numerics.coupling_iterations):
        progress(f"Liquid FEM / solvent coupling {iteration+1}/{config.numerics.coupling_iterations}")
        flow = solve_flow(config, solvent, progress, ionic=ionic)
        updated = solve_solvent(config, flow)
        if solvent is not None:
            error = float(np.max(np.abs(updated.concentration-solvent.concentration)))
        solvent = updated
        if config.electrical.liquid_model == "resistive":
            previous_ionic=ionic
            ionic = solve_ionic(config, solvent)
            field_error=np.inf if previous_ionic is None else float(np.linalg.norm(ionic.field_z-previous_ionic.field_z)/max(np.linalg.norm(ionic.field_z),1e-20))
            solvent.ionic = ionic
        if max(error,field_error) < config.numerics.coupling_tolerance: break
    return flow, solvent, {"iterations": iteration+1, "max_concentration_change": error if np.isfinite(error) else None,
        "max_field_relative_change":field_error if np.isfinite(field_error) else None,
        "converged": bool(max(error,field_error) < config.numerics.coupling_tolerance),
        "tolerance": config.numerics.coupling_tolerance}


def liquid_solution(config, progress=lambda x: None, initial_state=None):
    if config.supply.sample_control == "pressure":
        from .fluid.supply import pressure_solution
        state=initial_state
        def warm_solve(point, notify):
            nonlocal state
            flow,solvent,coupling=_fixed_flow_solution(point,notify,state)
            state=(solvent,getattr(solvent,'ionic',None))
            return flow,solvent,coupling
        return pressure_solution(config, warm_solve, progress)
    return _fixed_flow_solution(config, progress, initial_state)


def run(config=None, progress=lambda x: None, on_stage=lambda name, result: None):
    from copy import deepcopy
    config = deepcopy(config or Config()).validate()
    flow, solvent, coupling = liquid_solution(config, progress)
    on_stage("solvent", solvent)
    progress("Solving dielectric-weighted electric potential")
    electric = solve_electric(config, getattr(solvent, "ionic", None))
    on_stage("electric", electric)
    progress(f"Sampling {config.cell.count} finite-sphere transport responses for the continuous cell source")
    particles = track_cells(config, solvent, electric, progress=progress)
    on_stage("particles", particles)
    if config.electrical.liquid_model == "conducting" or config.electrical.voltage_kv == 0:
        # The conducting upstream liquid has exactly zero DC bulk electric field.
        zero = particles
    else:
        progress("Tracking 0 kV comparison")
        from copy import deepcopy
        zero_config = deepcopy(config)
        zero_config.electrical.voltage_kv = 0.
        zero_config.electrical.spray_current_na = 0.
        _, zero_solvent, _ = liquid_solution(zero_config, progress)
        zero_electric = solve_electric(zero_config, getattr(zero_solvent, "ionic", None))
        zero = track_cells(zero_config, zero_solvent, zero_electric, progress=progress)
    events = simulate_events(config, particles)
    spray = None
    if config.spray.enabled:
        progress("Solving experimental two-phase nozzle transient")
        spray = solve_spray(config, solvent, electric, progress)
        on_stage("spray", spray)
    g, f = config.geometry, config.fluid
    warnings = [
        "Gas tube OD and hydraulic supply length are assumed inputs; confirm actual hardware.",
        "Medium properties and diffusivity are assumed constants, not measurements of 140 mM ammonium formate / methanol.",
        "Viscosity uses log mixing, omitting the excess viscosity of water/methanol mixtures.",
        "Long-tube orientation is editable; horizontal segments use 3D cross-section particle sedimentation. Local tip fluid remains axisymmetric.",
        "Finite-sphere dilute tracking does not resolve membrane deformation, cell-cell interaction or lysis.",
        "Methanol exposure is extracellular concentration, not extraction yield or a lysis probability.",
        "Nozzle exit is not MS detection. Post-nozzle cell/droplet transport, desolvation and MS capture are not solved.",
        "Steady internal flow uses a zero-traction outlet and does not receive transient two-phase backpressure.",
        "Upwind solvent transport has numerical diffusion; grid convergence is required for quantitative exposure."
    ]
    if not coupling["converged"]:
        warnings.append("Fluid/solvent fixed-point coupling has not met its requested tolerance.")
    projection = solvent.diagnostics["max_fem_flux_projection_relative_correction"]
    if projection > .05:
        warnings.append(f"Local FEM-to-FV flux projection changes a section flux by up to {projection:.1%}; refine the tip mesh.")
    if particles.diagnostics["censored_or_reverse"]:
        warnings.append("Some cells are censored by the resolved-tip timeout or reverse flow; exited-cell means are conditional.")
    if config.electrical.liquid_model == "dielectric":
        warnings.append("Dielectric liquid mode is a sensitivity experiment, not a validated steady DC model for conductive ammonium formate.")
    ionic = getattr(solvent, "ionic", None)
    if ionic is not None:
        warnings.extend(ionic.diagnostics["limitations"])
        if config.electrical.spray_current_na is None:
            warnings.append("Spray current is unknown: remote electrical model uses an OPEN-CIRCUIT limit, not a loaded spray prediction.")
    if spray is not None:
        warnings.extend(spray.diagnostics["limitations"])
        if not spray.diagnostics["completed"]: warnings.append("Spray stopped at max_steps before the requested physical time.")
    return SimulationResult(config, flow, solvent, electric, particles, zero, events, spray, coupling, warnings)


def derived(result):
    c = result.config; g, f = c.geometry, c.fluid
    pressure_bulk = 8*f.sample_viscosity*g.bulk_length*f.qs/(np.pi*g.a**4)
    from .fluid.supply import infer_pressure, hydrostatic_assist_pa
    return {"sample_control": c.supply.sample_control,
        "actual_sample_flow_ul_min": f.sample_flow_ul_min,
        "sample_feed_pressure_mbar": c.supply.sample_pressure_mbar if c.supply.sample_control == "pressure" else None,
        "developed_pressure_for_flow_mbar": infer_pressure(c),
        "hydrostatic_assist_pa": hydrostatic_assist_pa(c),
        "sample_mean_speed_m_s": f.qs/(np.pi*g.a*g.a),
        "sample_volume_nl": np.pi*g.a*g.a*g.sample_length_mm*1e-3*1e12,
        "sample_volume_over_flow_s": np.pi*g.a*g.a*g.sample_length_mm*1e-3/f.qs,
        "developed_bulk_pressure_drop_pa": pressure_bulk,
        "gas": annular_gas(c), "coordinate": "middle lip z=0; inner/outer lips z=-1 mm by default",
        "voltage_comparison": {"zero_kv": result.particles_zero_voltage.diagnostics,
                               "applied_kv": c.electrical.voltage_kv,
                               "applied": result.particles.diagnostics}}
