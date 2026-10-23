"""Matched hydraulic-feed voltage comparisons using native 3-D solves."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .volume import solve_volume, VolumeSettings
from .phase import solve_phase, PhaseSettings


@dataclass
class VoltageComparison:
    current_volume: object
    zero_volume: object
    current_phase: object
    zero_phase: object
    delta_velocity: np.ndarray
    delta_phase_velocity: np.ndarray
    delta_liquid_fraction: np.ndarray
    diagnostics: dict


def _at(times, values, target):
    i = int(np.searchsorted(times, target, side='right'))
    if i == 0:return values[0].copy()
    if i >= len(times):return values[-1].copy()
    weight = (target-times[i-1])/(times[i]-times[i-1])
    return (1-weight)*values[i-1]+weight*values[i]


def _weighted_speed(velocity, grid, mask):
    weights = grid.volume[mask]
    speed2 = np.sum(velocity[mask]**2, axis=-1)
    return float(np.sqrt(np.sum(speed2*weights)/max(np.sum(weights), 1e-300)))


def solve_voltage_comparison(config, settings=None, progress=lambda value: None,
                             phase_settings=None, current_volume=None, current_phase=None):
    """Compare E on/off at identical prescribed actual inlet flow rates.

    In pressure-controlled experiments, use the already solved actual feed Q
    here. This isolates electric forcing; it does not claim a second complete
    constant-pressure hydraulic-network closure for 0 V.
    """
    config = deepcopy(config).validate()
    settings = deepcopy(settings or (current_volume.settings if current_volume is not None else VolumeSettings())).validate()
    phase_settings = deepcopy(phase_settings or (current_phase.settings if current_phase is not None else
        PhaseSettings(duration_us=1., axial_cells=12, max_steps=40))).validate()
    if current_volume is None:
        progress('Voltage comparison: native 3-D fields at the selected voltage')
        current_volume = solve_volume(config, settings, progress)
    if current_phase is None:
        progress('Voltage comparison: native 3-D phase with electric stress')
        current_phase = solve_phase(current_volume, config, phase_settings, progress)
    zero_config = deepcopy(config)
    zero_config.electrical.voltage_kv = 0.
    zero_config.electrical.spray_current_na = 0.
    # A sample-pool voltage is a second independent electrode. For the voltage
    # off control, ground it as well if it was explicitly supplied.
    zero_settings = deepcopy(settings)
    if zero_settings.sample_potential_v is not None:zero_settings.sample_potential_v = 0.
    progress('Voltage comparison: independent native 3-D 0 V fields')
    zero_volume = solve_volume(zero_config, zero_settings, progress)
    progress('Voltage comparison: native 3-D phase with zero electric stress')
    zero_phase = solve_phase(zero_volume, zero_config, phase_settings, progress)
    if any(not np.array_equal(a, b) for a, b in zip(current_volume.grid.edges, zero_volume.grid.edges)):
        raise ValueError('Voltage comparison volume grids must match')
    if any(not np.array_equal(a, b) for a, b in zip(current_phase.grid.edges, zero_phase.grid.edges)):
        raise ValueError('Voltage comparison phase grids must match; use the same phase settings')
    matched = float(min(current_phase.time_s[-1], zero_phase.time_s[-1]))
    current_u = _at(current_phase.time_s, current_phase.velocity, matched)
    zero_u = _at(zero_phase.time_s, zero_phase.velocity, matched)
    current_phi = _at(current_phase.time_s, current_phase.liquid_fraction, matched)
    zero_phi = _at(zero_phase.time_s, zero_phase.liquid_fraction, matched)
    delta = current_volume.velocity-zero_volume.velocity
    phase_delta = current_u-zero_u
    liquid = current_volume.material == 1
    initial_liquid = current_phase.liquid_fraction[0] > .5
    x, y, z = np.meshgrid(*current_phase.grid.centers, indexing='ij')
    downstream = z > 0
    v = current_phase.grid.volume
    current_outside = float(np.sum(current_phi[downstream]*v[downstream]))
    zero_outside = float(np.sum(zero_phi[downstream]*v[downstream]))
    diag = dict(method='Independent native 3-D electric-on/off solves at matched actual inlet flow',
        voltage_v=config.electrical.voltage_kv*1000, control_voltage_v=0.,
        sample_flow_ul_min=config.fluid.sample_flow_ul_min,
        sheath_flow_ul_min=config.fluid.sheath_flow_ul_min,
        hydraulic_control='fixed actual liquid feed in both cases; pressure-network reclosure is not implied',
        matched_phase_time_s=matched,
        selected_phase_completed=bool(current_phase.diagnostics['completed']),
        zero_phase_completed=bool(zero_phase.diagnostics['completed']),
        liquid_speed_rms_m_s=_weighted_speed(current_volume.velocity, current_volume.grid, liquid),
        zero_liquid_speed_rms_m_s=_weighted_speed(zero_volume.velocity, zero_volume.grid, liquid),
        liquid_velocity_difference_rms_m_s=_weighted_speed(delta, current_volume.grid, liquid),
        liquid_velocity_difference_max_m_s=float(np.max(np.linalg.norm(delta[liquid], axis=-1))),
        phase_velocity_difference_rms_m_s=_weighted_speed(phase_delta, current_phase.grid, current_phase.material != 2),
        phase_liquid_velocity_difference_rms_m_s=_weighted_speed(phase_delta, current_phase.grid, initial_liquid),
        phase_velocity_difference_max_m_s=float(np.max(np.linalg.norm(phase_delta, axis=-1))),
        selected_downstream_liquid_volume_m3=current_outside,
        zero_downstream_liquid_volume_m3=zero_outside,
        downstream_liquid_volume_difference_m3=current_outside-zero_outside,
        phase_fraction_difference_max=float(np.max(abs(current_phi-zero_phi))),
        mechanisms=[
            'Liquid-field electroosmotic wall slip changes the 3-D velocity profile and required pressure; prescribed mean feed flow remains fixed.',
            'The shared 3-D gas/liquid phase transient responds to Maxwell stress, advected free charge, Ohmic charge relaxation and capillary force, changing its velocity and liquid distribution.',
            'Open-circuit liquid has zero bulk electric field; with no specified current the bulk electroosmotic difference should vanish rather than be fabricated.',
        ],
        limits=[
            'Same actual flow is a controlled comparison, not a prediction that a pressure-controlled source keeps identical flow when voltage changes.',
            'Short-time leaky-dielectric phase dynamics do not establish a steady charged cone-jet or droplet spectrum; distributed electrode capacitance and species-resolved ion migration remain unclosed.',
            'The two phase histories are linearly interpolated to the same completed physical time; unfinished integration is reported.',
            'Cell viability, ion yield and true MS detection efficiency do not follow from a liquid-speed comparison.',
        ])
    return VoltageComparison(current_volume, zero_volume, current_phase, zero_phase,
                             delta, phase_delta, current_phi-zero_phi, diag)
