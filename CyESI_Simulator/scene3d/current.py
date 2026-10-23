"""Conditional cone-jet current scale, separate from prescribed-current CFD.

I = alpha sqrt(gamma K Q) is a steady cone-jet regime estimate, not an onset
criterion or a charge-conserving transient prediction. See docs/SCENE3D.md.
"""
from __future__ import annotations
import math

SOURCE = 'https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/analysis-of-selfheating-in-electrosprays-operating-in-the-conejet-mode/93B0FDA2530202F7430F4BC10FD5B09E'
COEFFICIENT_SOURCE = 'https://doi.org/10.1017/jfm.2025.10410'


def estimate_current(config, alpha=2.47):
    """Return an explicitly conditional SI estimate without fitting user current.

    The supporting sheath conductivity is only a proxy for the effective jet
    conductivity. It is not a measured mixture conductivity. The range 0.2--1
    microampere is compared after prediction; it never sets alpha or K here.
    No estimate is clipped to the ohmic supply limit and passed off as valid.
    """
    alpha=float(alpha)
    if not math.isfinite(alpha) or alpha<=0:
        raise ValueError('Cone-jet coefficient must be positive and finite')
    e,f,g=config.electrical,config.fluid,config.geometry
    k=e.sheath_conductivity_s_m
    q=f.qs+f.qh
    if min(k,q,f.surface_tension)<=0:
        raise ValueError('Conductivity, total liquid flow and surface tension must be positive')
    area=math.pi*(g.c*g.c-g.b*g.b)
    resistance=e.contact_distance_mm*.001/(k*area)
    current=alpha*math.sqrt(f.surface_tension*k*q) if e.voltage_kv else 0.
    voltage=abs(e.voltage_kv)*1000
    drop=current*resistance
    # V/R is a necessary supply bound, not a stable spray load-line closure.
    feasible=drop<=voltage
    return dict(mode='conditional_steady_cone_jet_scaling',formula='I = alpha sqrt(gamma K Q)',
        coefficient=alpha,predicted_current_a=current,predicted_current_ua=current*1e6,
        total_liquid_flow_m3_s=q,conductivity_proxy_s_m=k,surface_tension_n_m=f.surface_tension,
        contact_length_m=e.contact_distance_mm*.001,sheath_annulus_area_m2=area,
        remote_resistance_ohm=resistance,ohmic_supply_upper_bound_a=voltage/resistance,
        estimated_remote_voltage_drop_v=drop,estimated_local_voltage_v=max(0.,voltage-drop) if feasible else None,
        electrical_supply_feasible=feasible,stable_cone_jet_established=None,
        user_observed_range_ua=[.2,1.],within_user_range=.2<=current*1e6<=1.,
        assumed_coefficient_sensitivity_ua=[a*math.sqrt(f.surface_tension*k*q)*1e6 if voltage else 0. for a in [1.,3.]],
        sensitivity_is_confidence_interval=False,sources=[SOURCE,COEFFICIENT_SOURCE],
        limits=[
            'Valid only conditionally within a steady cone-jet regime; onset, dripping and multi-jet transitions are not predicted.',
            'Supporting sheath conductivity is an unmeasured proxy for effective jet conductivity; mixed ammonium formate, methanol and temperature can change it.',
            'The coefficient is a published approximate scaling value, not calibrated to the supplied current range or this coaxial device.',
            'Gas assistance, contact electrochemistry, space charge and evaporation are not resolved by this estimate.',
            'Remote V/R is only a necessary electrical supply bound; additional resolved liquid and spray voltage drops remain.',
            'A prescribed-current field solve verifies continuity of the imposed current and is not an independent current prediction.',
        ])
