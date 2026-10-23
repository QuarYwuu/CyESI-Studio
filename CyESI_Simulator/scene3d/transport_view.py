"""Display adapters for deposited cells and common-clock local studies.

These adapters never reinterpret a finite-radius tip crossing or a censored
trajectory as adhesion. Only an explicit capture/adhesion state may remain
stationary on a wall. Baseline theoretical acquisition rates remain separate
from the finite observation-window interaction study.
"""
from __future__ import annotations

import numpy as np


STATE_NAMES = {0: 'not_entered', 1: 'mobile', 2: 'cell_bound', 3: 'wall_bound',
               4: 'outlet_bound', 5: 'exited', 6: 'censored'}
COLORS = {'mobile': [.15, .8, .85, 1.], 'cell_bound': [.95, .5, .14, 1.],
          'wall_bound': [.95, .25, .65, 1.], 'outlet_bound': [.85, .25, .95, 1.]}


def _limit(cells, limit, selected_id):
    if len(cells) <= limit:
        return cells
    chosen = cells[:limit]
    selected = next((cell for cell in cells if cell['id'] == selected_id), None)
    if selected is not None and all(cell['id'] != selected_id for cell in chosen):
        chosen[-1] = selected
    return chosen


def add_bulk_deposits(scene, value, time_s, selected_id, max_cells):
    """Keep explicit irreversible long-tube captures visible after capture.

    The finite saved prehistory is the only counted deposition history. An
    indefinitely operated irreversible adsorber has no finite steady deposit
    population; deposits predating the saved source are not extrapolated.
    """
    cfg = scene.data.config
    status = scene.data.fields.get('cell_bulk_status')
    if not cfg.supply.wall_capture_rate_s or status is None:
        return value
    model = scene.model
    records = model.records
    captured_templates = np.asarray(status)[scene.data.ids] == 2
    templates = records['template_id']
    captured = np.flatnonzero(captured_templates[templates] &
        (records['injection_s']+model.delays[templates] < time_s))
    if not len(captured):
        return value
    existing = {cell['id'] for cell in value['cells']}
    count = len(captured)
    candidates = captured[:max_cells]
    if selected_id is not None and selected_id in captured and selected_id not in candidates:
        candidates = np.r_[candidates[:-1], selected_id]
    cells = list(value['cells'])
    for event_id in candidates:
        if int(event_id) in existing:
            continue
        index = int(templates[event_id])
        _, xyz, _, _ = scene._track_xyz(index)
        local = xyz[-1]
        cells.append(dict(id=int(event_id), event_id=int(event_id),
            position=scene._global(local).tolist(), local_position_m=local.tolist(),
            radius_m=cfg.cell.diameter_um*.5e-6,
            color=[.95, .25, .65, 1.], age_s=float(time_s-records['injection_s'][event_id]),
            methanol=0., trajectory_index=index, computed_bulk_xyz=True,
            stage='long_pipe', selected=int(event_id)==selected_id,
            interaction_status='bulk_wall_bound', conditional_adhesion=True))
    cells = _limit(cells, max_cells, selected_id)
    value['cells'] = cells
    # Parent counts classify all terminated non-exits as unresolved. Move only
    # known calibrated-hazard captures to the explicit adhesion category.
    value['counts']['transport_unresolved'] -= count
    value['counts']['wall_adhered'] = count
    value['live_physical_count'] += count
    value['visible_count'] = len(cells)
    value['omitted_count'] = max(0, value['live_physical_count']-len(cells))
    visible_count = sum(c.get('interaction_status') == 'bulk_wall_bound' for c in cells)
    value['bulk_wall_adhesion'] = dict(count=count, visible_count=visible_count,
        omitted_count=count-visible_count,
        scope='Conditional irreversible capture in the developed long-tube model; finite recorded feed prehistory only, not a steady deposit population')
    return value


def apply_interaction_frame(scene, value, time_s, selected_id, max_cells):
    result = getattr(scene, 'interaction_result', None)
    if result is None:
        return value
    params = result.parameters
    start, requested_end = params.start_s, params.start_s+params.duration_s
    end = float(result.time_s[-1])
    active = start-1e-12 <= time_s <= end+1e-12
    summary = dict(active=active, start_s=start, end_s=end, requested_end_s=requested_end,
        scope='Common-clock continuously fed local-tip scenario; global acquisition theory uses the independent baseline')
    value['interaction_study'] = summary
    if not active:
        return value
    k = min(max(int(np.searchsorted(result.time_s, time_s, side='right')-1), 0), len(result.time_s)-1)
    states = result.states[k]
    summary['counts'] = {name: int(np.sum(states == code)) for code, name in STATE_NAMES.items()}
    summary['snapshot_time_s'] = float(result.time_s[k])
    summary['event_count'] = len(result.event_ids)
    rows = result.frame(time_s)
    affected = {int(event_id) for event_id, state in zip(result.event_ids, states) if state != 0}
    original = value['cells']
    cells = [cell for cell in original if cell['id'] not in affected]
    radius = scene.data.config.cell.diameter_um*.5e-6
    records = scene.model.records
    for row in rows:
        event_id = int(row['event_id'])
        position = np.asarray(row['local_xyz_m'], float)
        status = row['status']
        methanol = row.get('methanol_fraction')
        if methanol is None or not np.isfinite(methanol):
            methanol = scene.volume.sample(position[None, :], 'methanol')[0]
        methanol = float(np.clip(methanol, 0., 1.))
        index = int(records['template_id'][event_id])
        cells.append(dict(id=event_id, event_id=event_id,
            position=scene._global(position).tolist(), local_position_m=position.tolist(),
            radius_m=radius, color=COLORS.get(status, COLORS['mobile']),
            age_s=float(time_s-records['injection_s'][event_id]), methanol=methanol,
            trajectory_index=index, computed_bulk_xyz=True, stage='resolved_tip',
            selected=event_id==selected_id, interaction_status=status,
            cluster_id=row.get('cluster_id'), interaction_contact_code=row.get('contact_code',0),
            conditional_adhesion=status in COLORS and status != 'mobile'))
    shown = _limit(cells, max_cells, selected_id)
    # Keep overall baseline accounting clearly separate from study outcomes;
    # the finite, censored local population is not a revised 600 s rate.
    value['counts_scope'] = 'Independent full-feed baseline, plus explicit upstream captures; local interaction counts are separate'
    ids = np.asarray(sorted(affected), int)
    baseline_live = ((time_s-records['injection_s'][ids] >= 0.) &
        (time_s-records['injection_s'][ids] <= scene.model.delays[records['template_id'][ids]]))
    live = value['live_physical_count']-int(np.sum(baseline_live))+len(rows)
    value.update(cells=shown, visible_count=len(shown), live_physical_count=live,
                 omitted_count=max(0, live-len(shown)))
    if selected_id in affected:
        value['selected_charge'] = None
        value['selected_charge_scope'] = 'Baseline charge history is not reused on an interaction-perturbed trajectory'
        value['external_selected'] = None
    return value


def add_interaction_event(scene, value, event_id):
    result = getattr(scene, 'interaction_result', None)
    if result is None or int(event_id) not in result.event_ids:
        return value
    history = result.event_history(int(event_id))
    if history is not None:
        local = np.asarray(history['local_xyz_m'], float).reshape(-1, 3)
        history['local_positions_m'] = local.tolist()
        history['positions'] = scene._global(local).tolist() if len(local) else []
        history['methanol'] = history['methanol_fraction']
    value['interaction_history'] = history
    value['interaction_affects_flight'] = True
    value['interaction_scope'] = 'Separate local observation-window trajectory; baseline nozzle_s, track, charge and acquisition status remain labeled baseline'
    return value
