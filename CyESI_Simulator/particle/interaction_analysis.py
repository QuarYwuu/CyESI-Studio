"""Observation-window outcomes; never fit global Poisson theory to one movie."""
import numpy as np
from ..visualization.process_data import classify_packets


def acquisition_for_study(model,study):
    """Reclassify actual perturbed exits against the chosen collection gates.

    Retained/censored cells have no modeled packet. Outside-window origins
    retain baseline timing as a stated background assumption. Boundary gates
    are withheld where changed future arrivals are not yet known. These are
    finite-window conditional signal counts, not measured MS detection rates.
    """
    p=model.acquisition;start=float(study.time_s[0]);end=float(study.time_s[-1])
    ids=np.asarray(study.event_ids,int);onsets=model.onsets.copy()
    onsets[ids]=study.exit_s+model.parameters.packet_delay_ms*.001
    cohort=np.zeros(len(onsets),bool)
    cohort[ids]=np.isfinite(study.exit_s)&(study.exit_s>=start)&(study.exit_s<end)
    valid=np.isfinite(onsets)
    ledger,scans=classify_packets(onsets[valid],model.records['event_id'][valid],model.delivered[valid],
        model.row,p,cohort[valid])
    boundary_ids=set()
    for scan in scans:
        if any(a<start-1e-12 or b>end+1e-12 for a,b in scan['gates']):boundary_ids.update(scan['event_ids'])
    width=p.signal_width_ms*.001
    for row in ledger:
        row['baseline_status']=model.by_id.get(row['event_id'],{}).get('status')
        if row['event_id'] in boundary_ids or row['packet_onset_s']+width>end+1e-12:
            row['status']='boundary_unresolved';row['captured_fraction']=None;row['pure_captured_fraction']=None
    counts={key:sum(row['status']==key for row in ledger) for key in
        ('pure_complete','pure_partial','pure_impulse','mixed_only','missed','not_delivered','boundary_unresolved')}
    return dict(start_s=start,end_s=end,duration_s=end-start,local_exit_count=len(ledger),
        local_outlet_count_per_second=None if end<=start else len(ledger)/(end-start),counts=counts,ledger=ledger,
        retained_at_end=study.diagnostics['retained_at_end'],
        source_rate_s=model.source_rate,global_theoretical_rate_modified=False,
        notes=['Actual event exit times are reclassified, including correlated cluster exits; no independent Poisson formula is applied to the perturbed cohort.',
            'Only packets assigned by the existing delivery/width/delay scenario are evaluated; this is not an intact-cell or actual mass-spectral identification rate.',
            'Background origins outside this study retain baseline timing. Gates reaching beyond the computed window are boundary-unresolved.',
            'Initial residents are included: exits divided by window duration is an observed flux, not injection-cohort recovery. Retained cells are not automatically irreversible losses.'])
