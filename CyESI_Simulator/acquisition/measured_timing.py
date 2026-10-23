"""Use exported accumulation gates instead of assuming a periodic scan clock.

CSV times MUST be accumulation starts, not Orbitrap readout timestamps. RAW
scan retention times need conversion using the instrument acquisition semantics
before import; this module deliberately does not guess that offset.
"""
from pathlib import Path
import csv
import html
import json
import math
import numpy as np


def load_gates(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'accumulation_start_s','actual_it_ms'} <= set(reader.fieldnames or []):
            raise ValueError('CSV  Required:  accumulation_start_s, actual_it_ms;  Do not interpret  RAW  scan timestamps as injection start times .')
        rows=[]
        for number,row in enumerate(reader,2):
            try:start=float(row['accumulation_start_s']);it=float(row['actual_it_ms'])
            except (ValueError,TypeError) as error:raise ValueError(f'CSV  Index  {number}  Invalid row time ') from error
            if not math.isfinite(start) or not math.isfinite(it) or it<=0:
                raise ValueError(f'CSV  Index  {number}  Rows require a finite start time and positive measured  IT')
            rows.append({'accumulation_start_s':start,'actual_it_ms':it,
                         'resolution_at_mz200':row.get('resolution_at_mz200','')})
    if not rows:raise ValueError('CSV  No acquisition gate data ')
    starts=np.array([r['accumulation_start_s'] for r in rows]);ends=starts+np.array([r['actual_it_ms'] for r in rows])*.001
    if np.any(np.diff(starts)<=0) or np.any(ends[:-1]>starts[1:]+1e-12):
        raise ValueError(' Acquisition gates must increase in time without overlap ;  This importer supports  Microscans=1  per-spectrum acquisition gates .')
    return rows


def analyze_gates(gates, nozzle_rate_s, signal_width_ms=0., delivery_fraction=1.,
                  identification_efficiency=None, observation_duration_s=None, source=''):
    if not math.isfinite(nozzle_rate_s) or nozzle_rate_s<0 or not math.isfinite(signal_width_ms) or signal_width_ms<0 or not 0<=delivery_fraction<=1:
        raise ValueError(' Input rate ,  Invalid signal width or delivery fraction ')
    if identification_efficiency is not None and not 0<=identification_efficiency<=1:raise ValueError(' Identification efficiency must be unset or within  0–1  Inner ')
    starts=np.array([r['accumulation_start_s'] for r in gates],float)
    it=np.array([r['actual_it_ms'] for r in gates],float)*.001
    if not len(starts) or not np.isfinite(starts).all() or not np.isfinite(it).all() or np.any(it<=0) or np.any(np.diff(starts)<=0):raise ValueError(' Invalid acquisition gate times ')
    ends=starts+it
    if np.any(ends[:-1]>starts[1:]+1e-12):raise ValueError(' Adjacent acquisition gates overlap, or  Microscans  Not expanded ')
    width=signal_width_ms*.001;left=starts-width;right=ends;support=right-left
    rate=nozzle_rate_s*delivery_fraction;mu=rate*support
    p0=np.exp(-mu);p1=mu*p0;nonempty=-np.expm1(-mu)
    # For ordered, nonoverlapping gates expanded by a common packet width,
    # supports containing a tagged onset have ordered endpoints. Higher-order
    # void inclusion/exclusion cancels, leaving single supports minus adjacent
    # pairs. This counts each cell origin once across any number of spectra.
    overlap=np.maximum(0.,np.minimum(right[:-1],right[1:])-np.maximum(left[:-1],left[1:]))
    union=right[1:]-left[:-1]
    unique=rate*(np.sum(support*p0)-np.sum(overlap*np.exp(-rate*union)))
    expected_nonempty=float(np.sum(nonempty));expected_pure=float(np.sum(p1))
    # Exact span of provided gates; no assumed last transient or hidden overhead.
    span=float(ends[-1]-starts[0])
    duration=span if observation_duration_s is None else float(observation_duration_s)
    if not math.isfinite(duration) or duration<=0 or duration<span-1e-12:raise ValueError(' The observation duration must cover all acquisition gates ')
    per_scan=[dict(row,empty_probability=float(p0[i]),pure_probability=float(p1[i]),
                  mixed_probability=float(max(0.,nonempty[i]-p1[i]))) for i,row in enumerate(gates)]
    return {'source':str(source),'gate_count':len(gates),'nozzle_cells_s':nozzle_rate_s,
        'packet_cells_s':rate,'signal_width_ms':signal_width_ms,'delivery_fraction':delivery_fraction,
        'observation_duration_s':duration,'duration_definition':'provided duration or first accumulation start to last accumulation end; no inferred last transient',
        'mean_actual_it_ms':float(np.mean(it)*1000),'min_actual_it_ms':float(np.min(it)*1000),'max_actual_it_ms':float(np.max(it)*1000),
        'median_start_spacing_ms':float(np.median(np.diff(starts))*1000) if len(starts)>1 else None,
        'expected_pure_spectra':expected_pure,'expected_nonempty_spectra':expected_nonempty,
        'aggregate_nonempty_single_cell_purity':expected_pure/expected_nonempty if expected_nonempty else None,
        'expected_unique_pure_cell_origins':float(unique),'unique_pure_opportunities_s':float(unique/duration),
        'conditional_identified_unique_cells_s':None if identification_efficiency is None else float(unique/duration*identification_efficiency),
        'actual_ms_detection_cells_s':None,'per_scan':per_scan,'limitations':[
            'Gate starts must be actual accumulation starts. RAW retention/readout timestamps are not automatically equivalent.',
            'Microscans=1, one nonoverlapping accumulation gate per exported record; no dd-MS2 mixing.',
            'Timing is treated as externally supplied, independent of a stationary Poisson source. AGC timing correlated with cell load needs a joint acquisition model.',
            'One rectangular packet per cell origin, common width. Unknown signal transport and identification remain explicit assumptions.',
            'Expected pure spectra and distinct pure cell origins are different. Pure sources do not prove adequate signal or identification.']}


def export(result,directory):
    from ..report.export import clean,write_csv
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'measured_timing.json').write_text(json.dumps(clean(result),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    write_csv(root/'per_scan.csv',result['per_scan'])
    purity=result['aggregate_nonempty_single_cell_purity'];purity=' Unknown ' if purity is None else f'{purity:.2%}'
    limitations=''.join(f'<li>{html.escape(x)}</li>' for x in result['limitations'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title> Import measured acquisition gates </title><style>body{{font:17px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:20px}}</style>
<h1> Import acquisition gates  /  Conditional single-cell acquisition opportunity </h1><p> Source : {html.escape(result['source'])}; {result['gate_count']}  acquisition gates .</p>
<p> Actual  IT  Range  {result['min_actual_it_ms']:.4g}–{result['max_actual_it_ms']:.4g} ms;  Computation duration  {result['observation_duration_s']:.6g} s.</p>
<p> Predicted outlet rate  {result['nozzle_cells_s']:.6g} cells/s;  Specified ion packets  {result['packet_cells_s']:.6g} /s;  Single-source fraction of nonempty spectra  {purity}.</p>
<p> Expected single-source spectra  {result['expected_pure_spectra']:.6g};  Unique single-source events  {result['expected_unique_pure_cell_origins']:.6g}  Count ;  Theoretical rates  {result['unique_pure_opportunities_s']:.6g} /s.</p>
<p> Measured  MS  Detection rate requires calibration .<a href="measured_timing.json"> Full computation </a> · <a href="per_scan.csv"> Per-spectrum probability  CSV</a></p><ul>{limitations}</ul></html>'''
    (root/'measured_timing_report.html').write_text(text,encoding='utf-8')
    return root
