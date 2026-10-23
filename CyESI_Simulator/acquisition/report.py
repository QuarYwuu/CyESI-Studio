"""Export acquisition settings, rates and event records."""
from pathlib import Path
import html
import json
import numpy as np
from .q_exactive import AcquisitionParameters,TRANSIENT_MS
from ..report.export import clean,write_csv
from ..visualization.figures import plt,serialized_plotting


@serialized_plotting
def figures(analysis,directory):
    root=Path(directory);rows=analysis['rows'];p=analysis['parameters']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fig.suptitle('Q Exactive Full MS | Conditional single-cell sampling opportunities',fontweight='bold')
    for injection in p['actual_injection_times_ms']:
        group=[x for x in rows if x['actual_injection_ms']==injection]
        x=np.arange(len(group));label=f'actual IT {injection:g} ms'
        axes[0,0].plot(x,[r['unique_pure_single_cell_events_s']*60 for r in group],'o-',label=label)
        axes[0,1].plot(x,[r['pure_single_cell_spectra_s'] for r in group],'o-',label=label)
    for ax in axes[0]:
        ax.set_xticks(range(4),[f'{r/1000:g}k' for r in TRANSIENT_MS]);ax.set_xlabel('Resolution at m/z 200');ax.legend(fontsize=8)
    axes[0,0].set(ylabel='Unique pure cell-origin events / min',title='Deduplicated; not a measured detection rate')
    axes[0,1].set(ylabel='Pure spectra / s',title='One long cell signal may yield multiple spectra')
    reference=[x for x in rows if x['resolution_at_mz200']==17500]
    labels=[f"IT {r['actual_injection_ms']:g}" for r in reference];bottom=np.zeros(len(reference))
    for key,label,color in [('empty_spectrum_probability','Empty','#a4b5c5'),
            ('single_cell_spectrum_probability','One cell origin','#229d91'),('mixed_spectrum_probability','Mixed origins','#d58353')]:
        values=np.array([r[key] for r in reference])*100
        axes[1,0].bar(labels,values,bottom=bottom,label=label,color=color);bottom+=values
    axes[1,0].set(ylabel='Recorded spectra (%)',title='17.5k: occupancy during ion accumulation',ylim=(0,100));axes[1,0].legend(fontsize=8)
    # Same accumulation-time occupancy applies at each resolution, while duty
    # and unique throughput differ with the readout period.
    axes[1,1].axis('off')
    text=(f"Nozzle cells: {analysis['nozzle_cells_s']:.3g}/s\n"
        f"Assumed delivery: {p['signal_delivery_fraction']:.0%}\n"
        f"Maximum IT: {p['max_injection_time_ms']:g} ms; actual IT is scenario input\n"
        f"Microscans: {p['microscans']}; assumed overhead: {p['overhead_ms']:g} ms\n"
        f"Rectangular signal width: {p['signal_width_ms']:g} ms (uncalibrated)\n\n"
        "Resolution sets transient length, not accumulation duration.\n"
        "Single origin is not adequate S/N or metabolite identification.\n"
        "Real MS detection efficiency remains uncalibrated.")
    if analysis.get('transport_pending'):text+='\n\nFLOW EDITED: previous transport fraction used.\nCFD and transit statistics need a new solve.'
    axes[1,1].text(0,1,text,va='top',fontsize=10,linespacing=1.6)
    for suffix in ('png','svg'):fig.savefig(root/f'acquisition_rates.{suffix}',dpi=150)
    plt.close(fig)


def export(analysis,directory):
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    AcquisitionParameters(**analysis['parameters']).save(root/'acquisition_parameters.json')
    (root/'acquisition_summary.json').write_text(json.dumps(clean(analysis),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(root/'acquisition_rates.csv',({k:v for k,v in row.items() if k!='expected_windows'} for row in analysis['rows']))
    write_csv(root/'expected_counts.csv',({'resolution_at_mz200':row['resolution_at_mz200'],
        'actual_injection_ms':row['actual_injection_ms'],**window} for row in analysis['rows'] for window in row['expected_windows']))
    figures(analysis,root)
    def number(value,digits=2):return ' Unknown ' if value is None else f'{value:.{digits}f}'
    rows=''.join('<tr>'+''.join(f'<td>{x}</td>' for x in [
        f"{r['resolution_at_mz200']:,}",number(r['actual_injection_ms']),number(r['transient_ms']),
        number(r['record_cycle_ms']),number(r['spectra_per_s']),
        ' No events ' if r['nonempty_single_cell_purity'] is None else number(100*r['nonempty_single_cell_purity'])+'%',
        number(100*r['mixed_spectrum_probability'])+'%',number(r['unique_pure_single_cell_events_s']),
        number(60*r['unique_pure_single_cell_events_s']),number(r['conditional_detected_unique_cells_s'])])+'</tr>' for r in analysis['rows'])
    sources=' · '.join(f'<a href="{html.escape(x["url"])}">{html.escape(x["title"])}</a>' for x in analysis['sources'])
    limits=''.join(f'<li>{html.escape(x)}</li>' for x in analysis['limitations'])
    p=analysis['parameters']
    pending='<p class="notice"> Updated flow preview :  Current flow with the previous transport fraction ,  Source  CFD,  Transit and finite-window statistics require an updated solution .</p>' if analysis.get('transport_pending') else ''
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Q Exactive  Acquisition and single-source rates </title><style>body{{font:16px/1.7 system-ui;max-width:1250px;margin:35px auto;padding:0 20px;color:#20354b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}img{{width:100%}}.notice{{background:#fff2db;padding:16px}}code{{background:#eef3f7}}</style>
<h1>Q Exactive:  Single-source rates from acquisition timing </h1>
{pending}
<p class="notice"> Method : Full MS; Maximum IT {p['max_injection_time_ms']:g} ms; Microscans {p['microscans']}. Actual  IT  Model settings ,  Overhead and packet width specified below ;  Detection calibration required . Default  30 ms / 1  Specified acquisition method ,  Actual injection time may differ by scan  30 ms.</p>
<p> Source : {html.escape(analysis['source'])}. Predicted outlet rate  {analysis['nozzle_cells_s']:.3f} cells/s;  Specified  {p['signal_delivery_fraction']:.1%}  Delivered to  C-trap  Cell-derived ion packets ,  Equivalent  {analysis['ion_packet_rate_s']:.3f}  Count /s;  Packet width  {p['signal_width_ms']:g} ms,  Default  0  Ideal impulse limit . Transport and signal delivery are separate factors . Identification efficiency : {number(p['single_event_identification_efficiency'])}.</p>
<h2> Metric definitions “ Single-source rate ”</h2><ul>
<li><b> Single-source scan purity </b>:  Among non-empty scans ,  Fraction containing a single source .</li>
<li><b> Unique single-source events /s</b>:  Distinct sources with at least one pure scan / Second ;  Each source counted once across scans .</li>
<li><b> Detection model /s</b>:  Unique-source rate multiplied by specified identification efficiency . Blank when efficiency is unspecified ,  Timing-based acquisition metric .</li></ul>
<table><tr><th> Resolution @200</th><th> Actual  IT ms</th><th> Transient  ms</th><th> Scan cycle  ms</th><th> Scan /s</th><th> Single-source purity of non-empty scans </th><th> Mixed fraction of all scans </th><th> Unique single-source /s</th><th> Unique single-source /min</th><th> Detection model /s</th></tr>{rows}</table>
<img src="acquisition_rates.png" alt="Theoretical acquisition occupancy and throughput">
<h2> Calculation method </h2><p> Original  Q Exactive  Four transient durations  64 / 128 / 256 / 512 ms. Parallel-acquisition approximation : <code>Tmicro=max(Ttransient,TactualIT)+Toverhead</code>;  Record cycle  <code>D=microscans×Tmicro+extra_delay</code>. Default  overhead=20 ms  Configurable value ,  Calibrate timing using  RAW  Timestamps / Actual  IT  Calibration .</p>
<p> Extend each accumulation gate backwards by the packet width ,  Union of the intervals  U,  Its length  L  Source onset times contributing to the scan .<code>μ=λsignal L</code>;  Empty-scan probability  <code>e⁻μ</code>,  Single-source scan probability  <code>μe⁻μ</code>,  Single-source purity of non-empty scans  <code>μ/(e^μ−1)</code>. Single-source scan rate  <code>μe⁻μ/D</code>.</p>
<p> For packets spanning multiple scans ,  Unique-source rate uses at least one pure scan per source  Poisson  Empty-interval probability :  Let  <code>a=L/D</code>, <code>Runique=λsignal e⁻μ [a−max(a−1,0)e^(−λsignal D)]</code>. Periodic gates and equal-width rectangular packets ;  Distinguishes missed ,  Mixed and duplicate scans . Window counts are steady-state expectations ,  Expected counts may be non-integer ,  Startup delay excluded .</p>
<h2> Parameters and data </h2><p><a href="acquisition_parameters.json"> Editable parameters  JSON</a> · <a href="acquisition_summary.json"> Results and source  CFD  Parameters </a> · <a href="acquisition_rates.csv"> Rate  CSV</a> · <a href="expected_counts.csv"> Expected counts by duration </a> · <a href="acquisition_rates.svg">SVG</a></p>
<h2> Applicability </h2><ul>{limits}</ul><h2> Timing references </h2><p>{sources}</p></html>'''
    (root/'acquisition_report.html').write_text(text,encoding='utf-8')
    return root
