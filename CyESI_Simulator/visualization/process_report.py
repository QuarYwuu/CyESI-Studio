"""Figures and animation from a shared continuous event stream."""
from pathlib import Path
import html
import json
import numpy as np
from .figures import serialized_plotting
from .process_data import STATUS_LABELS, STATUS_COLORS, CELL_COLORS


def _style():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    families={f.name for f in font_manager.fontManager.ttflist}
    font=next((s for s in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if s in families),'DejaVu Sans')
    return plt.rc_context({'font.family':font,'axes.unicode_minus':False,'font.size':10,'savefig.dpi':145})


def _color(event_id):return CELL_COLORS[int(event_id)%len(CELL_COLORS)]


def _neighbors(model,event_id,n=10):
    ids=model.records['event_id'];target=model.records['injection_s'][event_id]
    return sorted(ids[np.argsort(abs(model.records['injection_s']-target))[:n]].tolist())


def _focus_background(ax,model):
    f=model.data.fields;g=model.data.config.geometry
    values=np.ma.masked_where(~f['liquid_active'],f['methanol_fraction'])
    m=ax.pcolormesh(f['liquid_z_edges_m']*1e3,f['liquid_r_edges_m']*1e6,values,
        shading='flat',cmap='Blues',vmin=0,vmax=1,rasterized=True)
    ax.set(xlim=(-max(g.inner_recess_mm*1.5,.5),.03),ylim=(0,g.sheath_id_um*.5),
           xlabel='z (mm),  Sheath outlet  = 0',ylabel=' Radial  r (μm)')
    ax.axvline(g.inner_tip*1e3,color='#4c5966',ls='--',lw=.9)
    ax.text(g.inner_tip*1e3,g.sheath_id_um*.48,' Sample outlet ',ha='right',va='top',fontsize=9)
    ax.set_facecolor('#e5e5e5')
    return m


def _timeline(ax,model,center,span=2.,selected=None):
    a=center-span*.25;b=a+span;width=model.acquisition.signal_width_ms*.001
    for scan in model.scans:
        if scan['gates'][-1][1]<a or scan['gates'][0][0]>b:continue
        ids=scan['event_ids'];color='#148b72' if len(ids)==1 else '#d78a37' if ids else '#9aaebb'
        for left,right in scan['gates']:
            ax.broken_barh([(left,right-left)],(.72,.22),facecolors=color,alpha=.7)
        if ids:ax.text(scan['start_s'],1.0,str(len(ids)),fontsize=8,ha='left')
    for entry in model.ledger:
        t=entry['packet_onset_s']
        if t+width<a or t>b:continue
        y=.30 if entry['event_id']==selected else .12
        color=STATUS_COLORS[entry['status']]
        ax.plot([t,t+width],[y,y],color=color,lw=5 if width else 0)
        ax.plot(t,y,'o',color=color,ms=7 if entry['event_id']==selected else 4)
        ax.text(t,y-.13,f"#{entry['event_id']}",fontsize=8,rotation=35,ha='right')
    ax.set(xlim=(a,b),ylim=(-.20,1.16),yticks=[.2,.82],yticklabels=[' Cell-derived ion packets ',' Actual  IT  Accumulation gates '],xlabel=' Global time  (s)')
    ax.set_title(' Grey intervals :  No ions collected from this source ;  Green gates :  Single source ;  Orange gates :  Multiple sources ',fontsize=10)
    ax.axvline(center,color='#273b55',ls=':',lw=1)


@serialized_plotting
def export(model,directory,animation=True,progress=lambda s:None):
    import matplotlib.pyplot as plt
    from ..report.export import clean,write_csv
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    if model.default_event is None:tag=None
    else:tag=model.default_event
    with _style():
        progress(' Plot long-capillary spacing ,  Sedimentation and arrival timing ')
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');a,b,c,d=axes.flat
        g=model.data.config.geometry;length=g.sample_length_mm
        indices=np.unique(np.linspace(0,len(model.tracks)-1,min(24,len(model.tracks)),dtype=int))
        for i in indices:
            tr=model.tracks[i];keep=tr['z']<=g.inner_tip
            a.plot((tr['z'][keep]-model.start_z)*1e3,model.local_y[i][keep]*1e6,lw=.7,alpha=.6)
        clearance=(g.inner_id_um-model.data.config.cell.diameter_um)*.5
        a.axhline(clearance,color='#555',ls='--');a.axhline(-clearance,color='#555',ls='--')
        a.set(xlim=(0,length),ylim=(-g.inner_id_um*.55,g.inner_id_um*.55),xlabel=' Distance along sample capillary  (mm)',ylabel=' Cross-section projection  y (μm)',
              title=' Finite-sphere projection onto the gravity plane ;  Dashed line: accessible sphere-centre boundary ')
        if not model.has_bulk:a.text(.5,.5,' Long-capillary histories unavailable in this record ,  Upstream positions are interpolated ',transform=a.transAxes,ha='center',bbox=dict(facecolor='white'))
        if tag is not None:
            neighbors=_neighbors(model,tag,10);origin=model.records['injection_s'][tag]
            for eid in neighbors:
                index=model.records['template_id'][eid];times=model.records['injection_s'][eid]+model.crossing[index]-origin
                b.plot((model.stage_z-model.start_z)*1e3,times,'o-',ms=3,lw=2 if eid==tag else .8,color=_color(eid))
                if eid==tag:b.annotate(f'#{eid}',(length,times[-1]),fontsize=8)
            b.set(xlabel=' Axial position  (mm)',ylabel=f' Relative  #{tag}  Injection times  (s)',title=' Shared cell IDs :  Arrival time at each section  ( Crossing lines indicate an order exchange )')
            ordered=sorted(neighbors,key=lambda i:model.records['injection_s'][i])
            pairs=list(zip(ordered[:-1],ordered[1:]));x=np.arange(len(pairs))
            inlet=[model.records['injection_s'][j]-model.records['injection_s'][i] for i,j in pairs]
            outlet=[model.records['nozzle_s'][j]-model.records['nozzle_s'][i] for i,j in pairs]
            c.bar(x-.18,inlet,.36,label=' Inlet  Δt',color='#8bb5d5');c.bar(x+.18,outlet,.36,label=' Outlet timing for the same cell pair  Δt',color='#efab69')
            c.axhline(0,color='#777',lw=.8);c.legend(fontsize=8)
            c.set(xticks=x,xticklabels=[f'#{i}→#{j}' for i,j in pairs],ylabel=' Pairwise arrival interval  (s)',title=' Evolution of adjacent-cell arrival intervals ;  Negative intervals indicate an exit-order exchange ')
            c.tick_params(axis='x',labelrotation=40)
        initial=np.array([tr['r'][0] for tr in model.tracks])*1e6
        end=np.array([tr['r'][-1] for tr in model.tracks])[model.exited]*1e6
        bins=np.linspace(0,max(initial.max(),end.max() if len(end) else 0)*1.05+1e-9,17)
        d.hist(initial,bins=bins,alpha=.55,label=' Inlet  r');d.hist(end,bins=bins,alpha=.55,label=' Sheath outlet  r')
        d.set(xlabel=' Radial position  r (μm)',ylabel=' Saved response trajectories ',title=' Radial focusing and axial arrival intervals ');d.legend()
        fig.suptitle(f'01  Cell stream in the long capillary  | {length:g} mm; Q = {model.data.config.fluid.sample_flow_ul_min:.4g} μL/min',fontsize=16)
        fig.savefig(root/'process_01_long_tube.png');plt.close(fig)

        progress(' Plot outlet focusing and solvent exposure ')
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');a,b,c,d=axes.flat
        mesh=_focus_background(a,model);fig.colorbar(mesh,ax=a,label=' Methanol volume fraction ')
        for i in indices:
            tr=model.tracks[i];keep=tr['z']>=-max(g.inner_recess_mm*1.5,.5)*1e-3
            color=_color(i)
            a.plot(tr['z'][keep]*1e3,tr['r'][keep]*1e6,color=color,lw=.9)
            inner_r=float(np.interp(tr['inner_time'],tr['t'],tr['r']))
            if model.exited[i]:b.plot([0,1],[inner_r*1e6,tr['r'][-1]*1e6],'o-',color=color,ms=3,alpha=.65)
            exposure=tr['z']>=g.inner_tip-1e-4
            c.plot(tr['z'][exposure]*1e3,tr['C'][exposure]*100,lw=.8,alpha=.7)
            keep=tr['z']>=g.inner_tip
            d.plot((tr['t'][keep]-tr['inner_time'])*1e3,tr['r'][keep]*1e6,lw=.8,alpha=.7)
        a.set_title(' Computed methanol field  +  Shared finite-sphere trajectory ;  Grey: quartz ')
        b.set(xticks=[0,1],xticklabels=[' Sample outlet ',' Sheath outlet '],ylabel=' Radial position  r (μm)',title=' Outlet focusing contribution :  Radial positions at both outlets along the same trajectory ')
        c.axvline(g.inner_tip*1e3,color='#777',ls='--',lw=.8)
        c.set(xlabel='z (mm),  Dashed line: sample outlet ',ylabel=' External methanol  (%)',ylim=(0,100),title=' Solvent exchange from the sample capillary to the outlet mixing region ')
        d.set(xlabel=' Physical time since the sample outlet  (ms)',ylabel=' Radial  r (μm)',title=' Local focusing timescale ')
        fig.suptitle('02  Focusing region detail  |  Radial focusing and longitudinal spacing are distinct observables ',fontsize=16)
        fig.savefig(root/'process_02_focusing.png');plt.close(fig)

        progress(' Plot acquisition gates ,  Complete  /  Partial signals and unique-source recovery ')
        fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained',gridspec_kw={'height_ratios':[1.4,1,1.1]});a,b,c=axes
        center=model.by_id[tag]['packet_onset_s'] if tag in model.by_id else model.parameters.duration_s*.5
        _timeline(a,model,center,max(2.,model.row['record_cycle_ms']*.008,model.acquisition.signal_width_ms*.004),tag)
        for row in model.ledger:
            t=row['nozzle_s'];status=row['status'];j=list(STATUS_LABELS).index(status)
            b.plot(t,j,'o',ms=4,color=STATUS_COLORS[status])
        b.set(xlim=(0,model.parameters.duration_s),yticks=range(len(STATUS_LABELS)),yticklabels=list(STATUS_LABELS.values()),xlabel=' Sheath-outlet crossing time  (s)',title=' Unique outcomes across the outlet cohort :  Each marker represents one cell source ')
        summary=model.summary();counts=summary['counts'];statuses=[s for s in STATUS_LABELS if counts[s]]
        if statuses:
            bars=c.barh(range(len(statuses)),[counts[s] for s in statuses],color=[STATUS_COLORS[s] for s in statuses])
            c.set(yticks=range(len(statuses)),yticklabels=[STATUS_LABELS[s] for s in statuses],xlabel=' Distinct cell sources  ( Counts unique sources )')
            for bar,s in zip(bars,statuses):c.text(bar.get_width()+.1,bar.get_y()+bar.get_height()/2,f'{counts[s]} / {len(model.ledger)}',va='center')
            c.set_xlim(0,max(counts.values())*1.25+1)
        else:c.text(.5,.5,' No outlet events in this observation window ;  Extend the observation window or inspect cells in transit .',ha='center',transform=c.transAxes)
        width=model.acquisition.signal_width_ms
        fig.suptitle(f'03  MS acquisition opportunity  | R={model.parameters.resolution:,}; IT={model.parameters.actual_it_ms:g} ms;  Packet width ={width:g} ms',fontsize=15)
        c.set_title(' Denominator :  Outlet cells in the observation window ;  Delivery fraction is a scenario parameter .0 ms  Signal completeness is undefined for impulse packets .',fontsize=10)
        fig.savefig(root/'process_03_acquisition_recovery.png');plt.close(fig)

        _flight_figure(model,root)
        if animation and tag is not None:
            progress(' Linked long-capillary history of the same cell  →  Focusing  →  Acquisition gate playback ')
            _movie(model,tag,root/'process_movie.gif')
    write_csv(root/'cell_event_ledger.csv',model.ledger)
    write_csv(root/'source_stream.csv',(dict(event_id=int(i),injection_s=float(a),transport_sample_id=int(model.data.ids[k]),
        nozzle_s=float(b),transport_exited=bool(model.exited[k])) for i,a,k,b in zip(*[model.records[s] for s in ('event_id','injection_s','template_id','nozzle_s')])))
    write_csv(root/'accumulation_gates.csv',(dict(spectrum_id=scan['index'],gate_start_s=a,gate_end_s=b,distinct_origins=len(scan['event_ids']),
        event_ids=';'.join(map(str,scan['event_ids']))) for scan in model.scans for a,b in scan['gates']))
    # CSV arrival matrices make every crossing in the spacetime diagram auditable.
    write_csv(root/'stage_arrivals.csv',(dict(event_id=int(i),**{name+'_s':float(a+t) for name,t in zip(model.stage_names,model.crossing[k])})
        for i,a,k in zip(*[model.records[s] for s in ('event_id','injection_s','template_id')])))
    (root/'process_summary.json').write_text(json.dumps(clean(model.summary()),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (root/'theoretical_acquisition.json').write_text(json.dumps(clean(model.theory),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if model.flight is not None:
        (root/'selected_carrier_state.json').write_text(json.dumps(clean(model.flight),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(root/'selected_carrier_flight.csv',model.flight['history'])
    _html(model,root,animation and tag is not None)
    return root


def _flight_figure(model,root):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained');a,b,c,d=axes.flat
    flight=model.flight;gap=model.data.config.geometry.ms_gap_mm
    a.axvline(gap,color='#5f6870',lw=3);a.set(xlim=(0,gap*1.04),xlabel=' Distance from sheath outlet  z (mm)',ylabel=' Radial  r (μm)',title=' Cell-carrying droplets downstream of the emitter :  Trajectories from the selected transfer model ')
    if flight is None:
        for ax in axes.flat:ax.text(.5,.5,' Selected-cell inlet transfer has not been computed \n Inlet capture and biological integrity require calibration ',ha='center',va='center',transform=ax.transAxes)
    else:
        result=flight['result'];history=flight['history'];a.set_title(f" Cell  #{flight['event_id']};  Final state : {result['status']}",fontsize=10)
        if history:
            t=np.array([h['time_after_lip_s'] for h in history])*1e3
            a.plot([h['z_m']*1e3 for h in history],[h['r_m']*1e6 for h in history],color='#176bba')
            a.plot(history[-1]['z_m']*1e3,history[-1]['r_m']*1e6,'o',color='#d45c65')
            b.plot(t,[h['drop_radius_m']*1e6 for h in history]);b.set(xlabel=' Time since exit  (ms)',ylabel=' Carrier-droplet radius  (μm)',title=' Droplet size after evaporation ;  Breakup beyond the resolved domain is excluded ')
            c.plot(t,[h['intracellular_methanol_fraction']*100 for h in history],label=' Intracellular methanol ')
            c.plot(t,[h['external_methanol_fraction']*100 for h in history],label=' External liquid-film methanol ');c.legend()
            c.set(xlabel=' Time since exit  (ms)',ylabel=' Volume fraction  (%)',ylim=(0,100),title=' Solvent diffusion and membrane exchange ')
            d.plot(t,[h['membrane_lipid_remaining_fraction']*100 for h in history],label=' Remaining membrane lipid ')
            d.plot(t,[h['intracellular_tracer_fraction']*100 for h in history],label=' Intracellular tracer ');d.legend()
            d.set(xlabel=' Time since exit  (ms)',ylabel=' Remaining fraction  (%)',ylim=(0,100),title=' Membrane model scenario ;  Biological viability is a separate metric ')
        else:a.text(.04,.5,' Scenario outside the electrical load or membrane model bounds \n No resolved downstream trajectory .',transform=a.transAxes)
    fig.suptitle('04  Separate inlet-state and acquisition calculations  |  Geometric inlet passage  ≠  Ion-packet collection  ≠  Metabolite detection ',fontsize=15)
    fig.savefig(root/'process_04_inlet_state.png');plt.close(fig)


def _movie(model,tag,path):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained',gridspec_kw={'height_ratios':[1,1.3,1.1]});a,b,c=axes
    g=model.data.config.geometry;length=g.sample_length_mm
    a.set(xlim=(0,length),ylim=(-g.inner_id_um*.6,g.inner_id_um*.6),xlabel=' Axial distance in the capillary  (mm)',ylabel=' Projection  y (μm)')
    for bound in [-1,1]:a.axhline(bound*(g.inner_id_um-model.data.config.cell.diameter_um)*.5,color='#777',ls='--')
    for i in np.unique(np.linspace(0,len(model.tracks)-1,min(12,len(model.tracks)),dtype=int)):
        tr=model.tracks[i];keep=tr['z']<=g.inner_tip
        a.plot((tr['z'][keep]-model.start_z)*1e3,model.local_y[i][keep]*1e6,color='#b9c9d8',lw=.5)
    _focus_background(b,model)
    for i in np.unique(np.linspace(0,len(model.tracks)-1,min(12,len(model.tracks)),dtype=int)):
        tr=model.tracks[i];b.plot(tr['z']*1e3,tr['r']*1e6,color='#a5b6c7',lw=.6)
    center=model.by_id[tag]['packet_onset_s'] if tag in model.by_id else 0.
    _timeline(c,model,center,max(2.,model.row['record_cycle_ms']*.008),tag)
    s1=a.scatter([],[],s=22,zorder=5);s2=b.scatter([],[],s=28,zorder=5)
    marker=c.axvline(center,color='#c52d4a',lw=2);title=fig.suptitle('',fontsize=12)
    def frame(j):
        p=j/79;age=model.event_age(tag,p);time=model.global_time(tag,p);positions=model.positions(time)
        bulk=[x for x in positions if x['z_m']<=g.inner_tip];tip=[x for x in positions if x['z_m']>=b.get_xlim()[0]*1e-3]
        for scatter,points,keys in [(s1,bulk,('z_m','y_m')),(s2,tip,('z_m','r_m'))]:
            xy=np.array([[(x[keys[0]]-model.start_z if scatter is s1 else x[keys[0]])*1e3,x[keys[1]]*1e6] for x in points]).reshape(-1,2)
            scatter.set_offsets(xy);scatter.set_color([_color(x['event_id']) for x in points])
            scatter.set_sizes([100 if x['event_id']==tag else 20 for x in points])
        marker.set_xdata([time,time]);title.set_text(f' Track  #{tag} |  Since capillary entry  {age:.5f} s |  Global time  {time:.5f} s\n Playback speed varies by transport stage ;  Cell markers enlarged for visibility ,  Computed coordinates and event times ')
        return s1,s2,marker,title
    FuncAnimation(fig,frame,frames=80,interval=100,blit=False).save(path,writer=PillowWriter(fps=8),dpi=85)
    plt.close(fig)


def _html(model,root,animation):
    summary=model.summary();n=summary['nozzle_cohort_count'];pure=summary['pure_unique_count']
    images=['process_01_long_tube.png','process_02_focusing.png','process_03_acquisition_recovery.png','process_04_inlet_state.png']
    rows=''.join(f'<tr><td>{STATUS_LABELS[s]}</td><td>{count}</td></tr>' for s,count in summary['counts'].items())
    notes=''.join(f'<li>{html.escape(note)}</li>' for note in summary['notes'])
    movie='<img src="process_movie.gif" alt=" Continuous playback of the selected cell ">' if animation else ''
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CyESI  Continuous cell transport </title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#213448;max-width:1280px;margin:30px auto;padding:0 24px}}img{{width:100%;background:white;margin:18px 0;border-radius:10px}}section{{background:white;padding:20px;margin:18px 0;border-radius:10px}}td,th{{padding:6px 28px 6px 0;text-align:left}}</style>
<h1> Capillary-to-MS transport :  Linked transport histories for one cell cohort </h1>
<p> Source solution : {html.escape(model.data.source)}<br> Continuous source  {model.source_rate:.4g} cells/s;  This playback reuses  {len(model.tracks)}  Saved trajectories ;  Population response sample  {model.data.config.cell.count}  Entries .</p>
<section><b> Reading the results </b><p> Colours and IDs follow individual cells . Sedimentation and arrivals at successive capillary sections ,  Radial contraction at the outlet ;  Compare cell-derived ion packets with the measured  IT  Accumulation gate comparison . Long-capillary transport does not impose  Poisson  source into equally spaced events .</p>
<p> This window  {n}  outlet sources , {pure}  sources with at least one single-source spectrum . Counts from one stochastic realization ,  Acquisition opportunity uses a separate denominator from detection efficiency . Complete signal: full collection of the specified rectangular ion packet ;  Geometric passage is distinct from viability and MS identification .0 ms  Impulse limit ,  Signal completeness undefined .</p>
<table><tr><th> Mutually exclusive outcomes </th><th> Unique cell sources </th></tr>{rows}</table>
<p> Denominator:  [0, {model.parameters.duration_s:g}) s  cells crossing the sheath outlet within . Background sources contribute to mixed-packet classification ,  Packets crossing the window end remain tracked . Unresolved transport and inlet states remain censored . Outlet-to-packet delivery and delay are editable scenario parameters ;  Measured detection rate ,  Biological integrity at the inlet requires calibration .</p></section>
{movie}{''.join(f'<img src="{name}" alt="{name}">' for name in images)}
<section><b> Source computation and boundaries </b><p><a href="cell_event_ledger.csv"> Per-cell acquisition ledger </a> · <a href="source_stream.csv"> Continuous source </a> · <a href="stage_arrivals.csv"> Section arrival time </a> · <a href="accumulation_gates.csv"> Accumulation gates used </a> · <a href="process_summary.json"> Parameters and metric denominators </a> · <a href="theoretical_acquisition.json"> Population-wide theoretical acquisition rate </a></p><ul>{notes}</ul></section></html>'''
    (root/'process_report.html').write_text(text,encoding='utf-8')
