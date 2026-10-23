"""Cell-diameter plots using common axes and saved solver arrays."""
from pathlib import Path
import numpy as np
from .figures import serialized_plotting
from .process_report import _style

COLORS=['#2673b8','#d18a29','#8558a4','#497b4a']
MARKERS=['o','s','^','D']
LINESTYLES=['-','--','-.',':']
FILENAMES=['size_01_overview','size_02_long_tube','size_03_focusing','size_04_transit_exposure',
           'size_05_acquisition','size_06_conditional_inlet']


def _fmt(value,digits=3):return ' Unknown ' if value is None else f'{value:.{digits}g}'


def _box(ax,arrays,labels,colors,ylabel):
    finite=[np.asarray(a)[np.isfinite(a)] for a in arrays]
    artists=ax.boxplot(finite,positions=np.arange(len(arrays)),widths=.48,patch_artist=True,
        showfliers=False,whis=(5,95),medianprops={'color':'#1d2938','linewidth':1.5})
    for patch,color in zip(artists['boxes'],colors):patch.set(facecolor=color,alpha=.45)
    ax.set(xticks=range(len(labels)),xticklabels=labels,xlabel=' Equivalent sphere diameter  (μm)',ylabel=ylabel)
    ax.grid(axis='y',color='#e4e9ee',lw=.6);ax.set_axisbelow(True)


def _field(ax,case,zmin):
    data=case['data'];f=data.fields;g=case['config'].geometry
    field=np.ma.masked_where(~f['liquid_active'],f['methanol_fraction'])
    m=ax.pcolormesh(f['liquid_z_edges_m']*1e3,f['liquid_r_edges_m']*1e6,field,
        cmap='Blues',vmin=0,vmax=1,shading='flat',rasterized=True)
    ax.set(xlim=(zmin,0),ylim=(0,g.c*1e6),xlabel=' Axial  z (mm),  Sheath outlet =0',ylabel=' Centre radius  r (μm)',facecolor='#e4e4e4')
    ax.axvline(g.inner_tip*1e3,color='#6b7785',ls='--',lw=.9)
    return m


@serialized_plotting
def render(result,directory,progress=lambda s:None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Circle
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    cases=result['cases'];rows=result['rows'];diameters=[r['diameter_um'] for r in rows]
    labels=[f'{x:g}' for x in diameters];colors=[COLORS[i%len(COLORS)] for i in range(len(cases))]
    particles=[case['particles'] for case in cases];g=result['config'].geometry
    config=result['config'];n=result['parameters']['cells']
    subtitle=(f' Matched device :  Sample capillary  {g.inner_id_um:g} μm / {g.sample_length_mm:g} mm; {config.supply.orientation}; '
        f'Q={config.fluid.sample_flow_ul_min:.4g} μL/min; C={config.cell.concentration_per_ml:g}/mL;  Per group  n={n}')
    with _style(),PdfPages(root/'cell_size_comparison.pdf') as pdf:
        def save(fig,name):
            progress('Saving '+name)
            fig.savefig(root/(name+'.png'));fig.savefig(root/(name+'.svg'));pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained');a,b,c,d,e,f,h,j=axes.flat
        x=np.arange(len(rows));fraction=np.array([r['transport_fraction'] for r in rows])*100
        low=np.array([r['transport_sampling_ci95_low'] for r in rows])*100;high=np.array([r['transport_sampling_ci95_high'] for r in rows])*100
        a.bar(x,fraction,color=colors,alpha=.85);a.errorbar(x,fraction,yerr=[fraction-low,high-fraction],fmt='none',color='#283747',capsize=4)
        a.set(ylim=(0,110),ylabel=' Exited  /  All responses  (%)',title=' Model outlet fraction  + 95%  Sampling interval ',xticks=x,xticklabels=labels,xlabel=' Diameter  (μm)')
        for k,value in enumerate(fraction):a.text(k,min(106,value+3),f'{value:.1f}%',ha='center',fontsize=9)
        _box(b,[p.residence_time[p.exited] for p in particles],labels,colors,' Transit time  (s)');b.set_title(' Median among exiting particles ,  Quartiles and  P5–P95')
        _box(c,[p.stress['near_wall_time_s'] for p in particles],labels,colors,' Capillary near-wall duration  (s)');c.set_title(' Near-wall residence ,  All-response denominator ')
        r95=np.array([np.nan if r['outlet_r95_um'] is None else r['outlet_r95_um'] for r in rows])
        outer=np.array([np.nan if r['outlet_cell_envelope_r95_um'] is None else r['outlet_cell_envelope_r95_um'] for r in rows])
        d.plot(x,r95,'o-',color='#2673b8',label=' Sphere centre  r95');d.plot(x,outer,'s--',color='#d18a29',label='r95 +  Cell radius ')
        d.set(xticks=x,xticklabels=labels,ylabel=' Radial width  (μm)',xlabel=' Diameter  (μm)',title=' Centre and outer-radius focusing metrics ');d.legend(fontsize=8);d.set_ylim(bottom=0)
        _box(e,[p.outlet_methanol[p.exited]*100 for p in particles],labels,colors,' External methanol at outlet  (%)');e.set_title(' Outlet solvent composition ,  Conditioned on exit ');e.set_ylim(0,100)
        _box(f,[p.methanol_dose[p.exited]*1000 for p in particles],labels,colors,'∫ C Methanol  dt (ms)');f.set_title(' Equivalent pure-methanol exposure time ')
        _box(h,[p.stress['shear_dose_pa_s'][p.exited] for p in particles],labels,colors,' Shear dose  (Pa·s)');h.set_title(' Integrated stress exposure ,  Conditioned on exit ')
        for k,(case,row,color) in enumerate(zip(cases,rows,colors)):
            # Same inlet bore and true cell diameter for every category.
            center=k*65
            j.add_patch(Circle((center,0),g.inner_id_um/2,fill=False,edgecolor='#8695a3',lw=1.2))
            j.add_patch(Circle((center,0),row['diameter_um']/2,facecolor=color,alpha=.85))
            j.text(center,-34,f'{row["diameter_um"]:g} μm',ha='center',fontsize=9)
        j.set(xlim=(-33,(len(rows)-1)*65+33),ylim=(-45,35),aspect='equal',title=f' Physical size reference :  Grey circle  {g.inner_id_um:g} μm  Inner diameter ');j.axis('off')
        fig.suptitle(' Cell  /  Diameter effects on transport ,  Focusing ,  Exposure comparison \n'+subtitle,fontsize=14)
        fig.supxlabel(' Equivalent rigid-sphere model ; 0.5 / 2 μm  Brownian motion included .100%  Model outlet fraction ,  Sampling intervals exclude model discrepancy .',fontsize=10)
        save(fig,FILENAMES[0])

        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
        from ..fluid.supply import segments
        vectors=[np.array([s['gx'],s['gy']]) for s in segments(config) if np.hypot(s['gx'],s['gy'])>1e-9]
        direction=vectors[0]/np.linalg.norm(vectors[0]) if vectors else np.array([1.,0.])
        for ax,case,row,color in zip(axes.flat,cases,rows,colors):
            bulk=case['particles'].bulk_history;radius=row['diameter_um']/2
            if bulk is not None:
                projected=-(bulk['x_m']*direction[0]+bulk['y_m']*direction[1])*1e6
                for i in range(0,len(bulk['ids']),max(1,len(bulk['ids'])//24)):
                    ax.plot(bulk['distance_m'][:,i]*1e3,projected[:,i],color=color,lw=.75,alpha=.65)
            clearance=g.inner_id_um/2-radius-config.supply.wall_capture_gap_um
            for sign in [-1,1]:ax.axhline(sign*g.inner_id_um/2,color='#5a6774',lw=1.4);ax.axhline(sign*clearance,color='#8c939b',ls='--',lw=.85)
            ax.set(xlim=(0,g.sample_length_mm),ylim=(-g.inner_id_um*.6,g.inner_id_um*.6),
                xlabel=' Distance along capillary  (mm)',ylabel=' Projection in the gravity plane  (μm)',title=f'{row["diameter_um"]:g} μm |  Brownian motion  {" On " if row["brownian_enabled"] else " Off "}')
            ax.text(.02,.93,f' Near-wall fraction  {row["near_wall_exposed_fraction"]:.1%}\n Median transit  {_fmt(row["residence_p50_s"])} s',transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        fig.suptitle(' Capillary settling ,  Diffusion and finite-size boundaries \n Solid: capillary wall ,  Dashed: accessible sphere-centre boundary ;  Downward: transverse gravity direction ',fontsize=14)
        save(fig,FILENAMES[1])

        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained',sharex=True,sharey=True)
        zmin=-max(1.5*g.inner_recess_mm,.5)
        for ax,case,row,color in zip(axes.flat,cases,rows,colors):
            mesh=_field(ax,case,zmin);data=case['data'];p=case['particles']
            for index in range(0,len(data.ids),max(1,len(data.ids)//24)):
                tr=data.track(index);keep=tr['z']>=zmin*.001
                if np.sum(keep)<2:continue
                ax.plot(tr['z'][keep]*1e3,tr['r'][keep]*1e6,color=color,lw=.85,alpha=.7)
                if not p.exited[data.ids[index]]:
                    ax.plot(tr['z'][-1]*1e3,tr['r'][-1]*1e6,'x',color='#3f3f3f',ms=5)
            ax.set_title(f'{row["diameter_um"]:g} μm |  Sphere centre  r95={_fmt(row["outlet_r95_um"])} μm;  Outer radius  {_fmt(row["outlet_cell_envelope_r95_um"])} μm',fontsize=11)
        fig.colorbar(mesh,ax=list(axes.flat),label=' Methanol volume fraction ',shrink=.9)
        fig.suptitle(' Matched methanol  /  Sphere diameters in the same flow field :  Outlet focusing and wall capture \n Grey: quartz ; ×  Non-exiting trajectory endpoints ;  Coordinates ,  Shared concentration scale ',fontsize=14)
        save(fig,FILENAMES[2])

        fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');a,b,c,d=axes.flat
        time_end=max((float(np.max(p.residence_time[p.exited])) for p in particles if p.exited.any()),default=1.)*1.04
        dose_end=max((float(np.max(p.methanol_dose[p.exited]*1000)) for p in particles if p.exited.any()),default=1.)*1.04
        for k,(case,row,color) in enumerate(zip(cases,rows,colors)):
            p=case['particles'];ok=p.exited;label=f'{row["diameter_um"]:g} μm'
            if ok.any():
                t=np.sort(p.residence_time[ok]);a.step(np.r_[0,t,time_end],np.r_[0,np.arange(1,len(t)+1)/len(ok)*100,len(t)/len(ok)*100],where='post',color=color,ls=LINESTYLES[k],label=label)
                dose=np.sort(p.methanol_dose[ok]*1e3);b.step(np.r_[dose,dose_end],np.r_[np.arange(1,len(dose)+1)/len(dose)*100,100],where='post',color=color,ls=LINESTYLES[k],label=label)
                data=case['data'];zz=np.linspace(g.inner_tip-.1e-3,0,160);curves=[]
                for index,i in enumerate(data.ids):
                    if not p.exited[i]:continue
                    tr=data.track(index);unique=np.r_[np.diff(np.maximum.accumulate(tr['z']))>1e-14,True]
                    curves.append(np.interp(zz,tr['z'][unique],tr['C'][unique])*100)
                if curves:
                    q=np.quantile(curves,[.1,.5,.9],axis=0)
                    c.plot(zz*1e3,q[1],color=color,ls=LINESTYLES[k],label=label);c.fill_between(zz*1e3,q[0],q[2],color=color,alpha=.12)
            values=[row['transport_fraction']*100,row['wall_stopped']/len(ok)*100,row['censored_or_reverse']/len(ok)*100]
            bottom=0
            for value,gray,hatch in zip(values,['#7fa7c6','#e4ad6b','#b8b8c5'],['','///','xx']):
                d.bar(k,value,bottom=bottom,color=gray,hatch=hatch,edgecolor='#687885',lw=.5);bottom+=value
        a.axvline(np.pi*g.a*g.a*g.sample_length_mm*.001/config.fluid.qs,color='#555',ls='--',lw=.8,label=' Liquid volume  / Q')
        a.set(xlabel=' Time since capillary entry  (s)',ylabel=' Exited capillary  /  All responses  (%)',ylim=(0,103),title=' Cumulative exits :  All-response denominator ');a.legend(fontsize=9)
        b.set(xlabel=' Equivalent methanol exposure  ∫C dt (ms)',ylabel=' Cumulative fraction among exits  (%)',title=' Exposure distribution among exits ');b.legend(fontsize=9)
        c.set(xlabel=' Axial  z (mm),  Sheath outlet =0',ylabel=' External methanol  (%)',ylim=(0,100),title=' Saved exiting trajectories :  Median and  P10–P90 ( Distribution interval )');c.legend(fontsize=9)
        from matplotlib.patches import Patch
        d.set(xticks=np.arange(len(labels)),xticklabels=labels,xlabel=' Diameter  (μm)',ylabel=' All responses  (%)',ylim=(0,104),title=' Exclusive terminal states ;  Time-censored trajectories retained separately ')
        d.legend(handles=[Patch(facecolor=col,hatch=ha,label=la) for col,ha,la in zip(['#7fa7c6','#e4ad6b','#b8b8c5'],['','///','xx'],[' Exited ',' Wall capture ',' Censored  /  Reverse flow '])],fontsize=9)
        fig.suptitle(' Transit-time and exposure distributions \n'+subtitle,fontsize=14)
        save(fig,FILENAMES[3])

        ap=result['acquisition_parameters'];its=ap['actual_injection_times_ms'];resolutions=[17500,35000,70000,140000]
        fig,axes=plt.subplots(1,len(its),figsize=(max(10,5*len(its)),6),layout='constrained',squeeze=False)
        matrices=[]
        for it in its:
            matrices.append(np.array([[next(v['unique_pure_single_cell_events_s']*60 for v in result['acquisition_rows']
                if v['diameter_um']==diameter and v['actual_injection_ms']==it and v['resolution_at_mz200']==resolution)
                for resolution in resolutions] for diameter in diameters]))
        maximum=max(float(v.max()) for v in matrices) or 1.
        for ax,it,matrix in zip(axes.flat,its,matrices):
            mesh=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=maximum,aspect='auto')
            ax.set(xticks=range(4),xticklabels=['17,500','35,000','70,000','140,000'],yticks=range(len(labels)),
                yticklabels=labels,xlabel=' Resolution  @ m/z 200',ylabel=' Diameter  (μm)',title=f' Actual  IT = {it:g} ms')
            for (r,col),value in np.ndenumerate(matrix):ax.text(col,r,f'{value:.2f}',ha='center',va='center',color='white' if value>maximum*.58 else '#203448',fontsize=11)
        fig.colorbar(mesh,ax=list(axes.flat),label=' Theoretical unique-source events  / min',shrink=.7)
        fig.suptitle(f'Q Exactive Full MS:  Diameter and resolution comparison \n Shared packet width  {ap["signal_width_ms"]:g} ms,  Delivery fraction  {ap["signal_delivery_fraction"]:.0%}, Microscans={ap["microscans"]};  Detection calibration pending ',fontsize=13)
        save(fig,FILENAMES[4])

        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        metric_specs=[('intracellular_methanol_fraction',' Intracellular methanol  (%)',100),
            ('membrane_lipid_remaining_fraction',' Remaining membrane lipid  (%)',100),('intracellular_tracer_fraction',' Intracellular tracer  (%)',100)]
        for ax,(key,title,factor) in zip(axes.flat,metric_specs):
            before=[np.nan if not case['inlet'] or not case['inlet'].get('at_lip') else case['inlet']['at_lip'][key]*factor for case in cases]
            after=[np.nan if not case['inlet'] or not case['inlet'].get('arrived') else case['inlet'][key]*factor for case in cases]
            ax.plot(x,before,'o-',color='#2673b8',label=' At sheath outlet ');ax.plot(x,after,'s--',color='#d18a29',label=' Carrier inlet state ')
            ax.set(xticks=x,xticklabels=labels,xlabel=' Diameter  (μm)',ylim=(0,100),ylabel=title);ax.legend(fontsize=9)
        a,b,c=axes[1]
        flight=[np.nan if not case['inlet'] or not case['inlet'].get('arrived') else case['inlet']['flight_time_ms'] for case in cases]
        a.bar(x,flight,color=colors);a.set(xticks=x,xticklabels=labels,ylabel=' Sheath outlet to aperture  (ms)',xlabel=' Diameter  (μm)',title=' Carrier transfer time ;  Missing: unresolved  /  No aperture entry ')
        temp=[np.nan if not case['inlet'] or not case['inlet'].get('arrived') else case['inlet']['final_temperature_c'] for case in cases]
        b.plot(x,temp,'o-',color='#497b4a');b.set(xticks=x,xticklabels=labels,ylabel=' Droplet temperature at entry  (°C)',xlabel=' Diameter  (μm)',title=' Wall temperature  320°C  Separate particle thermal balance  /  Droplets ')
        c.axis('off');states=[]
        status_map={'entered_aperture':' Carrier aperture entry ','infeasible_electrical_load':' Electrical load incompatible ','pre_nozzle_shell_model_lost':' Membrane model ended before emitter ',
            'membrane_continuum_lost':' Membrane continuum limit ','rayleigh_instability_unresolved':' Droplet instability ,  Pending continuation ','no_saved_exit':' No saved exit trajectory '}
        for row,case in zip(rows,cases):
            status=(case['inlet'] or {}).get('status',' Not computed ');states.append(f'{row["diameter_um"]:g} μm: {status_map.get(status,status)}')
        c.text(0,.95,'\n\n'.join(states)+'\n\n Selected per group  1  Saved trajectories near median transit time .\n Representative-trajectory transfer result .',va='top',fontsize=11)
        fig.suptitle(' Lipid sphere model  /  Cell-state comparison in carrier droplets \n Shared carrier radius  20 μm, q/qR=0.15,  Membrane permeability  1e-6 m/s,  Dissolution  k=20/s,  Aspiration multiplier  1',fontsize=13)
        save(fig,FILENAMES[5])
        if result.get('brownian_control_directory') and 2 in diameters:
            from ..optimization.size_study import load_data
            control=load_data(root/result['brownian_control_directory'])
            old=next(c for c in control['cases'] if c['config'].cell.diameter_um==2)
            current=next(c for c in cases if c['config'].cell.diameter_um==2)
            _brownian_control(plt,old,current,save)
    _readme(result,root)
    return root


def _readme(result,root):
    config=result['config'];ap=result['acquisition_parameters']
    lines=['# 0.5 / 2 / 8 / 20 μm  Equivalent-sphere comparison ','',
        f' All sizes denote diameter . Current configuration :  Sample capillary  {config.geometry.sample_length_mm:g} mm, {config.supply.orientation},  Sample  {config.supply.sample_control}, {config.supply.sample_pressure_mbar:g} mbar,  Actual  Q={config.fluid.sample_flow_ul_min:.6g} μL/min,  Sheath  {config.fluid.sheath_flow_ul_min:g} μL/min,  Concentration  {config.cell.concentration_per_ml:g} cells/mL,  Voltage  {config.electrical.voltage_kv:g} kV.',
        f' Per group  {result["parameters"]["cells"]}  Responses ;  Study-specific Brownian threshold  {config.cell.brownian_below_um:g} μm ( Default threshold retained elsewhere ). Density , Zeta,  Matched concentration and supply .', '',
        '|  Diameter  μm |  Exited  /  Response  |  Median transit  s | r95 μm |  Outer radius  r95+R μm |  Outlet methanol  % |  Near-wall fraction  % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        lines.append(f'| {row["diameter_um"]:g} | {row["exited"]}/{row["samples"]} | {_fmt(row["residence_p50_s"],5)} | {_fmt(row["outlet_r95_um"],5)} | {_fmt(row["outlet_cell_envelope_r95_um"],5)} | {_fmt(None if row["mean_outlet_methanol_fraction"] is None else row["mean_outlet_methanol_fraction"]*100,5)} | {row["near_wall_exposed_fraction"]*100:.2f} |')
    lines+=['',' Figure files : ','']+[f'- [{name}.png]({name}.png) ( Additional  SVG)' for name in FILENAMES]
    lines+=['- [ Combined figures  PDF](cell_size_comparison.pdf)','- [ Source metrics  CSV](size_metrics.csv)','- [ Resolution settings  / IT  Value  CSV](size_acquisition.csv)','',
        ' Notes :  Box plots show median ,  Quartiles and  P5–P95;  Fraction error bars  Monte Carlo  Sampling interval ,  Excludes mesh  /  Model uncertainty .',
        ' Transit and exposure distributions conditioned on exit ;  State counts and cumulative exits use all responses . Wall capture ,  Censored  /  Reverse flow ,  Near-wall exposure reported separately .',
        f' Acquisition comparison uses a shared packet width  {ap["signal_width_ms"]:g} ms  And delivery fraction  {ap["signal_delivery_fraction"]:.0%},  Specified values . Packet width and identification efficiency held constant across diameters .',
        ' At matched outlet rates ,  Theoretical acquisition rates may coincide ;  Diameter affects transport and focusing; acquisition follows outlet timing .',
        ' One representative saved trajectory per diameter . Membrane permeability ,  Dissolution ,  Specified droplet radius and charge ,  Representative transfer sensitivity .',
        '0.5 μm  Equivalent particle-scale sphere ; 20 μm/50 μm  High confinement requires resolved deformable-cell validation . Experimental detection efficiency ,  Viability and grid-independence evidence pending .']
    if result.get('brownian_control_directory'):lines+=['',' Additional  `size_07_brownian_control.png`:  Matched  2 μm  Brownian on/off comparison ;  Earlier threshold data stored in  `brownian_cutoff_1_control/`.']
    if result.get('brownian_control_directory'):
        import json
        control=json.loads((root/result['brownian_control_directory']/'size_summary.json').read_text(encoding='utf-8'))
        row=next((r for r in control['rows'] if r['diameter_um']==2),None)
        if row and 'initial_exited' in row:
            lines+=['',f' Time-censoring analysis : 2 μm  At the earlier threshold  ( Brownian disabled ) Initial run  {row["initial_tip_timeout_s"]:g} s  Resolved tip trajectories  {row["initial_exited"]}/{row["samples"]}  Exits ;  Extended to  {row["final_tip_timeout_s"]:g} s  Result  {row["exited"]}/{row["samples"]}. Incomplete paths remain time-censored . Brownian-enabled comparison recomputes exits  /  Wall capture  /  Censored ,  See  cells.csv.']
    current=' Unknown ' if config.electrical.spray_current_na is None else f'{config.electrical.spray_current_na:g} nA'
    lines+=['',f' Voltage  {config.electrical.voltage_kv:g} kV,  Spray-current setting  {current}  Inherited reference setting ,  Reference input . Carrier entry uses a single-droplet charge-load estimate ,  Separate carrier-load and upstream-field closures .']
    (root/'comparison_readme.md').write_text('\n'.join(lines),encoding='utf-8')


def _brownian_control(plt,old,current,save):
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');a,b,c,d=axes.flat
    labels=[' Brownian disabled ',' Brownian enabled '];colors=['#d18a29','#2673b8'];cases=[old,current]
    horizon=max(float(np.nanmax(case['particles'].residence_time)) for case in cases)*1.04
    for case,label,color in zip(cases,labels,colors):
        p=case['particles'];ok=p.exited;times=np.sort(p.residence_time[ok])
        a.step(np.r_[0,times,horizon],np.r_[0,np.arange(1,len(times)+1)/len(ok)*100,len(times)/len(ok)*100],where='post',color=color,label=label)
    a.set(xlabel=' Time since capillary entry  (s)',ylabel=' Exited capillary  /  All responses  (%)',ylim=(0,103),title=' Cumulative transit-time distribution ');a.legend()
    _box(b,[case['particles'].stress['near_wall_time_s'] for case in cases],labels,colors,' Cumulative near-wall time  (s)');b.set_xlabel(' Matched  2 μm  Equivalent sphere ')
    _box(c,[case['particles'].outlet_radius[case['particles'].exited]*1e6 for case in cases],labels,colors,' Outlet centre positions  r (μm)');c.set_xlabel(' Matched  2 μm  Equivalent sphere ')
    for k,case in enumerate(cases):
        p=case['particles'];d.bar(k,p.exited.mean()*100,color=colors[k]);d.text(k,p.exited.mean()*100+1.5,f'{p.exited.sum()}/{len(p.status)}',ha='center')
    d.set(xticks=[0,1],xticklabels=labels,ylabel=' Exited  /  All responses  (%)',ylim=(0,110),title=' Outlet fraction ;  Tip censoring checked in both cases ')
    from ..particle.forces import brownian_diffusivity
    cfg=current['config'];diff=float(brownian_diffusivity(cfg.fluid.temperature_k,cfg.fluid.sample_viscosity,cfg.cell.diameter_um*.5e-6))
    fig.suptitle(f'2 μm  Brownian sensitivity :  Flow field ,  Concentration ,  Matched density and physical time step \n Free-space reference  D={diff*1e12:.3g} μm²/s; 200 s  One-dimensional  RMS  Displacement approximately  {np.sqrt(2*diff*200)*1e6:.3g} μm,  Free-space displacement scale ',fontsize=13)
    save(fig,'size_07_brownian_control')
