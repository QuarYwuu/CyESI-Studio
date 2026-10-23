"""Cell-diameter comparisons using a common dilute liquid-flow solution.

Diameter changes particle forces and wall clearance. Small-cell cases use the
stochastic reference integrator and its configured time step.
"""
from copy import deepcopy
from dataclasses import dataclass,field,asdict
from pathlib import Path
from types import SimpleNamespace
import json
import math
import time
import numpy as np
from ..acquisition.q_exactive import AcquisitionParameters,analyze as acquisition


@dataclass
class SizeParameters:
    diameters_um: list[float]=field(default_factory=lambda:[.5,2.,8.,20.])
    cells: int=1024
    saved_trajectories: int=80
    conditional_inlet: bool=True
    max_tip_timeout_s: float=20.
    brownian_cutoff_um: float=2.

    def validate(self,config):
        if not self.diameters_um or len(set(self.diameters_um))!=len(self.diameters_um):raise ValueError("Specify at least one diameter, without duplicates.")
        if len(self.diameters_um)>4:raise ValueError("Compare at most four diameters.")
        if any(not math.isfinite(d) or d<=0 or d>=config.geometry.inner_id_um for d in self.diameters_um):raise ValueError("Each diameter must be positive and fit inside the sample capillary.")
        if isinstance(self.cells,bool) or not isinstance(self.cells,int) or self.cells<16:raise ValueError("Each group requires at least 16 response trajectories.")
        if not isinstance(self.saved_trajectories,int) or self.saved_trajectories<1:raise ValueError("Save at least one spatial trajectory.")
        if not math.isfinite(self.max_tip_timeout_s) or self.max_tip_timeout_s<=0:raise ValueError("Maximum tip tracking time must be positive.")
        if not math.isfinite(self.brownian_cutoff_um) or self.brownian_cutoff_um<0:raise ValueError("Brownian-motion threshold must be nonnegative.")
        return self


def wilson(successes,total):
    if total<=0:return None,None
    p=successes/total;den=1+1.96**2/total
    center=(p+1.96**2/(2*total))/den
    half=1.96*math.sqrt(p*(1-p)/total+1.96**2/(4*total**2))/den
    return max(0.,center-half),min(1.,center+half)


def _conditional_state(data,particles):
    from ..cell_state.parameters import StateParameters
    from ..cell_state.membrane import LipidShell
    from ..cell_state.analysis import pre_nozzle
    from ..cell_state.flight import fly
    from ..cell_state.inlet import estimate_inlet
    eligible=[j for j,i in enumerate(data.ids) if particles.exited[i]]
    if not eligible:return dict(status='no_saved_exit',arrived=None,history=[],pre_history=[])
    median=float(np.median(particles.residence_time[particles.exited]))
    index=min(eligible,key=lambda j:abs(data.track(j)['end_time']-median))
    p=StateParameters(drop_radii_um=[20.],charge_fractions=[.15],methanol_permeabilities_m_s=[1e-6],dissolution_rates_s=[20.],flow_factors=[1.],sample_tracks=1)
    # A common 20 um-radius carrier isolates core-size effects. For custom
    # larger cores, require explicit assumptions rather than silently resizing.
    if data.config.cell.diameter_um>=40:return dict(status='carrier_not_larger_than_core',arrived=None,history=[],pre_history=[])
    shell=LipidShell(data.config.cell.diameter_um*.5e-6,p,1e-6,20.)
    state,pre=pre_nozzle(data,index,shell,True);metrics=shell.metrics(state)
    at_lip=metrics if metrics['shell_continuum_valid'] else None
    inlet=estimate_inlet(p)
    result,history=fly(data,shell,state,20e-6,.15,inlet['baseline_ambient_flow_l_min'],p,True)
    return dict(parameters=p.to_dict(),trajectory_index=index,transport_sample_id=int(data.ids[index]),
        at_lip=at_lip,inlet=inlet,**result,history=history,pre_history=pre)


def analyze(config,parameters=None,acquisition_parameters=None,progress=lambda s:None,on_case=lambda row:None):
    from ..engine import liquid_solution
    from ..electro.electrostatics import solve as electric_solution
    from ..particle.cell_transport import simulate
    from ..particle.events import simulate_events
    from ..particle.forces import brownian_diffusivity
    from ..visualization.explorer_data import ExplorerData
    config=deepcopy(config).validate();p=deepcopy(parameters or SizeParameters()).validate(config)
    config.cell.brownian_below_um=p.brownian_cutoff_um
    config.cell.brownian_enabled=None  # Apply this study's Brownian cutoff.
    ap=deepcopy(acquisition_parameters or AcquisitionParameters(signal_width_ms=20.)).validate()
    config.spray.enabled=False
    progress('Solving one shared liquid / solvent / ionic / electric field for the dilute size comparison')
    flow,solvent,coupling=liquid_solution(config,progress)
    electric=electric_solution(config,getattr(solvent,'ionic',None))
    cases=[];rows=[];acquisition_rows=[]
    # Finish deterministic cases first so slow Brownian progress is inspectable.
    order=sorted(p.diameters_um,key=lambda d:(d<=config.cell.brownian_below_um,d))
    for diameter in order:
        c=deepcopy(config);c.cell.diameter_um=float(diameter);c.cell.count=p.cells
        c.numerics.saved_trajectories=min(p.cells,p.saved_trajectories);c.validate()
        progress(f'Diameter {diameter:g} um: {p.cells} responses; Brownian={diameter<=c.cell.brownian_below_um}')
        started=time.perf_counter();particles=simulate(c,solvent,electric,progress=progress)
        initial_exited=int(particles.exited.sum());initial_timeout=c.numerics.particle_tip_timeout_s
        # Extend timed-out tracks with the same integration step; report unresolved paths as censored.
        while (np.any((particles.status==3)&(particles.stress['bulk_status']==0))
               and c.numerics.particle_tip_timeout_s<p.max_tip_timeout_s):
            c.numerics.particle_tip_timeout_s=min(p.max_tip_timeout_s,c.numerics.particle_tip_timeout_s*4)
            progress(f'Diameter {diameter:g} um: unresolved tip paths; extending horizon to {c.numerics.particle_tip_timeout_s:g} s at unchanged dt')
            particles=simulate(c,solvent,electric,progress=progress)
        events=simulate_events(c,particles)
        data=ExplorerData.from_result(SimpleNamespace(config=c,solvent=solvent,electric=electric,particles=particles,spray=None,events=events),source=f'size_{diameter:g}_um')
        ok=particles.exited;n=int(ok.sum());rate=c.cell.concentration_per_ml*c.fluid.sample_flow_ul_min/60000
        low,high=wilson(n,p.cells)
        sample=lambda a,q:float(np.quantile(np.asarray(a)[ok],q)) if n else None
        radius=diameter*.5e-6
        diffusivity=float(brownian_diffusivity(c.fluid.temperature_k,c.fluid.sample_viscosity,radius))
        # Free-space references explain competing size scalings; they are not
        # substituted for the actual wall-limited trajectories.
        settling=(c.cell.density-c.fluid.sample_density)*9.80665*diameter**2*1e-12/(18*c.fluid.sample_viscosity)
        row=dict(diameter_um=diameter,samples=p.cells,exited=n,
            initial_exited=initial_exited,initial_tip_timeout_s=initial_timeout,final_tip_timeout_s=c.numerics.particle_tip_timeout_s,
            wall_stopped=int(np.sum(particles.status==2)),censored_or_reverse=int(np.sum(particles.status==3)),
            brownian_enabled=diameter<=c.cell.brownian_below_um,
            actual_sample_flow_ul_min=c.fluid.sample_flow_ul_min,source_cells_s=rate,
            transport_fraction=n/p.cells,transport_sampling_ci95_low=low,transport_sampling_ci95_high=high,
            nozzle_cells_s=rate*n/p.cells,
            nozzle_unresolved_upper_s=rate*(n+int(np.sum(particles.status==3)))/p.cells,
            residence_p05_s=sample(particles.residence_time,.05),residence_p50_s=sample(particles.residence_time,.5),
            residence_p95_s=sample(particles.residence_time,.95),mean_residence_s=particles.diagnostics['mean_residence_s'],
            outlet_r50_um=None if not n else sample(particles.outlet_radius,.5)*1e6,
            outlet_r95_um=None if not n else sample(particles.outlet_radius,.95)*1e6,
            outlet_cell_envelope_r95_um=None if not n else sample(particles.outlet_radius,.95)*1e6+diameter*.5,
            mean_outlet_methanol_fraction=particles.diagnostics['mean_outlet_methanol_fraction'],
            mean_methanol_dose_fraction_s=float(np.mean(particles.methanol_dose[ok])) if n else None,
            mean_shear_dose_pa_s=particles.diagnostics['mean_shear_dose_pa_s'],
            near_wall_exposed_fraction=float(np.mean(particles.stress['near_wall_time_s']>0)),
            mean_near_wall_time_s=float(np.mean(particles.stress['near_wall_time_s'])),
            bulk_censored=int(np.sum(particles.stress['bulk_status']==3)),
            free_brownian_diffusivity_um2_s=diffusivity*1e12,
            free_brownian_1d_rms_200s_um=math.sqrt(2*diffusivity*200)*1e6,
            free_settling_speed_um_s=settling*1e6,
            stokes_relaxation_us=c.cell.density*diameter**2*1e-12/(18*c.fluid.sample_viscosity)*1e6,
            actual_ms_detection_cells_s=None,biologically_intact_fraction=None)
        ac=acquisition(row['nozzle_cells_s'],ap)
        for item in ac['rows']:
            acquisition_rows.append(dict(diameter_um=diameter,**{k:v for k,v in item.items() if k!='expected_windows'}))
        progress(f'Diameter {diameter:g} um: exits {n}/{p.cells}; median transit {row["residence_p50_s"]}; computing selected carrier state')
        inlet=_conditional_state(data,particles) if p.conditional_inlet else None
        case=dict(config=c,particles=particles,data=data,events=events,inlet=inlet)
        row['elapsed_s']=time.perf_counter()-started
        if inlet is not None:
            row['conditional_inlet_status']=inlet['status'];row['conditional_carrier_arrived']=inlet.get('arrived')
        rows.append(row);cases.append(case);on_case(dict(row))
    cases.sort(key=lambda x:p.diameters_um.index(x['config'].cell.diameter_um));rows.sort(key=lambda x:p.diameters_um.index(x['diameter_um']))
    acquisition_rows.sort(key=lambda x:(p.diameters_um.index(x['diameter_um']),x['actual_injection_ms'],x['resolution_at_mz200']))
    return dict(config=config,parameters=asdict(p),acquisition_parameters=ap.to_dict(),rows=rows,cases=cases,
        acquisition_rows=acquisition_rows,coupling=coupling,flow_diagnostics=flow.diagnostics,
        solvent_diagnostics=solvent.diagnostics,electric_diagnostics=electric.diagnostics)


def save_data(result,directory,source=''):
    from ..report.export import clean,write_csv
    root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    result['config'].save(root/'base_config.json')
    summary={k:v for k,v in result.items() if k not in ('config','cases')}
    summary.update(source=str(source),config=result['config'].to_dict(),limitations=[
        'Finite-size independent rigid-sphere sensitivity; 0.5 um is a particle-scale surrogate, not a typical human lymphocyte.',
        'All diameters share one recomputed pressure-corrected carrier flow. Concentration, density, zeta and seed are fixed; fluid is not size-rescaled.',
        'Flux-weighted inlet samples are conditioned on finite-size clearance, so no unmeasured entrance sieving efficiency is assigned.',
        'The size study explicitly uses a 2 um Brownian threshold by default (editable), since 2 um free diffusion is appreciable over long transits. It does not change the main simulator default.',
        'Wall-adjacent time, tip wall stops and timeout/reverse-flow censoring are distinct. Zero wall-adhesion hazard is not proof of 100% experimental recovery.',
        'Residence and outlet-composition distributions condition on exits; non-exits remain visible in the status counts and cumulative-exit denominator.',
        'Wilson intervals quantify Monte Carlo sampling only, not model or mesh uncertainty. No mesh-independence claim is made.',
        'Full MS rates assume common packet width / delivery. Smaller cells are not assigned shorter ion peaks or better identification by fiat.',
        'Conditional inlet states use one representative saved exit per diameter with identical uncalibrated membrane and carrier assumptions; not population entry efficiencies.',
        'Core diameter 20 um in a 50 um bore pushes the unresolved rigid-sphere closure; deformation, wall-resolved hydrodynamics and cell-cell interaction are absent.',
        'Real biological integrity, ion yield, metabolite identification and MS detection efficiency remain uncalibrated.' ])
    (root/'size_summary.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(root/'size_metrics.csv',result['rows']);write_csv(root/'size_acquisition.csv',result['acquisition_rows'])
    for case in result['cases']:
        diameter=case['config'].cell.diameter_um;path=root/f'size_{diameter:g}_um';path.mkdir(exist_ok=True)
        case['config'].save(path/'config.json');data=case['data'];particle=case['particles']
        np.savez_compressed(path/'fields.npz',**data.fields)
        (path/'summary.json').write_text(json.dumps(clean(dict(particles=particle.diagnostics,events=data.summary.get('events',{}),
            electric=result['electric_diagnostics'],coupling=result['coupling'])),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(path/'cells.csv',(dict(cell_id=i,status=int(particle.status[i]),initial_r_um=particle.initial_radius[i]*1e6,
            residence_s=particle.residence_time[i],outlet_r_um=particle.outlet_radius[i]*1e6,
            methanol_dose_fraction_s=particle.methanol_dose[i],outlet_methanol_fraction=particle.outlet_methanol[i],
            **{k:v[i] for k,v in particle.stress.items()}) for i in range(case['config'].cell.count)))
        if case['inlet'] is not None:
            (path/'conditional_inlet.json').write_text(json.dumps(clean(case['inlet']),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return root


def load_data(directory):
    """Restore plotting inputs without recomputing stochastic trajectories."""
    import csv
    from ..config import Config
    from ..visualization.explorer_data import ExplorerData
    root=Path(directory);result=json.loads((root/'size_summary.json').read_text(encoding='utf-8'))
    result['config']=Config.from_dict(result['config']);cases=[]
    for row in result['rows']:
        path=root/f'size_{row["diameter_um"]:g}_um';data=ExplorerData.load(path)
        with (path/'cells.csv').open(encoding='utf-8-sig',newline='') as stream:raw=list(csv.DictReader(stream))
        array=lambda name:np.array([float(x[name]) if x[name] not in ('','None') else np.nan for x in raw])
        status=array('status').astype(np.int8)
        fields=data.fields;stress={k[5:]:v for k,v in fields.items() if k.startswith('cell_')}
        bulk={k[len('bulk_trajectory_'):]:v for k,v in fields.items() if k.startswith('bulk_trajectory_')}
        particle=SimpleNamespace(status=status,exited=status==1,residence_time=array('residence_s'),
            initial_radius=array('initial_r_um')*1e-6,outlet_radius=array('outlet_r_um')*1e-6,
            outlet_methanol=array('outlet_methanol_fraction'),methanol_dose=array('methanol_dose_fraction_s'),
            stress=stress,bulk_history=bulk or None,diagnostics=data.summary['particles'])
        inlet=json.loads((path/'conditional_inlet.json').read_text(encoding='utf-8')) if (path/'conditional_inlet.json').exists() else None
        cases.append(dict(config=data.config,particles=particle,data=data,events=data.summary.get('events',{}),inlet=inlet))
    result['cases']=cases;return result
