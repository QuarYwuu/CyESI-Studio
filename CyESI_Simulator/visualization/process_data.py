"""Continuous cell events assembled from saved transport responses.

Animation uses stored spatial responses; ensemble rates use the full response
population. C-trap signal widths are supplied by the acquisition scenario.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import numpy as np
from ..particle.events import ContinuousCellStream
from ..acquisition.q_exactive import AcquisitionParameters, timing, analyze
from ..cell_state.analysis import outlet_rates


STATUS_LABELS = {
    'pure_complete': ' Complete packet  /  Single-source ', 'pure_partial': ' Partial packet  /  Single-source scan ',
    'pure_impulse': ' Impulse signal  /  Single-source scan ', 'mixed_only': ' Mixed scans only ',
    'missed': ' Missed gate ', 'not_delivered': ' No ion packet  ( Specified )',
}
STATUS_COLORS = dict(pure_complete='#148b72', pure_partial='#379ac2',
    pure_impulse='#148b72', mixed_only='#d78a37', missed='#d45c65', not_delivered='#8a96a4')
CELL_COLORS = ['#176bba','#9c52a9','#d17823','#158a7c','#c85374','#6572bd','#458b34','#8b6551']


@dataclass
class ProcessParameters:
    duration_s: float = 60.
    resolution: int = 17500
    actual_it_ms: float = 30.
    packet_delay_ms: float = 0.  # assumed constant lip-to-packet onset delay
    startup: str = 'steady'

    def validate(self, p):
        if self.resolution not in (17500,35000,70000,140000):raise ValueError("Unsupported resolution.")
        if not np.isfinite([self.duration_s,self.actual_it_ms,self.packet_delay_ms]).all():raise ValueError("Parameters must be finite.")
        if not 0 < self.duration_s <= 3600:raise ValueError("Observation duration must be between 0 and 3600 s.")
        if not 0 < self.actual_it_ms <= p.max_injection_time_ms:raise ValueError("Actual accumulation time must be positive and no greater than the maximum.")
        if self.packet_delay_ms < 0 or self.startup not in ('steady','empty'):raise ValueError("Invalid delay or startup mode.")
        return self


def classify_packets(onsets, event_ids, delivered, row, p, cohort):
    """Half-open gates; integrate rectangular packets and deduplicate origins.

    Background events outside the cohort remain in all occupancy calculations.
    Gates extend beyond the cohort boundary so a last packet is not truncated.
    A pure fragment is NOT a completely collected packet. A zero-width packet
    has an overlap flag, but no definable duration or captured-mass fraction.
    """
    onsets=np.asarray(onsets,float);event_ids=np.asarray(event_ids,int)
    delivered=np.asarray(delivered,bool);cohort=np.asarray(cohort,bool)
    if not (onsets.shape==event_ids.shape==delivered.shape==cohort.shape) or onsets.ndim!=1:
        raise ValueError('Packet arrays must be matching one-dimensional arrays')
    if len(np.unique(event_ids))!=len(event_ids):raise ValueError('Event IDs must be unique')
    if not np.isfinite(onsets).all():raise ValueError('Packet onsets must be finite')
    width=p.signal_width_ms*1e-3;cycle=row['record_cycle_ms']*1e-3
    micro=row['microscan_period_ms']*1e-3;fill=row['actual_injection_ms']*1e-3
    # Include only gates capable of seeing the evaluated cohort and its tails.
    target=onsets[cohort]
    start=math.floor(float(target.min())/cycle)-1 if len(target) else 0
    stop=math.ceil(float(target.max()+width)/cycle)+1 if len(target) else 0
    count=np.zeros(len(onsets),int);pure_count=count.copy()
    coverage=np.zeros(len(onsets));pure_coverage=coverage.copy();scans=[]
    order=np.argsort(onsets);sorted_t=onsets[order]
    for record in range(start,stop+1):
        gates=[(record*cycle+j*micro,record*cycle+j*micro+fill) for j in range(p.microscans)]
        hits={}
        for a,b in gates:
            # Positive-width interval intersection is strict at the left edge;
            # impulses at the gate start enter, those at the end do not.
            low=np.searchsorted(sorted_t,a-width,side='right' if width else 'left')
            high=np.searchsorted(sorted_t,b,side='left')
            for i in order[low:high]:
                if delivered[i]:hits[i]=hits.get(i,0.)+(max(0.,min(onsets[i]+width,b)-max(onsets[i],a)) if width else 1.)
        ids=sorted(hits)
        pure=len(ids)==1
        for i,amount in hits.items():
            count[i]+=1;coverage[i]+=amount
            if pure:pure_count[i]+=1;pure_coverage[i]+=amount
        scans.append(dict(index=record,start_s=record*cycle,gates=gates,
                          event_ids=[int(event_ids[i]) for i in ids]))
    rows=[]
    for i in np.flatnonzero(cohort):
        captured=min(1.,coverage[i]/width) if width else None
        pure_fraction=min(1.,pure_coverage[i]/width) if width else None
        status=('not_delivered' if not delivered[i] else 'missed' if count[i]==0 else
                'mixed_only' if pure_count[i]==0 else 'pure_impulse' if width==0 else
                'pure_complete' if pure_fraction>=1-1e-9 else 'pure_partial')
        rows.append(dict(event_id=int(event_ids[i]),packet_onset_s=float(onsets[i]),status=status,
                         scan_count=int(count[i]),pure_scan_count=int(pure_count[i]),
                         captured_fraction=captured,pure_captured_fraction=pure_fraction))
    return rows,scans


class ProcessModel:
    def __init__(self,data,acquisition=None,parameters=None):
        if data.rate_preview:raise ValueError("Flow rate changed. Run the flow solver to update capillary and focusing playback.")
        self.data=data;self.acquisition=deepcopy(acquisition or AcquisitionParameters()).validate()
        self.parameters=deepcopy(parameters or ProcessParameters()).validate(self.acquisition)
        self.row=timing(self.parameters.resolution,self.parameters.actual_it_ms,self.acquisition)
        if self.acquisition.signal_width_ms>60000:raise ValueError("The process view supports signal widths up to 60 s. Use acquisition analysis for longer signals.")
        self.tracks=[data.track(i) for i in range(len(data.ids))]
        self.delays=np.array([track['end_time'] for track in self.tracks])
        saved_exits=np.array([track['z'][-1]>=-1e-9 for track in self.tracks])
        if 'response_exited' in data.fields:
            saved_exits &= data.fields['response_exited'][data.ids]
        self.exited=saved_exits
        self.rates=outlet_rates(data.config,data.summary,data.rate_config)
        self.source_rate=self.rates['injection_cells_s'];g=data.config.geometry
        self.start_z=g.inner_tip-g.sample_length_mm*1e-3
        self.stage_z=np.r_[self.start_z+np.arange(4)/4*g.sample_length_mm*1e-3,g.inner_tip,0.]
        self.stage_names=[' Inlet ','25%','50%','75%',' Sample outlet ',' Sheath outlet ']
        self.crossing=np.full((len(self.tracks),len(self.stage_z)),np.nan)
        from ..fluid.supply import segments
        transverse=[np.array([s['gx'],s['gy']]) for s in segments(data.config) if np.hypot(s['gx'],s['gy'])>1e-9]
        vector=transverse[0] if transverse else np.array([1.,0.])
        self.projection_direction=vector/np.linalg.norm(vector)
        self.local_y=[]
        for i,track in enumerate(self.tracks):
            # Use recorded bulk histories for upstream paths; label geometric substitutes separately.
            for j,z in enumerate(self.stage_z):
                cross=np.flatnonzero(track['z']>=z-1e-12)
                if len(cross):
                    k=int(cross[0]);self.crossing[i,j]=track['t'][0] if k==0 else np.interp(z,track['z'][k-1:k+1],track['t'][k-1:k+1])
            self.local_y.append(self._local_y(i))
        self.has_bulk='bulk_trajectory_time_s' in data.fields
        # A fixed 60 s guard keeps source events and assigned trajectories stable across method changes.
        preroll=max(self.delays)+60. if self.parameters.startup=='steady' else 0.
        width=self.acquisition.signal_width_ms*.001;cycle=self.row['record_cycle_ms']*.001
        horizon=self.parameters.duration_s+width+cycle
        if self.source_rate*(preroll+horizon+width+cycle)>100000:
            raise ValueError("More than 100000 display events. Shorten the observation duration.")
        self.stream=ContinuousCellStream(self.source_rate,self.delays,self.exited,
             seed=data.config.numerics.seed+101,start_time=-preroll)
        self.records=self.stream.extend(horizon)
        r=self.records;self.cohort=np.isfinite(r['nozzle_s'])&(r['nozzle_s']>=0)&(r['nozzle_s']<self.parameters.duration_s)
        self.onsets=r['nozzle_s']+self.parameters.packet_delay_ms*.001
        # Independent random stream: changing resolution/width/delivery cannot
        # change the feed train or the trajectory attached to an existing ID.
        self.delivered=np.random.default_rng(data.config.numerics.seed+102).random(len(self.onsets))<self.acquisition.signal_delivery_fraction
        valid=np.isfinite(self.onsets)
        self.ledger,self.scans=classify_packets(self.onsets[valid],r['event_id'][valid],self.delivered[valid],
            self.row,self.acquisition,self.cohort[valid])
        self.by_id={x['event_id']:x for x in self.ledger}
        for x in self.ledger:
            i=x['event_id'];k=r['template_id'][i]
            x.update(injection_s=float(r['injection_s'][i]),nozzle_s=float(r['nozzle_s'][i]),
                     trajectory_index=int(k),transport_sample_id=int(data.ids[k]))
        selected=sorted(self.ledger,key=lambda x:x['nozzle_s'])
        self.default_event=selected[len(selected)//2]['event_id'] if selected else (int(r['event_id'][0]) if len(r['event_id']) else None)
        self.flight=None
        theory_parameters=deepcopy(self.acquisition)
        theory_parameters.actual_injection_times_ms=sorted(set(theory_parameters.actual_injection_times_ms+[self.parameters.actual_it_ms]))
        self.theory=analyze(self.rates['nozzle_cells_s'] or 0.,theory_parameters,data.source,data.config.to_dict())

    def _local_y(self,index):
        f=self.data.fields;track=self.tracks[index]
        if 'bulk_trajectory_y_m' not in f:return track['r'].copy()
        found=np.flatnonzero(f['bulk_trajectory_ids']==self.data.ids[index])
        if not len(found):return track['r'].copy()
        j=found[0];t=f['bulk_trajectory_time_s'][:,j]
        unique=np.r_[np.diff(t)>1e-12,True];t=t[unique]
        # Show the gravity plane, with gravity pointing down on the screen.
        # Using y alone would misleadingly show centering when horizontal
        # sedimentation actually takes place along the solver's local +x.
        y=-(self.projection_direction[0]*f['bulk_trajectory_x_m'][:,j][unique]
            +self.projection_direction[1]*f['bulk_trajectory_y_m'][:,j][unique])
        # Tip solver is axisymmetric, so only |r| is resolved after the bulk.
        # Use the last meridional projection for display.
        end_r=np.hypot(f['bulk_trajectory_x_m'][-1,j],f['bulk_trajectory_y_m'][-1,j])
        projected=track['r']*y[-1]/max(end_r,1e-30)
        return np.where(track['t']<=t[-1],np.interp(track['t'],t,y),projected)

    def event_track(self,event_id):return self.tracks[self.records['template_id'][int(event_id)]]

    def event_age(self,event_id,progress):
        tr=self.event_track(event_id)
        extra=self.acquisition.signal_width_ms*.001+self.row['record_cycle_ms']*.002+self.parameters.packet_delay_ms*.001
        return float(np.interp(progress,[0.,.55,.72,.86,1.],
                  [0.,tr['bulk_time'],tr['inner_time'],tr['end_time'],tr['end_time']+max(extra,.05)]))

    def global_time(self,event_id,progress):return float(self.records['injection_s'][event_id]+self.event_age(event_id,progress))

    def positions(self,time_s):
        r=self.records;ages=time_s-r['injection_s'];k=r['template_id']
        live=np.flatnonzero((ages>=0)&(ages<=self.delays[k]))
        result=[]
        for i in live:
            j=k[i];tr=self.tracks[j];age=ages[i]
            result.append(dict(event_id=int(i),trajectory_index=int(j),age_s=float(age),
                z_m=float(np.interp(age,tr['t'],tr['z'])),r_m=float(np.interp(age,tr['t'],tr['r'])),
                y_m=float(np.interp(age,tr['t'],self.local_y[j])),
                methanol=float(np.interp(age,tr['t'],tr['C']))))
        return result

    def conservation(self,time_s):
        r=self.records;k=r['template_id'];created=r['injection_s']<=time_s
        done=created&(r['injection_s']+self.delays[k]<=time_s)
        return dict(created=int(created.sum()),in_transit=int((created&~done).sum()),
             crossed_lip=int((done&self.exited[k]).sum()),transport_unresolved=int((done&~self.exited[k]).sum()))

    def summary(self):
        counts={status:sum(x['status']==status for x in self.ledger) for status in STATUS_LABELS}
        n=len(self.ledger);pure=sum(counts[k] for k in ('pure_complete','pure_partial','pure_impulse'))
        return dict(source=self.data.source,source_config=self.data.config.to_dict(),
            acquisition=self.acquisition.to_dict(),parameters=asdict(self.parameters),
            saved_trajectory_count=len(self.tracks),full_response_count=self.data.config.cell.count,
            computed_bulk_history=self.has_bulk,injection_rate_s=self.source_rate,
            full_population_nozzle_rate_s=self.rates['nozzle_cells_s'],
            cohort='middle-lip crossings in [0, duration); background packets retained; tails evaluated beyond duration',
            nozzle_cohort_count=n,counts=counts,pure_unique_count=pure,
            pure_recovery_of_nozzle_cohort=None if not n else pure/n,
            complete_pure_recovery_of_nozzle_cohort=None if not n or not self.acquisition.signal_width_ms else counts['pure_complete']/n,
            actual_ms_detection_rate_s=None,biologically_intact_entry_fraction=None,
            notes=[
                'The continuous movie resamples saved spatial trajectories, not all transport response samples. Counts are one realization, not fitted rates.',
                'Source feed is unbounded Poisson. Observation duration and the number of saved trajectories do not limit the physical cell population.',
                'Full-population theoretical rates and the saved-path movie realization are reported separately.',
                'Radial focusing does not guarantee axial time isolation. Independent transit delays do not regularize a stationary Poisson source.',
                'A captured ion packet is a cell-origin signal, not a physically intact cell. Full duration coverage assumes a rectangular packet of uniform intensity.',
                'Zero-width packets have undefined signal completeness. Pulse width and lip-to-packet delay require calibration.',
                'Nozzle-to-packet delivery is an assumption including all intervening transfer losses; geometric inlet entry is not multiplied into it again.',
                'The optional lipid-carrier flight is a conditional selected-cell case, not an estimated inlet capture efficiency.',
                'Censored transport, unfinished pipe transit, modeled adhesion and missed ion-collection windows are distinct states.' ])


def selected_flight(model,event_id,parameters=None):
    """One actual saved response + explicit carrier/shell/inlet assumptions."""
    from ..cell_state.parameters import StateParameters
    from ..cell_state.membrane import LipidShell
    from ..cell_state.analysis import pre_nozzle
    from ..cell_state.inlet import estimate_inlet
    from ..cell_state.flight import fly
    p=deepcopy(parameters or StateParameters()).validate()
    index=int(model.records['template_id'][event_id])
    if not model.exited[index]:raise ValueError("The selected trajectory has not exited. Carrier transfer requires an outlet state.")
    # Use the middle listed scenario, record every assumption in the export.
    middle=lambda values:values[len(values)//2]
    permeability=middle(p.methanol_permeabilities_m_s);dissolution=middle(p.dissolution_rates_s)
    radius=middle(p.drop_radii_um);charge=middle(p.charge_fractions);factor=middle(p.flow_factors)
    shell=LipidShell(model.data.config.cell.diameter_um*.5e-6,p,permeability,dissolution)
    state,pre=pre_nozzle(model.data,index,shell,True);inlet=estimate_inlet(p)
    result,history=fly(model.data,shell,state,radius*1e-6,charge,inlet['baseline_ambient_flow_l_min']*factor,p,True)
    # Flight histories are decimated for storage. Append the computed terminal
    # state so an aperture crossing is displayed at the aperture, not one output
    # interval before it. This adds no extrapolation or new flight dynamics.
    if 'final_z_mm' in result:
        terminal=dict(time_after_lip_s=result['flight_time_ms']*.001,
            z_m=result['final_z_mm']*.001,r_m=result['final_r_um']*1e-6,
            drop_radius_m=result['final_drop_radius_um']*1e-6,temperature_k=result['final_temperature_c']+273.15,
            external_methanol_fraction=result['final_coating_methanol_fraction'],
            **{k:result[k] for k in ('intracellular_methanol_fraction','intracellular_tracer_fraction',
                'membrane_lipid_remaining_fraction','dissolved_lipid_fraction','max_transmembrane_potential_v',
                'max_pore_area_fraction','shell_continuum_valid')})
        if history and abs(history[-1]['time_after_lip_s']-terminal['time_after_lip_s'])<1e-12:history[-1]=terminal
        else:history.append(terminal)
    return dict(event_id=int(event_id),trajectory_index=index,transport_sample_id=int(model.data.ids[index]),
        parameters=p.to_dict(),scenario=dict(drop_radius_um=radius,charge_fraction=charge,
        permeability_m_s=permeability,dissolution_rate_s=dissolution,flow_factor=factor),
        inlet=inlet,result=result,history=history,pre_history=pre)
