"""Full-MS gating model for the original Q Exactive (not HF/HF-X).

Resolution/transient table: Thermo Q Exactive Getting Started / AGC FAQ;
independently published in Michalski et al., MCP 2011, Table I.
Actual IT, packet width and efficiencies are explicit scenario inputs.
"""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import json
import math
import numpy as np

TRANSIENT_MS={17500:64.,35000:128.,70000:256.,140000:512.}
SOURCES=[
    {'title':'Thermo Q Exactive: resolution, transient time, AGC and ion gating',
     'url':'https://www.thermofisher.com/us/en/home/technical-resources/technical-reference-library/mass-spectrometry-support-center/liquid-chromatography-mass-spectrometry-instruments-support/q-exactive-hybrid-quadrupole-orbitrap-mass-spectrometer-support/q-exactive-hybrid-quadrupole-orbitrap-mass-spectrometer-support-getting-started.html'},
    {'title':'Michalski et al. 2011, Q Exactive, Table I and parallel injection/detection',
     'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC3284220/'}]


@dataclass
class AcquisitionParameters:
    instrument: str='Q Exactive'
    mode: str='Full MS'
    max_injection_time_ms: float=30.  # user-confirmed method
    actual_injection_times_ms: list[float]=field(default_factory=lambda:[5.,10.,30.])
    microscans: int=1  # user-confirmed method
    overhead_ms: float=20.  # assumed transfer/control overhead, not measured
    extra_record_delay_ms: float=0.  # optional measured gap between recorded spectra
    signal_width_ms: float=0.  # rectangular cell-origin ion packet, unknown; impulse limit
    signal_delivery_fraction: float=1.  # assumed nozzle-to-ion-packet transmission
    single_event_identification_efficiency: float | None=None  # unknown until calibrated
    observation_durations_s: list[float]=field(default_factory=lambda:[60.,600.])
    purity_target: float=.9
    seed: int=43

    def validate(self):
        for key,value in vars(self).items():
            values=value if isinstance(value,list) else [value]
            if any(isinstance(x,(int,float)) and not math.isfinite(x) for x in values):
                raise ValueError(f' Acquisition parameters must be finite : {key}')
        if self.instrument!='Q Exactive' or self.mode!='Full MS':
            raise ValueError(' Acquisition model supports the original  Q Exactive  Of  Full MS  Continuous acquisition ')
        if self.max_injection_time_ms<=0 or not self.actual_injection_times_ms:
            raise ValueError('Maximum IT  And actual  IT  Setting must be positive ')
        if min(self.actual_injection_times_ms)<=0 or max(self.actual_injection_times_ms)>self.max_injection_time_ms:
            raise ValueError(' Actual  IT  Setting must be  > 0  And at most  Maximum IT')
        if not isinstance(self.microscans,int) or isinstance(self.microscans,bool) or not 1<=self.microscans<=10:
            raise ValueError('Microscans  Must be  1–10  Integers ')
        if min(self.overhead_ms,self.extra_record_delay_ms,self.signal_width_ms)<0:
            raise ValueError(' Overhead and signal width must be non-negative ')
        if not 0<=self.signal_delivery_fraction<=1:
            raise ValueError(' Delivery fraction must be in  0–1  Inner ')
        if self.single_event_identification_efficiency is not None and not 0<=self.single_event_identification_efficiency<=1:
            raise ValueError(' Identification efficiency must be unspecified or in  0–1  Inner ')
        if not self.observation_durations_s or min(self.observation_durations_s)<=0 or not 0<self.purity_target<1:
            raise ValueError(' Observation duration must be positive ,  Purity target must be in  0–1  Inner ')
        if not isinstance(self.seed,int) or isinstance(self.seed,bool) or self.seed<0:
            raise ValueError(' Random seed must be a non-negative integer ')
        return self

    def to_dict(self):return asdict(self)

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))).validate()


def _purity(mu):
    if mu==0:return None  # no nonempty spectra; purity is undefined
    if mu>700:return 0.
    return mu/math.expm1(mu)


def timing(resolution,injection_ms,p):
    transient=TRANSIENT_MS[resolution]
    micro=max(transient,injection_ms)+p.overhead_ms
    period=micro*p.microscans+p.extra_record_delay_ms
    # A spectrum combines m accumulation gates. A cell's rectangular signal
    # of width w contributes if its onset is in [gate_start-w, gate_end].
    # Adjacent expanded gates merge when w exceeds the gap between gates.
    expanded=injection_ms+p.signal_width_ms
    support=(expanded+(p.microscans-1)*min(micro,expanded))*1e-3
    return {'resolution_at_mz200':resolution,'transient_ms':transient,
        'actual_injection_ms':injection_ms,'microscan_period_ms':micro,
        'record_cycle_ms':period,'spectra_per_s':1000/period,
        'collection_duty_fraction':p.microscans*injection_ms/period,
        'cell_onset_support_s':support}


def occupancy(rate,times,identification_efficiency=None):
    """Exact stationary Poisson occupancy for identical rectangular packets.

    N distinct cell origins in one spectrum is Poisson(lambda*|U|), where U
    is the union of accumulation gates expanded backwards by packet width.
    For a tagged cell, K spectra can see its packet. If K>1, their onset
    supports are connected intervals sharing the tagged onset. The Poisson
    void inclusion/exclusion terms cancel except adjacent pairs, giving
    E[K] exp(-lambda L) - E[(K-1)+] exp(-lambda(L+D)). This counts a cell only
    once if at least one spectrum is pure; it does not count split spectra as
    different cells. Periodic equal gates within each record are assumed.
    """
    duration=times['record_cycle_ms']*1e-3;support=times['cell_onset_support_s']
    mu=rate*support;p0=math.exp(-mu);p1=mu*p0
    nonempty=-math.expm1(-mu)
    multiple=max(0.,nonempty-p1)
    copies=support/duration
    captured=min(1.,copies)
    unique_probability=p0*(min(copies,1.)+max(0.,copies-1)*(-math.expm1(-rate*duration)))
    unique_rate=rate*unique_probability
    detected=None if identification_efficiency is None else unique_rate*identification_efficiency
    return {'ion_packet_rate_s':rate,'mean_cell_origins_per_spectrum':mu,
        'empty_spectrum_probability':p0,'single_cell_spectrum_probability':p1,
        'mixed_spectrum_probability':multiple,'nonempty_single_cell_purity':_purity(mu),
        'pure_single_cell_spectra_s':p1/duration,
        'unique_pure_single_cell_events_s':unique_rate,
        'source_cell_pure_recovery_fraction':None if rate==0 else unique_probability,
        'sampling_opportunity_cells_s':rate*captured,
        'sampling_missed_cells_s':rate*(1-captured),
        'sampled_but_never_pure_cells_s':max(0.,rate*captured-unique_rate),
        'conditional_detected_unique_cells_s':detected}


def analyze(nozzle_rate,parameters=None,source='',source_config=None,rate_inputs=None,transport_pending=False):
    p=(parameters or AcquisitionParameters()).validate()
    if nozzle_rate is None or not math.isfinite(nozzle_rate) or nozzle_rate<0:
        raise ValueError(' Specify a finite non-negative steady outlet rate ')
    packet_rate=nozzle_rate*p.signal_delivery_fraction
    # mu/(exp(mu)-1) decreases monotonically. Below this mu, the target is met.
    lo,hi=0.,50.
    for _ in range(70):
        mid=(lo+hi)/2
        if _purity(mid)>=p.purity_target:lo=mid
        else:hi=mid
    rows=[]
    for injection in p.actual_injection_times_ms:
        for resolution in TRANSIENT_MS:
            times=timing(resolution,injection,p)
            row=dict(times,**occupancy(packet_rate,times,p.single_event_identification_efficiency))
            row['nozzle_pure_recovery_fraction']=None if nozzle_rate==0 else row['unique_pure_single_cell_events_s']/nozzle_rate
            row['packet_rate_for_target_purity_s']=lo/times['cell_onset_support_s']
            row['target_purity']=p.purity_target
            row['expected_windows']=[{'duration_s':t,
                'pure_single_cell_spectra':row['pure_single_cell_spectra_s']*t,
                'unique_pure_single_cell_events':row['unique_pure_single_cell_events_s']*t,
                'conditional_detected_unique_cells':None if row['conditional_detected_unique_cells_s'] is None
                    else row['conditional_detected_unique_cells_s']*t} for t in p.observation_durations_s]
            rows.append(row)
    return {'model':'stationary Poisson cell-origin packets and periodic Full MS accumulation gates',
        'instrument':p.instrument,'mode':p.mode,'source':str(source),'source_config':source_config,
        'rate_inputs':rate_inputs,'transport_pending':transport_pending,
        'parameters':p.to_dict(),'nozzle_cells_s':nozzle_rate,'ion_packet_rate_s':packet_rate,
        'actual_ms_detection_rate_s':None,'rows':rows,'sources':SOURCES,
        'definitions':{
            'nonempty_single_cell_purity':'P(N=1 | N>=1), distinct cell origins per recorded spectrum',
            'unique_pure_single_cell_events_s':'distinct cell-origin packets with at least one uncontaminated spectrum, counted once',
            'pure_single_cell_spectra_s':'pure recorded spectra per second; one long cell signal can yield several',
            'conditional_detected_unique_cells_s':'unique pure opportunities multiplied by user-calibrated identification efficiency, if supplied'},
        'limitations':[
            'User method: Full MS, Maximum IT 30 ms, Microscans 1 initially; all remain editable.',
            'Actual IT can be shorter than Maximum IT under AGC. IT scenarios are assumptions, not AGC predictions.',
            'Resolution denotes m/z 200. Transient times are sourced; the default 20 ms overhead is an unmeasured scenario.',
            'Cycle = microscans * (max(transient, actual IT) + overhead) + extra record delay; no dd-MS2 or polarity-switch simulation.',
            'One cell produces one rectangular ion packet with common width; zero width is an impulse limit. Any gate overlap contributes.',
            'Pulse width at the C-trap is not inlet flight time; extraction, inlet dispersion, adsorption and ion transport are uncalibrated.',
            'Nozzle-to-ion-packet delivery defaults to an explicit 100% scenario; it is not inferred from geometric droplet entry.',
            'Poisson arrivals, constant independent transmission, identical packets and stationary timing are assumed; correlated trains and AGC feedback require measured data.',
            'Single origin does not guarantee adequate ions, S/N, identifiable metabolites or an intact living cell. Real MS detection remains uncalibrated.',
            'Expected counts are ensemble means, not rounded Monte Carlo counts. Cells split across spectra are deduplicated analytically.']}


def sample_timeline(row,p,rate,duration_s=1.):
    """One seeded realization for viewing gate overlap; never a new rate fit."""
    cycle=row['record_cycle_ms']*1e-3;micro=row['microscan_period_ms']*1e-3
    width=p.signal_width_ms*1e-3;fill=row['actual_injection_ms']*1e-3
    rng=np.random.default_rng(p.seed)
    count=rng.poisson(rate*(duration_s+width))
    arrivals=np.sort(rng.uniform(-width,duration_s,count))
    scans=[]
    for index in range(int(math.ceil(duration_s/cycle))):
        gates=[(index*cycle+j*micro,index*cycle+j*micro+fill) for j in range(p.microscans)]
        ids=set()
        for left,right in gates:
            a=np.searchsorted(arrivals,left-width,side='left')
            b=np.searchsorted(arrivals,right,side='left')
            ids.update(range(a,b))
        scans.append({'index':index,'start_s':index*cycle,'gates':gates,'cell_ids':sorted(ids)})
    return {'arrivals_s':arrivals,'scans':scans,'signal_width_s':width,'duration_s':duration_s}
