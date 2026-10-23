"""One solved record, an independent editable draft, and reusable view models."""
from collections import OrderedDict,Counter
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import threading
import numpy as np
from ..config import Config
from ..visualization.explorer_data import ExplorerData
from ..visualization.process_data import ProcessModel,ProcessParameters,selected_flight
from ..acquisition.q_exactive import AcquisitionParameters,analyze
from ..cell_state.analysis import outlet_rates
from ..cell_state.parameters import StateParameters
from ..fluid.supply import estimate_flow
from .compute import physical_key,digest,WorkbenchSolver


class Memo:
    def __init__(self,limit=4):
        self.limit=limit;self.values=OrderedDict();self.stats=Counter();self.lock=threading.RLock();self.inflight={}
    def get(self,key,fn):
        # Work on one missing key must not block ready values used by the UI.
        # Futures still coalesce a report/export and a view requesting one key.
        with self.lock:
            if key in self.values:
                self.stats['hit']+=1;self.values.move_to_end(key);return self.values[key]
            future=self.inflight.get(key)
            owner=future is None
            if owner:future=Future();self.inflight[key]=future
            else:self.stats['wait']+=1
        if not owner:return future.result()
        try:value=fn()
        except BaseException as error:
            with self.lock:self.inflight.pop(key,None)
            future.set_exception(error)
            raise
        with self.lock:
            self.stats['build']+=1;self.values[key]=value;self.values.move_to_end(key)
            while len(self.values)>self.limit:self.values.popitem(last=False)
            self.inflight.pop(key,None)
        future.set_result(value)
        return value

    def peek(self,key):
        """Return a ready value immediately, without waiting for its producer."""
        with self.lock:return self.values.get(key)


@dataclass
class Snapshot:
    name:str
    data:object
    acquisition:object
    resolution:int
    actual_it_ms:float


class WorkbenchSession:
    def __init__(self,config=None):
        self.draft=deepcopy(config or Config());self.base=None;self.data=None;self.pins=[]
        self.acquisition=AcquisitionParameters(signal_width_ms=20.)
        self.state_parameters=StateParameters()
        self.resolution=17500;self.actual_it_ms=30.;self.duration_s=600.
        self.now=10.;self.speed=1.;self.playing=True;self.selected_event=None;self.startup='steady'
        self.revision=0;self.files=Memo(3);self.concentrations=Memo(4);self.models=Memo(3)
        self.pin_counter=0
        self.acquisitions=Memo(12);self.flights=Memo(8);self.solver=WorkbenchSolver()
        self._last_positions=None
        self._draft_key=physical_key(self.draft);self._base_key=None
        self._event_key=digest(self.draft.events.__dict__)
        self._method_key=digest(self.acquisition.to_dict())
        self.views_ready=True;self.process_error=None
        self._accepted_model=None;self._accepted_process_key=None
        self.interaction_result=None;self.interaction_scene=None;self._interaction_key=None
        self.interaction_cache=Memo(2);self.interaction_fields=Memo(2);self.interaction_acquisitions=Memo(4)

    @property
    def pending(self):return self.base is None or self._draft_key!=self._base_key

    @property
    def current_pin(self):return self.pins[-1] if self.pins else None

    def read(self,directory):
        path=Path(directory).resolve()
        marker=path/'record.json'
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8')).get('status')!='complete':
            raise ValueError('该数值记录尚未完整保存, 请选择已完成的计算结果.')
        # Ionic and complete response revisions affect forces and event rates,
        # even when the displayed field archive itself did not change.
        names=('fields.npz','config.json','summary.json','spray.npz','ions.npz','cells.csv','record.json')
        token=(str(path),tuple((name,(path/name).stat().st_mtime_ns,(path/name).stat().st_size)
            for name in names if (path/name).exists()))
        return self.files.get(token,lambda:ExplorerData.load(path))

    def load(self,directory):
        data=self.read(directory)
        self.adopt(data,reset_draft=True)
        return data

    def adopt(self,data,reset_draft=False):
        self.base=data
        self._base_key=physical_key(data.config)
        if reset_draft:self.draft=deepcopy(data.config)
        elif physical_key(self.draft)==physical_key(data.config) and self.draft.supply.sample_control=='pressure':
            self.draft.fluid.sample_flow_ul_min=data.config.fluid.sample_flow_ul_min
        self._draft_key=physical_key(self.draft);self._event_key=digest(self.draft.events.__dict__)
        self.selected_event=None;self.now=10. if self.startup=='steady' else 0.
        self._sync();self.revision+=1

    def edit(self,config):
        self.draft=deepcopy(config).validate()
        self._draft_key=physical_key(self.draft);self._event_key=digest(self.draft.events.__dict__)
        self._sync();self.revision+=1

    def _sync(self):
        previous=self.data
        if self.base is None:self.data=None;return
        if self.pending:self.data=self.base
        else:
            value=self.draft.cell.concentration_per_ml
            self.data=self.concentrations.get((self._data_key(self.base),value,self._event_key),
                lambda:self._rate_view(value))
        self._last_positions=None
        if previous is not None and (previous.fields is not self.data.fields or previous.config.cell.concentration_per_ml!=self.data.config.cell.concentration_per_ml):
            self.selected_event=None

    def _rate_view(self,value):
        """Change dilute source rates without eagerly materializing a feed train.

        The shared ProcessModel builds its own identity-preserving finite
        realization in the view worker. Old finite-window counts cannot be
        scaled and are therefore withheld until that realization is ready.
        """
        base=self.base
        if value==base.config.cell.concentration_per_ml and self._event_key==digest(base.config.events.__dict__):return base
        config=deepcopy(base.config);config.cell.concentration_per_ml=float(value)
        config.events=deepcopy(self.draft.events);config.validate()
        summary=deepcopy(base.summary);rates=outlet_rates(config,summary)
        summary['events']=dict(rate_cells_s=rates['injection_cells_s'],
            steady_nozzle_cells_s=rates['nozzle_cells_s'],time_isolated_nozzle_cells_s=rates['time_isolated_nozzle_cells_s'],
            collection_efficiency=config.events.collection_efficiency,summaries=[],
            ms_entry_cells_s=None,ms_detected_single_cell_events_s=None,
            finite_window_pending=True,source='dilute source-rate update; shared continuous process realization',
            note='Previous finite-window counts are withheld. The workbench continuous-process ledger provides a newly seeded realization; rates are ensemble means.')
        return ExplorerData(config,base.fields,summary,base.spray,base.source)

    def set_method(self,parameters,resolution,actual_it):
        p=deepcopy(parameters).validate();options=ProcessParameters(duration_s=self.duration_s,
            resolution=resolution,actual_it_ms=actual_it,startup=self.startup).validate(p)
        self.acquisition=p;self.resolution=options.resolution;self.actual_it_ms=actual_it
        self._method_key=digest(p.to_dict())
        self._last_positions=None;self.revision+=1

    def _data_key(self,data):
        # Solved/reweighted data objects are immutable. Hash their physical
        # inputs once instead of copying the entire config every animation frame.
        key=getattr(data,'_workbench_physical_key',None)
        if key is None:key=physical_key(data.config);data._workbench_physical_key=key
        events=getattr(data,'_workbench_event_key',None)
        if events is None:events=digest(data.config.events.__dict__);data._workbench_event_key=events
        return (id(data.fields),key,data.config.cell.concentration_per_ml,events)

    def _process_key(self):
        return (None if self.data is None else self._data_key(self.data),self._method_key,
            self.resolution,self.actual_it_ms,self.duration_s,self.startup)

    def view_key(self):
        """Cheap signature for discarding obsolete asynchronous view results."""
        return (self._process_key(),tuple((id(p.data),p.name) for p in self.pins))

    def peek_process(self):
        key=self._process_key()
        if self._accepted_process_key==key:return self._accepted_model
        return self.models.peek(key)

    def accept_views(self,result,key):
        """Keep the displayed model alive independently of bounded study caches.

        A late worker result cannot replace the model of newer UI inputs.
        Retaining one accepted model also prevents an export or stale task from
        causing a supposedly ready animation to rebuild on the Tk thread.
        """
        if key!=self.view_key() or result is None:return False
        self._accepted_model=result.get('model');self._accepted_process_key=self._process_key()
        self._last_positions=None
        return True

    def prepare_views(self):
        """Warm reusable view models on a captured session in a worker thread."""
        self.metrics();self.acquisition_result()
        if self.current_pin is not None:self.metrics(self.current_pin.data)
        try:model=self.process();error=None
        except (ValueError,KeyError,TypeError,IndexError) as problem:model=None;error=str(problem)
        return dict(process_error=error,model=model)

    def process(self):
        if self.data is None:return None
        p=self.acquisition;key=self._process_key()
        if self._accepted_process_key==key and self._accepted_model is not None:return self._accepted_model
        return self.models.get(key,lambda:ProcessModel(self.data,p,ProcessParameters(duration_s=self.duration_s,
            resolution=self.resolution,actual_it_ms=self.actual_it_ms,startup=self.startup)))

    def positions(self):
        model=self.process()
        if model is None:return []
        key=(id(model),self.now)
        if self._last_positions is None or self._last_positions[0]!=key:
            points=model.positions(self.now)
            status=self.data.fields.get('cell_bulk_status')
            if self.data.config.supply.wall_capture_rate_s and status is not None:
                templates=model.records['template_id']
                capture_template=np.asarray(status)[self.data.ids]==2
                captures=np.flatnonzero(capture_template[templates]&
                    (model.records['injection_s']+model.delays[templates]<self.now))
                for event in captures:
                    j=int(templates[event]);track=model.tracks[j]
                    points.append(dict(event_id=int(event),trajectory_index=j,age_s=float(self.now-model.records['injection_s'][event]),
                        z_m=float(track['z'][-1]),r_m=float(track['r'][-1]),y_m=float(model.local_y[j][-1]),
                        methanol=0.,interaction_status='bulk_wall_bound',conditional_adhesion=True))
            self._last_positions=(key,points)
        points=self._last_positions[1]
        if self.active_interactions():
            result=self.interaction_result
            k=int(np.clip(np.searchsorted(result.time_s,self.now,side='right')-1,0,len(result.time_s)-1))
            affected=set(int(i) for i,st in zip(result.event_ids,result.states[k]) if st!=0)
            points=[p for p in points if p['event_id'] not in affected]
            for row in result.frame(self.now):
                i=row['event_id'];x,y,z=row['local_xyz_m'];template=int(model.records['template_id'][i])
                points.append(dict(event_id=i,trajectory_index=template,age_s=self.now-model.records['injection_s'][i],
                    z_m=z,r_m=float(np.hypot(x,y)),y_m=y,x_m=x,methanol=row['methanol_fraction'],
                    interaction_status=row['status'],cluster_id=row['cluster_id'],contact_code=row['contact_code']))
        return points

    def interaction_source_key(self):
        return (None if self.data is None else self._data_key(self.data),self.duration_s,self.startup)

    def has_interactions(self):
        return self.interaction_result is not None and self._interaction_key==self.interaction_source_key()

    def active_interactions(self):
        return self.has_interactions() and self.interaction_result.time_s[0]-1e-12<=self.now<=self.interaction_result.time_s[-1]+1e-12

    def compute_interactions(self,parameters,progress=lambda s:None):
        if self.data is None or self.pending:raise ValueError('请先应用物理参数并完成二维求解')
        from ..particle.axisymmetric_interactions import AxisymmetricScene
        from ..particle.interactions import simulate_interactions
        key=(self.interaction_source_key(),digest(parameters.to_dict()))
        def build():
            # Force gradients, liquid-side samplers and ionic sidecars depend
            # on fields/source, not the adhesion rates or collection method.
            scene=self.interaction_fields.get(self.interaction_source_key(),lambda:AxisymmetricScene(self.data,self.process()))
            result=simulate_interactions(scene,parameters,progress,field_provider=scene.field_provider)
            result.diagnostics['saved_bulk_azimuth']=bool('bulk_trajectory_x_m' in self.data.fields)
            if not result.diagnostics['saved_bulk_azimuth']:
                result.diagnostics['notes'].append('Archived responses lack azimuth; meridian placement is an explicit initial-condition assumption, not resolved 3-D contacts.')
            return scene,result
        return self.interaction_cache.get(key,build)

    def accept_interactions(self,scene,result,key):
        if key!=self.interaction_source_key():return False
        self.interaction_scene=scene;self.interaction_result=result;self._interaction_key=key
        self._last_positions=None;self.revision+=1
        return True

    def clear_interactions(self):
        self.interaction_result=None;self.interaction_scene=None;self._interaction_key=None
        self._last_positions=None;self.revision+=1

    def interaction_acquisition(self):
        if not self.has_interactions():return None
        from ..particle.interaction_analysis import acquisition_for_study
        key=(id(self.interaction_result),self._method_key,self.resolution,self.actual_it_ms)
        return self.interaction_acquisitions.get(key,lambda:acquisition_for_study(self.process(),self.interaction_result))

    def acquisition_result(self,data=None):
        data=data or self.data
        if data is None:return None
        key=(self._data_key(data),self._method_key,self.actual_it_ms)
        def compute():
            p=deepcopy(self.acquisition)
            p.actual_injection_times_ms=sorted(set([x for x in p.actual_injection_times_ms if x<=p.max_injection_time_ms]+[self.actual_it_ms]))
            return analyze(outlet_rates(data.config,data.summary)['nozzle_cells_s'],p,
                source=data.source,source_config=data.config.to_dict())
        return self.acquisitions.get(key,compute)

    def selected_row(self,data=None):
        result=self.acquisition_result(data)
        if result is None:return None
        return next(r for r in result['rows'] if r['resolution_at_mz200']==self.resolution
            and r['actual_injection_ms']==self.actual_it_ms)

    def metrics(self,data=None):
        data=data or self.data
        if data is None:return dict(injection=None,nozzle=None,transit=None,r95=None,pure=None,purity=None,transport=None)
        p=data.summary.get('particles',{});rates=outlet_rates(data.config,data.summary);row=self.selected_row(data)
        return dict(injection=rates['injection_cells_s'],nozzle=rates['nozzle_cells_s'],
            transit=p.get('mean_residence_s'),r95=p.get('outlet_r95_um'),
            pure=None if row is None else row['unique_pure_single_cell_events_s'],
            purity=None if row is None else row['nonempty_single_cell_purity'],transport=rates['modeled_transport_fraction'])

    def draft_preview(self):
        c=self.draft
        q=estimate_flow(c) if c.supply.sample_control=='pressure' else c.fluid.sample_flow_ul_min
        return dict(flow_ul_min=q,rate_cells_s=max(0.,q)*c.cell.concentration_per_ml/60000,
            approximate=c.supply.sample_control=='pressure',pending=self.pending)

    def pin(self,name=None):
        if self.data is None:raise ValueError('请先载入或计算一组结果')
        self.pin_counter+=1
        snapshot=Snapshot(name or f' Reference  {self.pin_counter}',self.data,deepcopy(self.acquisition),self.resolution,self.actual_it_ms)
        self.pins.append(snapshot);self.pins=self.pins[-4:];self.revision+=1
        return snapshot

    def flight(self,event_id,parameters=None,model=None):
        if self.has_interactions() and event_id in self.interaction_result.event_ids:
            raise ValueError('该细胞已有相互作用轨迹; 不能复用独立轨迹的条件飞行, 请返回基线后计算')
        model=model or self.process()
        if model is None:raise ValueError('请先载入运输结果')
        p=deepcopy(parameters or self.state_parameters).validate()
        index=int(model.records['template_id'][event_id])
        # Flight physics depends on the transport response, not injection time,
        # concentration or instrument resolution. Re-label a reused response.
        key=(id(model.data.fields),physical_key(model.data.config),index,digest(p.to_dict()))
        cached=self.flights.get(key,lambda:selected_flight(model,event_id,p))
        return dict(cached,event_id=int(event_id))

    def cache_summary(self):
        return dict(files=dict(self.files.stats),concentration=dict(self.concentrations.stats),
            process=dict(self.models.stats),acquisition=dict(self.acquisitions.stats),flight=dict(self.flights.stats),
            solver=dict(self.solver.cache.counts))


def catalog(root='outputs'):
    """Metadata only. Reading the library never decompresses a field archive."""
    root=Path(root);folders=list(root.iterdir()) if root.exists() else []
    runs=root/'workbench'/'runs'
    if runs.exists():folders+=list(runs.iterdir())
    rows=[]
    for path in folders:
        if (path/'verification_only.json').exists():continue
        if not (path/'fields.npz').exists() or not (path/'config.json').exists():continue
        try:
            marker=path/'record.json'
            if marker.exists() and json.loads(marker.read_text(encoding='utf-8')).get('status')!='complete':continue
            c=Config.load(path/'config.json')
            rows.append(dict(path=str(path.resolve()),name=path.name,stamp=(path/'fields.npz').stat().st_mtime,
                length_cm=c.geometry.sample_length_mm/10,diameter_um=c.cell.diameter_um,
                flow_ul_min=c.fluid.sample_flow_ul_min,concentration=c.cell.concentration_per_ml,
                voltage_kv=c.electrical.voltage_kv,spray=(path/'spray.npz').exists()))
        except (OSError,ValueError,KeyError,TypeError):continue
    return sorted(rows,key=lambda r:r['stamp'],reverse=True)
