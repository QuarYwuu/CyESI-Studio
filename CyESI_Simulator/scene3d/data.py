"""Identity-preserving 3-D reconstruction of saved CyESI numerical responses.

This adapter does not call a CFD solver.  Long-pipe transverse coordinates are
the saved x/y histories; axisymmetric tip responses retain the last bulk
azimuth.  A revolved axisymmetric field is explicitly not a native 3-D solve.
All public coordinates and lengths use SI metres, in x/y/z order.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections import OrderedDict

import numpy as np

from ..visualization.process_data import ProcessModel, ProcessParameters, selected_flight
from ..cell_state.parameters import StateParameters


def _clean(value):
    """JSON standard values only: unresolved NaN/Inf is null, never fabricated."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _mix_color(c, alpha=1.):
    c=float(np.clip(c,0,1))
    return [0.12+0.78*c, 0.58-0.22*c, 0.95-0.77*c, float(alpha)]


def _surface(profile, identifier, color, layer, segments=32):
    """Revolve an r,z polyline around the physical axial z direction."""
    profile=np.asarray(profile,float)
    theta=np.arange(segments)*2*np.pi/segments
    positions=[];indices=[]
    for r,z in profile:
        positions.extend(np.c_[r*np.cos(theta),r*np.sin(theta),np.full(segments,z)].ravel().tolist())
    for i in range(len(profile)-1):
        for j in range(segments):
            a=i*segments+j;b=i*segments+(j+1)%segments
            c=(i+1)*segments+j;d=(i+1)*segments+(j+1)%segments
            indices.extend([a,b,c,b,d,c])
    return dict(id=identifier,positions=positions,indices=indices,color=[*color[:3],1.],
                opacity=color[3],layer=layer,doubleSided=True)


def _annular_pipe(inner,outer,start,end,identifier,color,layer):
    # The closed profile includes finite quartz thickness and the real open bore.
    return _surface([(inner,start),(inner,end),(outer,end),(outer,start),(inner,start)],
                    identifier,color,layer)


def _contour_segments(r,z,values,level=.5,active=None):
    """Piecewise-linear marching squares; return only actual sampled contours.

    Ambiguous saddles are split at the cell-centre sign. No fabricated Taylor
    cone, disconnected-drop connection, boundary closing, or extrapolation.
    """
    values=np.asarray(values,float);result=[]
    for i in range(len(r)-1):
        for j in range(len(z)-1):
            if active is not None and not np.asarray(active[i:i+2,j:j+2]).all():
                continue
            q=np.array([[r[i],z[j]],[r[i+1],z[j]],[r[i+1],z[j+1]],[r[i],z[j+1]]])
            v=np.array([values[i,j],values[i+1,j],values[i+1,j+1],values[i,j+1]])
            if not np.isfinite(v).all():continue
            hits=[]
            for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                if (v[a]<level) != (v[b]<level):
                    hits.append(q[a]+(level-v[a])/(v[b]-v[a])*(q[b]-q[a]))
            if len(hits)==2:result.append(hits)
            elif len(hits)==4:
                # Connect the regions around the corresponding alternating
                # corner, with a centre-value asymptotic choice.
                pairs=[(0,1),(2,3)] if (v.mean()>=level)==(v[0]>=level) else [(0,3),(1,2)]
                result.extend([[hits[a],hits[b]] for a,b in pairs])
    return result


class SceneData:
    """One source, one steady Poisson feed, and one clock for every scene layer."""
    def __init__(self,data,acquisition=None,parameters=None,state_parameters=None,path=None):
        self.data=data
        self.state_parameters=deepcopy(state_parameters or StateParameters()).validate()
        self.model=ProcessModel(data,acquisition,parameters or ProcessParameters(duration_s=600.,startup='steady'))
        self._xyz={};self._geometry={};self._flights=OrderedDict()
        self._bulk_columns={int(k):i for i,k in enumerate(data.fields.get('bulk_trajectory_ids',[]))}
        source_identity=str(data.source).replace('\\','/')
        if '/outputs/' in source_identity:
            source_identity='outputs/'+source_identity.split('/outputs/',1)[1]
        token=json.dumps(dict(source=source_identity,config=data.config.to_dict(),
                             ids=data.ids.tolist()),sort_keys=True,default=str)
        self.source_id=hashlib.sha256(token.encode()).hexdigest()[:16]
        self.path=path

    def set_path(self,path):
        """Set display coordinate mapping only, without changing solved fields.

        A new native run must recompute transport under the route's gravity.
        Merely assigning a curve never claims those dynamics were recomputed.
        """
        self.path=path;self._geometry.clear();self._flights.clear()

    def _global(self,xyz):
        xyz=np.asarray(xyz,float)
        if self.path is None:return xyz
        station=self.path.length_m+xyz[...,2]-self.data.config.geometry.inner_tip
        return self.path.map(station,xyz[...,0],xyz[...,1])

    def _track_xyz(self,index):
        index=int(index)
        if index in self._xyz:return self._xyz[index]
        tr=self.model.tracks[index];f=self.data.fields;col=self._bulk_columns.get(int(self.data.ids[index]))
        if col is None:
            xyz=np.c_[tr['r'],np.zeros(len(tr['r'])),tr['z']]
            result=(tr['t'],xyz,0.,False)
        else:
            bt=np.asarray(f['bulk_trajectory_time_s'][:,col],float)
            bx=np.asarray(f['bulk_trajectory_x_m'][:,col],float)
            by=np.asarray(f['bulk_trajectory_y_m'][:,col],float)
            valid=np.isfinite(bt)&np.isfinite(bx)&np.isfinite(by)
            bt,bx,by=bt[valid],bx[valid],by[valid]
            if not len(bt):
                result=(tr['t'],np.c_[tr['r'],np.zeros(len(tr['r'])),tr['z']],0.,False)
            else:
                keep=np.r_[np.diff(bt)>1e-12,True];bt,bx,by=bt[keep],bx[keep],by[keep]
                azimuth=float(np.arctan2(by[-1],bx[-1]))
                x=tr['r']*np.cos(azimuth);y=tr['r']*np.sin(azimuth)
                bulk=tr['t']<=bt[-1]+1e-12
                x[bulk]=np.interp(tr['t'][bulk],bt,bx)
                y[bulk]=np.interp(tr['t'][bulk],bt,by)
                result=(tr['t'],np.c_[x,y,tr['z']],azimuth,True)
        self._xyz[index]=result
        return result

    def metadata(self):
        g=self.data.config.geometry;m=self.model
        lower=g.inner_tip-g.sample_length_mm*.001
        warnings=list(self.data.summary.get('warnings',[]))
        warnings.extend([
            'Saved-data mode: axisymmetric fields are revolved for viewing; this is not a native three-dimensional CFD solution.',
            'Bulk histories preserve stored transverse x/y. Tip motion preserves bulk exit azimuth; azimuthal tip dynamics are unresolved.',
            'The external lipid-carrier flight is conditional on assumed carrier size, charge, membrane and inlet conditions; it is not a capture-efficiency prediction.',
            'Saved short-time phase snapshots have their own numerical clock. They do not establish a steady spray at the long-pipe observation time.',
            'Acquisition packets represent assumed cell-origin ion signals, not intact cells travelling inside the mass spectrometer.',
        ])
        if not self._bulk_columns:warnings.append('No saved 3-D bulk histories: upstream transverse azimuth is unresolved and shown on one meridian.')
        bounds=[[-g.external_radius_mm*.001,-g.external_radius_mm*.001,lower],
                    [g.external_radius_mm*.001,g.external_radius_mm*.001,g.ms_gap_mm*.001]]
        if self.path is not None:
            points=np.r_[self.path.centers_m,self._global([[0.,0.,g.ms_gap_mm*.001]])]
            bounds=[(points.min(axis=0)-g.external_radius_mm*.001).tolist(),
                    (points.max(axis=0)+g.external_radius_mm*.001).tolist()]
            warnings.append('Saved trajectories mapped onto an editable route: this display mapping alone does not recompute route gravity or Dean flow.')
        from ..fluid.supply import segments
        transverse_gravity=any(np.hypot(row['gx'],row['gy'])>1e-9 for row in segments(self.data.config))
        return _clean(dict(mode='axisymmetric_reconstruction',source_id=self.source_id,
            process_projection=dict(up_transverse_unit=-m.projection_direction,
                transverse_gravity=transverse_gravity,plane='gravity_meridian',
                description='Transverse gravity points down in the 2-D process view; saved SI coordinates are unchanged.'),
            source=self.data.source,coordinate_system=dict(axes=['transverse_x','transverse_y','axial_z'],unit='m',middle_lip_z=0.),
            config=self.data.config.to_dict(),parameters=vars(m.parameters),
            acquisition=m.acquisition.to_dict(),timing=m.row,state_parameters=self.state_parameters.to_dict(),
            duration_s=m.parameters.duration_s,default_event=m.default_event,
            bounds=bounds,path=None if self.path is None else self.path.to_dict(),
            landmarks=dict(sample_entry_z_m=lower,resolved_start_z_m=g.liquid_start,
                inner_lip_z_m=g.inner_tip,middle_lip_z_m=0.,gas_lip_z_m=-g.gas_recess_mm*.001,
                ms_aperture_z_m=g.ms_gap_mm*.001,electrode_contact_z_m=-self.data.config.electrical.contact_distance_mm*.001),
            summary=m.summary(),acquisition_theory=m.theory,warnings=warnings,
            phase_times_s=[] if self.data.spray is None else self.data.spray['time_s'],
            spatial_response_count=len(self.data.ids),full_response_count=self.data.config.cell.count))

    def geometry(self,phase_frame=-1):
        sp=self.data.spray
        phase_frame=int(np.clip(phase_frame if phase_frame>=0 else (len(sp['time_s'])-1 if sp is not None else 0),
                                0,len(sp['time_s'])-1 if sp is not None else 0))
        if phase_frame in self._geometry:return self._geometry[phase_frame]
        g=self.data.config.geometry;start=g.inner_tip-g.sample_length_mm*.001
        sheath_start=min(g.liquid_start,-self.data.config.electrical.contact_distance_mm*.001)
        gas_end=-g.gas_recess_mm*.001;gas_start=gas_end-g.gas_supply_length_mm*.001
        meshes=[_annular_pipe(g.a,g.b,start,g.inner_tip,'inner_quartz',[.48,.78,.98,.13],'quartz'),
                _annular_pipe(g.c,g.d,sheath_start,0.,'sheath_quartz',[.54,.82,.96,.12],'quartz'),
                _annular_pipe(g.gas_id_um*.5e-6,g.gas_od_um*.5e-6,gas_start,gas_end,'gas_quartz',[.63,.78,.88,.1],'quartz'),
                _surface([(g.a,start),(g.a,g.liquid_start)],'upstream_sample',[.12,.5,.94,.13],'fluid'),
                _annular_pipe(self.state_parameters.inlet_diameter_mm*.5e-3,
                              max(self.state_parameters.inlet_diameter_mm*.6e-3,g.external_radius_mm*.001),
                              g.ms_gap_mm*.001,g.ms_gap_mm*.001+.12e-3,'ms_aperture',[.58,.66,.74,.6],'ms')]
        lines=[];f=self.data.fields
        # Actual FV cell patches on two perpendicular meridian planes. White
        # masked quartz remains empty; no interpolation crosses the quartz wall.
        re=np.asarray(f['liquid_r_edges_m']);ze=np.asarray(f['liquid_z_edges_m'])
        active=np.asarray(f['liquid_active'],bool);C=np.asarray(f['methanol_fraction'])
        stride_r=max(1,len(C)//48);stride_z=max(1,C.shape[1]//100)
        for angle in [0.,np.pi*.5]:
            pos=[];cols=[];inds=[]
            for i in range(0,C.shape[0],stride_r):
                for j in range(0,C.shape[1],stride_z):
                    if not active[i,j] or not np.isfinite(C[i,j]):continue
                    # Keep original-cell footprints when downsampling: do not
                    # bridge a masked solid wall with a coarse rectangle.
                    r0,r1=re[i:i+2];z0,z1=ze[j:j+2];k=len(pos)//3
                    pts=np.array([[r0,z0],[r1,z0],[r1,z1],[r0,z1]])
                    pos.extend(np.c_[pts[:,0]*np.cos(angle),pts[:,0]*np.sin(angle),pts[:,1]].ravel().tolist())
                    cols.extend(_mix_color(C[i,j],1.)*4);inds.extend([k,k+1,k+2,k,k+2,k+3])
            meshes.append(dict(id=f'methanol_meridian_{angle}',positions=pos,indices=inds,colors=cols,
                color=[1.,1.,1.,1.],opacity=.6,doubleSided=True,layer='fluid'))
        # A sample of actual velocity and E vectors, not decorative field lines.
        for field,layer,color in [('velocity','fluid',[.1,.8,.9,.7]),('field','electric',[.96,.76,.16,.7])]:
            rr,zz=(self.data.r,self.data.z) if field=='velocity' else (self.data.er,self.data.ez)
            sr=max(1,len(rr)//7);sz=max(1,len(zz)//18);R,Z=np.meshgrid(rr[::sr],zz[::sz],indexing='ij')
            keyr,keyz=('velocity_r_m_s','velocity_z_m_s') if field=='velocity' else ('field_r_v_m','field_z_v_m')
            ur=np.asarray(self.data.sample(R,Z,keyr));uz=np.asarray(self.data.sample(R,Z,keyz))
            magnitude=np.hypot(ur,uz);valid=np.isfinite(magnitude)&(magnitude>1e-20)
            if field=='velocity':valid&=np.asarray(self.data.sample(R,Z,'liquid_active'))>.99
            length=min(g.c*.6, max(np.ptp(zz)/40,1e-6));segments=[]
            for r,z,u,v,n in zip(R[valid],Z[valid],ur[valid],uz[valid],magnitude[valid]):
                for angle in [0.,np.pi*.5,np.pi,1.5*np.pi]:
                    a=np.array([r*np.cos(angle),r*np.sin(angle),z]);b=a+length*np.array([u*np.cos(angle),u*np.sin(angle),v])/n
                    segments.extend([*a,*b])
            lines.append(dict(id=field+'_vectors',positions=segments,color=color,layer=layer,width=1.,primitive='segments'))
        # Saved phase interface, including detached connected components.
        droplets=[]
        if sp is not None:
            rr=(sp['radial_edges_m'][:-1]+sp['radial_edges_m'][1:])*.5
            zz=(sp['axial_edges_m'][:-1]+sp['axial_edges_m'][1:])*.5
            for i,profile in enumerate(_contour_segments(rr,zz,sp['liquid_fraction'][phase_frame],.5,sp.get('active'))):
                meshes.append(_surface(profile,f'phi_05_{i}',[.25,.82,.94,.65],'phase',segments=20))
            droplets=self.data.droplet_frame(phase_frame)
        if self.path is not None:
            # Every layer shares the same continuous physical coordinate map.
            # Long-tube walls need a swept surface rather than two end rings.
            mapped=[]
            for mesh in meshes:
                if mesh['id'] in ('inner_quartz','upstream_sample'):continue
                pts=np.asarray(mesh['positions']).reshape(-1,3)
                mesh['positions']=self._global(pts).ravel().tolist();mapped.append(mesh)
            for radius,identifier,color in [(g.a,'inner_quartz_inner',[.48,.78,.98,.13]),
                                           (g.b,'inner_quartz_outer',[.48,.78,.98,.13])]:
                mapped.append(self.path.sweep(radius,segments=32,id=identifier,color=[*color[:3],1.],
                    opacity=color[3],doubleSided=True,layer='quartz'))
            for station,identifier in [(0.,'inner_quartz_entry'),(self.path.length_m,'inner_quartz_lip')]:
                cap=_surface([(g.a,start if station==0 else g.inner_tip),
                              (g.b,start if station==0 else g.inner_tip)],identifier,[.48,.78,.98,.15],'quartz')
                cap['positions']=self._global(np.asarray(cap['positions']).reshape(-1,3)).ravel().tolist();mapped.append(cap)
            stations=self.path.station_m[self.path.station_m<=self.path.length_m+g.liquid_start-g.inner_tip]
            end=self.path.length_m+g.liquid_start-g.inner_tip
            stations=np.unique(np.r_[0.,stations,end])
            if end>0:
                mapped.append(self.path.sweep(g.a,segments=24,stations=stations,id='upstream_sample',
                              color=[.12,.5,.94,1.],opacity=.1,doubleSided=True,layer='fluid'))
            for line in lines:
                line['positions']=self._global(np.asarray(line['positions']).reshape(-1,3)).ravel().tolist()
            meshes=mapped
        result=_clean(dict(source_id=self.source_id,mode='axisymmetric_reconstruction',meshes=meshes,lines=lines,
            active_phase_frame=phase_frame,phase_time_s=None if sp is None else sp['time_s'][phase_frame],
            phase_clock='independent_saved_short_transient',droplets=droplets,
            scalar_ranges=dict(methanol=[0.,1.],velocity_m_s=[0.,float(np.nanmax(np.hypot(f['velocity_r_m_s'],f['velocity_z_m_s'])))],
                               electric_v_m=[0.,float(np.nanmax(np.hypot(f['field_r_v_m'],f['field_z_v_m'])))]),
            notes=['Surface geometry uses physical SI coordinates; display distortion belongs only to the camera.',
                   'Sheath tube is drawn only over the known upstream electrical-contact distance; total sheath length is unspecified.',
                   'Each displayed phase patch is revolved from the computed phi=0.5 contour, not a prescribed cone.']))
        self._geometry[phase_frame]=result
        # Keep only a few assembled phase geometries in memory.
        if len(self._geometry)>4:self._geometry.pop(next(iter(self._geometry)))
        return result

    def _validate_event(self,event_id):
        if isinstance(event_id,bool):raise ValueError('Invalid event ID')
        i=int(event_id)
        if i!=float(event_id) or not 0<=i<len(self.model.records['event_id']):raise ValueError('Unknown event ID')
        return i

    def event(self,event_id):
        i=self._validate_event(event_id);m=self.model;k=int(m.records['template_id'][i]);t,xyz,theta,bulk=self._track_xyz(k)
        row=dict(m.by_id.get(i,{}));row.update(event_id=i,trajectory_index=k,transport_sample_id=int(self.data.ids[k]),
            injection_s=m.records['injection_s'][i],nozzle_s=m.records['nozzle_s'][i],
            end_age_s=m.delays[k],exited=bool(m.exited[k]),computed_bulk_xyz=bulk,
            tip_azimuth_rad=theta,track=dict(time_s=t,positions=self._global(xyz),
                local_positions_m=xyz,methanol=m.tracks[k]['C']),
            position_unit='m',flight_available=i in self._flights)
        return _clean(row)

    def frame(self,time_s,selected_id=None,max_cells=1200):
        time_s=float(time_s)
        if not np.isfinite(time_s) or time_s<0 or time_s>self.model.parameters.duration_s:raise ValueError('Time outside observation interval')
        max_cells=int(max_cells)
        if not 1<=max_cells<=5000:raise ValueError('Visible cell limit must be 1..5000')
        selected_id=None if selected_id is None else self._validate_event(selected_id)
        m=self.model;r=m.records;ages=time_s-r['injection_s'];templates=r['template_id']
        live=np.flatnonzero((ages>=0)&(ages<=m.delays[templates]))
        visible=live[:max_cells]
        if selected_id is not None and selected_id in live and selected_id not in visible:
            visible=np.r_[visible[:-1],selected_id]
        cells=[]
        for i in visible:
            k=int(templates[i]);t,xyz,theta,bulk=self._track_xyz(k);age=ages[i]
            position=[float(np.interp(age,t,xyz[:,a])) for a in range(3)]
            c=float(np.interp(age,t,m.tracks[k]['C']));color=_mix_color(c)
            cells.append(dict(id=int(i),event_id=int(i),position=self._global(position).tolist(),local_position_m=position,radius_m=self.data.config.cell.diameter_um*.5e-6,
                color=color,age_s=float(age),methanol=c,trajectory_index=k,computed_bulk_xyz=bulk,
                stage='long_pipe' if position[2]<self.data.config.geometry.liquid_start else 'resolved_tip',selected=int(i)==selected_id))
        # Only the selected, explicitly calculated conditional carrier may move
        # downstream of z=0. Other crossed cells remain accounted in the ledger.
        exterior=None
        if selected_id in self._flights:
            flight=self._flights[selected_id];h=flight['history'];elapsed=time_s-r['nozzle_s'][selected_id]
            # Subtracting an injection-time offset of hundreds of seconds can
            # round a microsecond terminal time slightly above its endpoint.
            if h and -1e-10<=elapsed<=h[-1]['time_after_lip_s']+1e-10:
                elapsed=float(np.clip(elapsed,0.,h[-1]['time_after_lip_s']))
                ht=[v['time_after_lip_s'] for v in h]
                xyz=np.array([v['position'] for v in h]);pos=[float(np.interp(elapsed,ht,xyz[:,a])) for a in range(3)]
                exterior=dict(id=selected_id,event_id=selected_id,position=pos,
                    radius_m=float(np.interp(elapsed,ht,[v['drop_radius_m'] for v in h])),color=[.96,.62,.2,.8],
                    core_radius_m=self.data.config.cell.diameter_um*.5e-6,
                    carrier_charge_c=flight.get('result',{}).get('carrier_charge_c'),
                    stage='conditional_lipid_carrier',conditional=True)
                state_keys=['intracellular_methanol_fraction','intracellular_tracer_fraction',
                            'membrane_lipid_remaining_fraction','dissolved_lipid_fraction',
                            'max_transmembrane_potential_v','max_pore_area_fraction','temperature_k',
                            'external_methanol_fraction']
                exterior['state']={key:float(np.interp(elapsed,ht,[v[key] for v in h]))
                                    for key in state_keys if all(key in v for v in h)}
                cells=[cell for cell in cells if cell['id']!=selected_id]
                cells.append(exterior)
            else:
                pre=flight.get('pre_history',[])
                selected=next((cell for cell in cells if cell['id']==selected_id),None)
                if selected is not None and pre:
                    age=ages[selected_id];pt=[v['time_s'] for v in pre]
                    if pt[0]<=age<=pt[-1]+1e-10:
                        selected['state']={key:float(np.interp(age,pt,[v[key] for v in pre]))
                            for key in ('intracellular_methanol_fraction','intracellular_tracer_fraction',
                                        'membrane_lipid_remaining_fraction','dissolved_lipid_fraction',
                                        'max_transmembrane_potential_v','max_pore_area_fraction')
                            if all(key in v for v in pre)}
        near=np.flatnonzero(np.isfinite(m.onsets)&(m.onsets>time_s-2.)&(m.onsets<time_s+.25))
        packet_rows=[dict(event_id=int(i),onset_s=float(m.onsets[i]),delivered=bool(m.delivered[i]),
                          status=m.by_id.get(int(i),{}).get('status','outside_cohort')) for i in near[-200:]]
        micro=m.row['microscan_period_ms']*.001;fill=m.row['actual_injection_ms']*.001
        cycle=m.row['record_cycle_ms']*.001;record=math.floor(time_s/cycle)
        gates=[[record*cycle+j*micro,record*cycle+j*micro+fill] for j in range(m.acquisition.microscans)]
        return _clean(dict(source_id=self.source_id,time_s=time_s,cells=cells,counts=m.conservation(time_s),
            live_physical_count=len(live),visible_count=len(visible),omitted_count=max(0,len(live)-len(visible)),
            selected_id=selected_id,external_selected=exterior,packets=packet_rows,
            acquisition=dict(record=record,gates_s=gates,collecting=any(a<=time_s<b for a,b in gates),
                packet_width_s=m.acquisition.signal_width_ms*.001),
            notes=['A display cap never changes computed rates or conservation counts.',
                   'Lip-passed origins without conditional flight remain downstream-unresolved, not lost.']))

    def flight(self,event_id):
        i=self._validate_event(event_id)
        if i in self._flights:
            self._flights.move_to_end(i);return self._flights[i]
        value=selected_flight(self.model,i,self.state_parameters)
        k=int(self.model.records['template_id'][i]);_,xyz,angle,_=self._track_xyz(k)
        for row in value['history']:
            row['position']=self._global([row['r_m']*np.cos(angle),row['r_m']*np.sin(angle),row['z_m']]).tolist()
        value.update(conditional=True,source_id=self.source_id,
            coordinate_mode='saved_axisymmetric_flight_at_terminal_bulk_azimuth',
            lip_position_m=self._global(xyz[-1]).tolist(),notes=[
                'Carrier radius and charge are scenario assumptions; cells are not assigned automatically to numerically resolved droplets.',
                'Gas background is a frozen saved nozzle field plus approximate inlet suction; no joint nozzle/inlet 3-D gas solve.',
                'Membrane continuum validity is not biological viability, and geometric inlet entry is not mass-spectral identification.'])
        value=_clean(value);self._flights[i]=value
        if len(self._flights)>8:self._flights.popitem(last=False)
        return value
