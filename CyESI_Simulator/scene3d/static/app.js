import {SceneRenderer} from './renderer.js';
import {SliceView} from './slices.js';

const $=id=>document.getElementById(id);
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dot=(a,b)=>a.reduce((v,x,k)=>v+x*b[k],0);
const sub=(a,b)=>a.map((v,k)=>v-b[k]);
const norm=a=>Math.hypot(...a);
const fmt=(n,d=3)=>n==null||!Number.isFinite(Number(n))?'—':Number(n).toLocaleString('en-GB',{maximumFractionDigits:d});
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state={meta:null,geometry:null,frame:null,event:null,selected:null,time:0,playing:true,follow:false,
 revision:0,token:'',busyFrame:false,busyMeta:false,lastFrame:0,lastMeta:0,field:'methanol',phase:-1,fieldClock:'volume',sceneEpoch:0,pendingFlight:null,autoTrip:false};
let renderer,cutView;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,9000);}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-CyESI-Session':state.token},body:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw new Error(value.error||response.statusText);return value;}
function safeRun(fn){return (...args)=>Promise.resolve().then(()=>fn(...args)).catch(e=>toast(e.message));}

// A single continuous display lens. Physical data stay in SI. The centreline
// is compressed along arclength, while local cross-sections retain one scale.
// Nearest-segment mapping is cached for static vertices; this is visualization,
// never a substitute for field or particle integration.
function makeMapper(path,lens){
 if(!lens)return p=>[p[2]*4000,p[1]*4000,-p[0]*4000];
 if(!path)return null;
 const s=path.station_m,c=path.centers_m,L=s.at(-1),n=s.length;
 const display=Array.from({length:n},()=>[0,0,0]);
 const derivative=q=>{
  let scale=(4/.0015)/Math.sqrt(1+((q-L)/.0015)**2);
  if(path.metadata.kind==='up_bend_horizontal'){
   const a=path.metadata.up_length_m,b=a+path.metadata.bend_length_m;
   const distance=Math.max(a-q,0,q-b),fade=Math.exp(-((distance/.01)**2));
   scale=Math.max(scale,80*fade);
  }
  return scale;
 };
 for(let i=n-2;i>=0;i--){const f=derivative((s[i]+s[i+1])/2);display[i]=display[i+1].map((v,k)=>v-(c[i+1][k]-c[i][k])*f);}
 const lastT=path.tangents.at(-1),end=c.at(-1),cache=new Map();
 const orient=p=>[p[2],p[1],-p[0]];
 return p=>{
  const key=p.join(',');if(cache.has(key))return cache.get(key);
  const terminal=dot(sub(p,end),lastT);
  let q;
  if(terminal>0){
   const residual=sub(sub(p,end),lastT.map(x=>x*terminal));
   const axial=4*Math.asinh(terminal/.0015);
   q=lastT.map((x,k)=>x*axial+residual[k]*4000);
  }else{
   // Coarse closest point, followed by exact segment projections nearby.
   const stride=Math.max(1,Math.floor(n/48));let nearest=0,best=Infinity;
   for(let i=0;i<n;i+=stride){const d=dot(sub(p,c[i]),sub(p,c[i]));if(d<best){best=d;nearest=i;}}
   const lastD=dot(sub(p,end),sub(p,end));if(lastD<best)nearest=n-1;
   let seg=0,t=0,projection=c[0];best=Infinity;
   for(let i=Math.max(0,nearest-stride-1);i<Math.min(n-1,nearest+stride+1);i++){
    const d=sub(c[i+1],c[i]),r=sub(p,c[i]),u=clamp(dot(r,d)/Math.max(dot(d,d),1e-30),0,1);
    const point=c[i].map((v,k)=>v+u*d[k]),e=norm(sub(p,point));
    if(e<best){best=e;seg=i;t=u;projection=point;}
   }
   q=display[seg].map((v,k)=>v+t*(display[seg+1][k]-v)+(p[k]-projection[k])*4000);
  }
  const result=orient(q);if(cache.size<300000)cache.set(key,result);return result;
 };
}
function globalPoint(local){const p=state.meta?.path;if(!p)return local;const n=p.centers_m.length-1,c=p.centers_m[n],N=p.normals[n],B=p.binormals[n],T=p.tangents[n],dz=local[2]-state.meta.landmarks.inner_lip_z_m;return c.map((v,k)=>v+N[k]*local[0]+B[k]*local[1]+T[k]*dz);}
function setMapping(){if(!renderer||!state.meta)return;renderer.setCoordinateMapper(makeMapper(state.meta.path,$('lens').checked));renderer.setTransform({axialLens:$('lens').checked,cellExaggeration:Number($('magnify').value)});}
function configureInputs(meta){const c=meta.config;
 for(const [id,value] of Object.entries({length:c.geometry.sample_length_mm/10,pressure:c.supply.sample_pressure_mbar,sheath:c.fluid.sheath_flow_ul_min,voltage:c.electrical.voltage_kv,gaspressure:c.fluid.gas_pressure_kpa,diameter:c.cell.diameter_um,concentration:c.cell.concentration_per_ml,cellzeta:c.cell.zeta_mv,current:meta.mode==='native_cartesian_tip'?(c.electrical.spray_current_na==null?'':c.electrical.spray_current_na/1000):.6,conductivity:meta.mode==='native_cartesian_tip'?c.electrical.sheath_conductivity_s_m:.005,quartzzeta:c.electrical.quartz_zeta_mv,resolution:meta.parameters.resolution,it:meta.parameters.actual_it_ms,packet:meta.acquisition.signal_width_ms}))$(id).value=value;
 $('brownian').value=c.cell.brownian_enabled==null?'auto':c.cell.brownian_enabled?'on':'off';
 $('bulk-wall-rate').value=c.supply.wall_capture_rate_s??0;$('bulk-wall-gap').value=c.supply.wall_capture_gap_um??.1;
 document.querySelectorAll('[data-geometry]').forEach(el=>el.value=c.geometry[el.dataset.geometry]);$('contact-distance').value=c.electrical.contact_distance_mm;
 $('currentmode').value=meta.native_volume?.current_mode||'specified';$('current').disabled=$('currentmode').value==='scaling';
 $('time').max=meta.duration_s;$('duration').textContent=fmt(meta.duration_s,0)+' s';
 const p=meta.path?.metadata;if(p?.kind==='up_bend_horizontal'){$('bend').value=p.bend_radius_m*1000;$('upfraction').value=p.up_fraction*100;}
 if(p?.control_points_m)$('waypoints').value=p.control_points_m.map(p=>p.map(v=>Number((v*1000).toFixed(4))).join(', ')).join('\n');
 $('custompath').checked=p?.kind==='editable_control_points';
 if(meta.native_volume_settings){for(const [id,key] of Object.entries({refinement:'refinement',axialcells:'axial_cells',flowsteps:'flow_steps',coupling:'coupling_iterations',asymmetry:'gas_asymmetry'}))$(id).value=meta.native_volume_settings[key];}
}
function showMeta(meta,changed,sameSource=false){
 state.meta=meta;state.token=meta.session_token;
 cutView?.updateMeta(meta);
 const native=meta.mode==='native_cartesian_tip';
 const unresolved=native&&['liquid','gas'].some(k=>{const d=meta.native_volume?.[k];return d?.momentum_converged===false||(d?.momentum_converged==null&&d?.last_velocity_relative_change>(meta.config.numerics.flow_tolerance||2e-6));});
 $('mode').textContent=native?(unresolved?' 3D field  ·  Solver preview ':' 3D Cartesian  ·  Capillary transport '):' Saved results  ·  Axisymmetric reconstruction ';
 $('mode').style.color=native&&!unresolved?'#65dfbf':'#d8be83';
 $('scope-title').textContent=native?' 3D domain ':' Saved domain ';
 $('scope-text').textContent=native?' Outlet :  3D gas-liquid flow  /  Species  /  Electric field ;  Long capillary :  Fully developed flow along the capillary path . Transient tip model .':' Axisymmetric field reconstruction ;  Select “ Run 3D simulation ” Switch to the Cartesian solver .';
 const summary=meta.summary;
 const theoryRows=(meta.acquisition_theory?.rows||[]).filter(r=>Math.abs(r.actual_injection_ms-meta.parameters.actual_it_ms)<1e-7);$('method-comparison').innerHTML='<table><thead><tr><th> Resolution </th><th> Single-source  / s</th><th> Accumulation duty cycle </th></tr></thead><tbody>'+theoryRows.map(r=>'<tr><td>'+r.resolution_at_mz200+'</td><td>'+fmt(r.unique_pure_single_cell_events_s,3)+'</td><td>'+fmt(r.collection_duty_fraction*100,1)+'%</td></tr>').join('')+'</tbody></table>';
 $('actual-flow').textContent=' Actual sample  '+fmt(meta.config.fluid.sample_flow_ul_min,4)+' µL/min ·  Sheath  '+fmt(meta.config.fluid.sheath_flow_ul_min,2)+' µL/min · '+fmt(meta.config.geometry.sample_length_mm/10,1)+' cm  Sample capillary ';
 $('input-rate').textContent=fmt(summary.injection_rate_s,2);$('outlet-rate').textContent=fmt(summary.full_population_nozzle_rate_s,2);
 $('complete-rate').textContent=fmt((summary.counts?.pure_complete||0)/meta.duration_s,3);
 $('diagnostics').textContent=JSON.stringify({grid:meta.native_volume?.shape,
 liquid:meta.native_volume?.liquid,gas:meta.native_volume?.gas,electric:meta.native_volume?.electric,
 supply:meta.native_volume?.supply,transport:meta.native_transport,phase:meta.native_phase},null,2);
 $('warnings').innerHTML=(meta.warnings||[]).map(w=>'<li>'+esc(w)+'</li>').join('');
 $('record-path').textContent=meta.saved_record?' New record : '+meta.saved_record:' Source record : '+meta.source;
 $('job').classList.toggle('failed',meta.job.state==='failed');
 $('job').textContent=meta.job.state==='failed'?' Job stopped : '+meta.job.error:meta.job.state==='running'?' Running  · '+meta.job.message:' Ready  · '+(native?' Cartesian result available ':' Load a saved result  /  Run a 3D simulation ');
 $('cancel').hidden=meta.job.state!=='running';if(meta.job.state==='cancelled')$('job').textContent=' Job cancelled  ·  Previous results retained ';
 for(const id of ['run','load','method','flight'])$(id).disabled=meta.job.state==='running';
 $('compare-voltage').disabled=meta.job.state==='running'||!native;
 $('interaction-run').disabled=meta.job.state==='running'||!native;
 $('interaction-jump').disabled=!meta.interaction_study;
 $('interaction-clear').disabled=meta.job.state==='running'||!meta.interaction_study;
 const study=meta.interaction_study;
 if(study){const p=study.parameters||{},d=study.diagnostics||{};$('interaction-diagnostics').textContent=JSON.stringify(d,null,2);$('interaction-info').textContent=' Local window  '+fmt(p.start_s,3)+'–'+fmt(Number(p.start_s)+Number(p.duration_s),3)+' s;  Current flow field . System rates use independent transport ,  Local statistics reported separately .'+(d.limits?.length?' '+d.limits.join(' '):'');const totals={' Cell-pair contacts ':fmt(d.pair_contacts,0)+'  Pairs ',' Wall  /  Outlet contacts ':fmt(d.wall_contacts,0)+' / '+fmt(d.outlet_contacts,0),' Cell bonds  /  Wall  /  Outlet bonds ':[d.pair_bonds_formed,d.wall_bonds_formed,d.outlet_bonds_formed].map(v=>fmt(v,0)).join(' / '),' Unbinding events ':fmt(d.bonds_released,0)+'  Events ',' Maximum induced velocity ':fmt(d.max_pair_disturbance_m_s*1000,5)+' mm/s'};$('interaction-totals').innerHTML=Object.entries(totals).map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>').join('');}else{$('interaction-counts').innerHTML='<dt> Local model </dt><dd> Not computed </dd>';$('interaction-totals').innerHTML='';$('interaction-info').textContent='';$('interaction-diagnostics').textContent=' Ready to compute ';}
 if(meta.voltage_comparison){const v=meta.voltage_comparison;$('voltage-info').textContent=' Matched-flow comparison :  Maximum liquid velocity difference  '+fmt(v.liquid_velocity_difference_max_m_s*1000,4)+' mm/s;  Interface velocity difference  RMS '+fmt(v.phase_liquid_velocity_difference_rms_m_s*1000,3)+' mm/s,  Matched  t='+fmt(v.matched_phase_time_s*1e6,3)+' µs.';}
 const estimate=meta.native_volume?.current_estimate||meta.current_estimate;
 if(estimate){const transient=meta.native_phase?.circuit_total_supply_current_a_range;$('current-estimate').textContent=' Scaling estimate  '+fmt(estimate.predicted_current_ua,3)+' µA · '+(estimate.within_user_range?' Within reference range ':' Outside reference range ')+' ·  Supply constraint '+(estimate.electrical_supply_feasible?' Feasible ':' Incompatible ')+'.'+(transient?' Transient supply current  '+transient.map(v=>fmt(v*1e6,3)).join('–')+' µA ( Includes displacement current ).':'')+'K  Specified conductivity ;  Cone-jet scaling estimate .';}
 const phase=meta.phase_times_s||[];$('phasetime').max=Math.max(0,phase.length-1);$('phasetime').disabled=!phase.length;
 if(changed&&!sameSource){configureInputs(meta);setMapping();state.time=clamp(state.time,0,meta.duration_s);}
 if(changed){$('resolution').value=meta.parameters.resolution;$('it').value=meta.parameters.actual_it_ms;$('packet').value=meta.acquisition.signal_width_ms;const p=meta.charge_parameters;if(p){$('membranecap').value=p.membrane_capacitance_f_m2;$('internalconductivity').value=p.cytosol_conductivity_s_m;}if(meta.native_phase_settings){$('phaseduration').value=meta.native_phase_settings.duration_us;$('phasesteps').value=meta.native_phase_settings.max_steps;}}
}
async function refreshRecords(){const records=await api('/api/records'),current=$('records').value;$('records').replaceChildren();for(const record of records){const option=document.createElement('option');option.value=record.id;option.textContent=record.label;$('records').append(option);}const saved=records.find(r=>r.source===state.meta?.saved_record);$('records').value=saved?.id||current||records[0]?.id||'';}
async function refreshMeta(){if(state.busyMeta)return;state.busyMeta=true;try{const meta=await api('/api/meta');const changed=meta.revision!==state.revision,sameSource=state.meta?.source_id===meta.source_id;showMeta(meta,changed,sameSource);if(changed){state.revision=meta.revision;state.sceneEpoch++;state.event=null;state.selected=sameSource&&state.selected!=null?state.selected:meta.default_event;await loadGeometry();await selectEvent(state.selected,false);await requestFrame();await refreshRecords();}if(state.pendingFlight!=null&&meta.job.state==='complete'&&meta.job.kind==='flight'){await showFlight(state.pendingFlight);state.pendingFlight=null;}}finally{state.busyMeta=false;}}
async function loadGeometry(){const epoch=state.sceneEpoch,request=(state.geometryRequest||0)+1;state.geometryRequest=request;try{await cutView?.refresh();}catch(error){if(request!==state.geometryRequest||epoch!==state.sceneEpoch)return;throw error;}if(request!==state.geometryRequest||epoch!==state.sceneEpoch)return;const plane=cutView?.parameters()||{slice_axis:'both',slice_position_m:0};const query=new URLSearchParams({field:state.field,phase:state.phase,clock:state.fieldClock,...plane});let geometry;try{geometry=await api('/api/geometry?'+query);}catch(error){if(request!==state.geometryRequest||epoch!==state.sceneEpoch)return;throw error;}if(request!==state.geometryRequest||epoch!==state.sceneEpoch||geometry.revision!==state.revision)return;state.geometry=geometry;applyScene();const scales=geometry.native_field?.scales||geometry.scales;const scale=scales?.[0];
 if(scale){$('legend-title').textContent=scale.label+' · '+scale.unit;$('legend-low').textContent=fmt(scale.min,2);$('legend-high').textContent=fmt(scale.max,2);const palettes={methanol:'linear-gradient(90deg,#0d61d1,#1abaa8,#f0ae1f,#e8481e)',sequential:'linear-gradient(90deg,#1e2961,#1473ad,#1eb89c,#f7db45)',diverging:'linear-gradient(90deg,#2959bf,#e6f0fa,#d43330)'};document.querySelector('.gradient').style.background=palettes[scale.palette]||palettes.sequential;}
 const pt=geometry.phase_time_s;$('phasevalue').textContent=pt==null?' No 3D interface transient available ':fmt(pt*1e6,3)+' µs ·  Interface time scale ';$('phase-caption').textContent=pt==null?' Interface not computed ':' Interface  t = '+fmt(pt*1e6,3)+' µs ·  Transient solution ';
 const detached=(geometry.droplets||[]).filter(d=>d.detached);$('phase-drops').textContent=detached.length?' Detected  '+detached.length+'  Detached components : '+detached.slice(0,12).map(d=>fmt(d.equivalent_radius_m*1e6,2)+' µm'+(d.resolved?'':' ( Coarse-grid component )')).join('; '):' No detached droplets in this interval ,  Resolved-component statistics .';
 if(!loadGeometry.initialized){cameraPreset('whole');loadGeometry.initialized=true;}
 buildLabels();
}
function applyScene(){if(!state.geometry||!renderer)return;const geometry=state.geometry;const lines=(geometry.lines||[]).map(line=>({...line,mode:line.mode||line.primitive}));if(state.event?.track?.positions?.length>1&&!state.event.interaction_affects_flight){const colors=state.event.track.methanol.flatMap(c=>[.12+.78*c,.58-.22*c,.95-.77*c,.95]);lines.push({id:'selected-path',positions:state.event.track.positions.flat(),colors,layer:'paths',width:2});}const ih=state.event?.interaction_history;if(ih?.positions?.length>1)lines.push({id:'interaction-path',positions:ih.positions.flat(),color:[1,.55,.78,.95],layer:'paths',width:2});renderer.setScene({meshes:geometry.meshes,lines});applyLayers();}
function applyLayers(){document.querySelectorAll('[data-layer]').forEach(el=>renderer.setLayer(el.dataset.layer,el.checked));renderer.setOpacity('quartz',Number($('opacity').value)*3);}
function cameraPreset(name){state.follow=false;$('follow').classList.remove('active');renderer.setOpacity('ms',name==='tip'?.12:1);document.querySelectorAll('[data-camera]').forEach(b=>b.classList.toggle('active',b.dataset.camera===name));if(name==='whole')renderer.setCameraPreset('whole',{z_min:state.meta?.landmarks.sample_entry_z_m,z_max:state.meta?.landmarks.ms_aperture_z_m});else{const z=name==='tip'?-.00035:state.meta.landmarks.ms_aperture_z_m;renderer.focus(globalPoint([0,0,z]),name==='tip'?6.4:4.2);renderer.setCamera({yaw:.12,pitch:.13});}}
function buildLabels(){const holder=$('labels');holder.replaceChildren();if(!state.meta)return;const m=state.meta,l=m.landmarks,landmarks=[];
 if(m.path){landmarks.push([' Sample inlet ',m.path.centers_m[0]]);if(m.path.metadata.kind==='up_bend_horizontal'){let index=0;const target=m.path.metadata.up_length_m;while(index<m.path.station_m.length-1&&m.path.station_m[index]<target)index++;landmarks.push([' Bend ',m.path.centers_m[index]]);}}
 else landmarks.push([' Sample inlet ',globalPoint([0,0,l.sample_entry_z_m])]);
 landmarks.push([' Electrode  ·  Upstream  '+fmt(-l.electrode_contact_z_m*100,2)+' cm',globalPoint([0,0,l.electrode_contact_z_m])],[' Sample outlet  ·  Recess  '+fmt(-l.inner_lip_z_m*1000,2)+' mm',globalPoint([0,0,l.inner_lip_z_m])],[' Sheath outlet ',globalPoint([0,0,0])],['MS  Inlet  · 0.6 mm',globalPoint([0,0,l.ms_aperture_z_m])]);
 state.labels=landmarks.map(([text,position])=>{const el=document.createElement('div');el.className='landmark';el.textContent=text;holder.append(el);return {el,position};});
}
function updateLabels(){const occupied=[];for(const label of state.labels||[]){const p=renderer.project(label.position);let visible=p.visible&&p.x>0&&p.x<$('scene').clientWidth&&p.y>160&&p.y<$('scene').clientHeight-70;let y=p.y;for(const old of occupied)if(Math.abs(p.x-old.x)<160&&Math.abs(y-old.y)<25)y=old.y+26;if(visible){occupied.push({x:p.x,y});label.el.style.left=p.x+'px';label.el.style.top=y+'px';}label.el.hidden=!visible;}const cell=state.frame?.cells.find(c=>c.id===state.selected),el=$('selected-overlay');if(!cell){el.hidden=true;return;}const p=renderer.project(cell.position);el.hidden=!p.visible;el.style.left=p.x+'px';el.style.top=p.y+'px';el.querySelector('span').textContent='#'+state.selected+' ·  Cell position ';}
async function selectEvent(id,jump=false){if(id==null)return;const event=await api('/api/event?id='+encodeURIComponent(id));state.selected=event.event_id;state.event=event;$('eventid').value=event.event_id;$('selected-title').textContent=' Cell  #'+event.event_id;
 if(event.interaction_history&&!event.interaction_history.status)event.interaction_history.status=event.interaction_history.statuses;
 if(!state.interactionStartTouched&&state.interactionSeedSource!==state.meta?.source_id&&event.nozzle_s!=null){$('interaction-start').value=Number(clamp(event.nozzle_s-.1,0,state.meta.duration_s).toFixed(6));state.interactionSeedSource=state.meta.source_id;}
 if(jump){state.time=clamp(event.nozzle_s!=null?event.nozzle_s-.05:event.injection_s,0,state.meta.duration_s);state.playing=false;$('play').textContent=' Play ';}
 applyScene();drawExposure();await requestFrame();
}
async function requestFrame(){if(state.busyFrame||!state.meta)return;state.busyFrame=true;try{const frame=await api('/api/frame?t='+state.time.toFixed(6)+(state.selected==null?'':'&selected='+state.selected));if(frame.revision!==state.revision)return;state.frame=frame;
 const cells=frame.cells.map(c=>({...c,color:c.selected&&!String(c.interaction_status||'').includes('bound')?[1.,.91,.42,1.]:c.color}));
 const selected=frame.cells.find(c=>c.id===state.selected),charge=frame.selected_charge;
 if(selected&&charge&&!charge.extrapolated&&$('showcharge').checked){const direction=charge.polarization_direction_global||charge.polarization_direction_local;const r=selected.radius_m*Number($('magnify').value)*1.3;if(direction&&norm(direction)>.01){for(const sign of [-1,1])cells.push({id:'charge-pole-'+sign,position:selected.position.map((v,k)=>v+sign*r*direction[k]),radius_m:selected.radius_m*.28,color:sign>0?[1,.35,.25,1]:[.25,.62,1,1],stage:'charge_marker'});}}
 renderer.setCells(cells);
 $('live-count').textContent=fmt(frame.live_physical_count,0);$('display-count').textContent=fmt(frame.visible_count,0)+'  Visible '+(frame.omitted_count?' ·  Hidden  '+fmt(frame.omitted_count,0):'');
 $('conservation').innerHTML=Object.entries({" Injected ":frame.counts.created," In transit ":frame.counts.in_transit," Passed sheath outlet ":frame.counts.crossed_lip," Capillary wall binding ":frame.counts.wall_adhered??0," Unresolved or stopped ":frame.counts.transport_unresolved}).map(([k,v])=>'<dt>'+k+'</dt><dd>'+fmt(v,0)+'</dd>').join('');
 const local=frame.interaction_study;if(local){const names={mobile:' Free transport ',cell_bound:' Cell-cell binding ',wall_bound:' Wall binding ',outlet_bound:' Outlet binding ',exited:' Exited ',censored:' Window ended ',not_entered:' Before local domain '};$('interaction-counts').innerHTML='<dt> Current time </dt><dd>'+(local.active?' Within local domain ':' Outside local domain ')+'</dd>'+Object.entries(local.counts||{}).map(([k,v])=>'<dt>'+esc(names[k]||k)+'</dt><dd>'+fmt(v,0)+'</dd>').join('');}
 updateInspector();showCharge(charge);drawGates();cutView?.updateFrame();if(state.follow){const c=frame.cells.find(c=>c.id===state.selected);if(c)renderer.focus(c.position,renderer.camera.distance);}
 }finally{state.busyFrame=false;}}
const statuses={pure_complete:' Complete single-source ',pure_partial:' Partial single-source ',pure_impulse:' Single-source impulse ',mixed_only:' Mixed signal ',missed:' Missed gate ',not_delivered:' Not delivered  ( Specified )'};
const interactionNames={mobile:' Mobile ',cell_bound:' Cell-cell binding ',wall_bound:' Bound to wall ',outlet_bound:' Outlet binding ',bulk_wall_bound:' Irreversible capillary capture ',exited:' Local outlet crossed ',censored:' Incomplete local transit '};
function showCharge(q){if(!q){$('charge-info').innerHTML='<dt> Charge model </dt><dd> Select a cell in the capillary </dd>';return;}const values={'ζ  Surface-equivalent charge ':fmt(q.q_zeta_c*1e12,4)+' pC',' Electrophoretic charge ':fmt(q.q_electrophoretic_effective_c*1e15,4)+' fC',' Positive  /  Negative polarization pole ':'+ / − '+fmt(q.positive_pole_charge_c*1e15,4)+' fC',' Induced pole voltage ':fmt(q.membrane_pole_voltage_v*1000,3)+' mV','Debye  Length ':fmt(q.debye_length_m*1e9,3)+' nm',' Local conductivity ':fmt(q.medium_conductivity_s_m,5)+' S/m',' Model scope ':q.extrapolated?' Outlet state ,  Tip domain ':q.validity?.membrane_voltage_within_linear_guard?' Linear charging model ':' Membrane voltage exceeds the linear range '};if(q.validity?.linear_dh_small_potential===false)values['ζ  Surface charge approximation ']=' Low-potential approximation ';$('charge-info').innerHTML=Object.entries(values).map(([k,v])=>'<dt>'+k+'</dt><dd>'+esc(v)+'</dd>').join('');}
async function showFlight(id){const value=await api('/api/flight?id='+id),r=value.result;state.flight=value;const labels={pre_nozzle_shell_model_lost:' Membrane model ended before exit ,  Membrane continuum limit ',membrane_continuum_lost:' Membrane model ended during transfer ',entered_aperture:' Entered inlet aperture ',impacted_inlet_face:' Hit inlet face ',rayleigh_instability_unresolved:' Reached  Rayleigh  Instability ,  Daughter droplets unresolved ',coating_depleted_unresolved:' Carrier coating depleted ,  Continuation unresolved ',flight_timeout:' Transfer time limit ',pore_area_model_out_of_range:' Aperture outside model domain '};$('flight-info').textContent=(labels[r.status]||r.status||' Carrier transfer result ')+'. Transfer  '+fmt(r.flight_time_ms,3)+' ms,  Membrane fraction  '+fmt(r.membrane_lipid_remaining_fraction*100,1)+'%,  Carrier charge  '+fmt(r.carrier_charge_c*1e15,3)+' fC. Aperture entry metric .';if(value.history?.length){const path={id:'conditional-flight',positions:value.history.flatMap(row=>row.position),color:[.98,.65,.25,.9],layer:'paths'};state.geometry.lines=state.geometry.lines.filter(l=>l.id!=='conditional-flight');state.geometry.lines.push(path);applyScene();}await selectEvent(id);}
function updateInspector(){const e=state.event;if(!e)return;const c=state.frame?.cells.find(c=>c.id===state.selected);const age=c?.age_s??(state.time-e.injection_s),stage=c?.stage;
 $('selected-stage').textContent=c?.interaction_status?(interactionNames[c.interaction_status]||c.interaction_status):stage==='long_pipe'?' Capillary transit ':stage==='resolved_tip'?' Resolved coaxial tip ':stage==='conditional_lipid_carrier'?' Lipid-carrier transfer ':age<0?' Not injected ':e.exited&&age>=e.end_age_s?' Exited capillary  ·  Downstream continuation ':' Incomplete  /  Transport stopped ';
 const values={' Injection time ':fmt(e.injection_s,3)+' s',' Sheath outlet arrival ':e.nozzle_s==null?' Not reached ':fmt(e.nozzle_s,3)+' s',' Elapsed time ':fmt(Math.max(0,age),3)+' s',' External methanol exposure ':c?fmt(c.methanol*100,1)+'%':'—',' Acquisition outcome ':statuses[e.status]||' Outside observation cohort '};
 if(e.interaction_affects_flight){values[' Independent outlet time ']=values[' Sheath outlet arrival '];delete values[' Sheath outlet arrival '];values[' Coupled outlet time ']=e.interaction_history?.exit_s==null?' No exit in this window ':fmt(e.interaction_history.exit_s,6)+' s';values[' Acquisition outcome ']=' Reference : '+values[' Acquisition outcome '];}
 if(c?.position){values[' Physical  X / Y / Z']=c.position.map(x=>fmt(x*1000,3)).join(' / ')+' mm';}
 if(c?.carrier_charge_c!=null)values[' Carrier net charge ']=fmt(c.carrier_charge_c*1e15,3)+' fC';
 if(c?.interaction_status){values[' Interaction state ']=interactionNames[c.interaction_status]||c.interaction_status;if(c.cluster_id!=null)values[' Cluster ID ']=c.cluster_id;}
 const contacts=c?.interaction_contact_code??c?.contact_code;if(contacts!=null){const names=[];if(contacts&1)names.push(' Other cells ');if(contacts&2)names.push(' Wall ');if(contacts&4)names.push(' Outlet ');values[' Current contact ']=names.length?names.join(', '):' None ';}
 if(e.interaction_history?.events?.length)values[' Local event history ']=e.interaction_history.events.length+'  Entries  ( Includes binding ,  Unbinding and boundary events )';
 if(e.interaction_affects_flight)values[' Acquisition  /  Transfer domain ']=' Local trajectory updated ;  Recompute downstream transfer for this trajectory ';
 if(e.captured_fraction!=null)values[' Packet coverage ']=fmt(e.captured_fraction*100,1)+'%';
 $('cell-info').innerHTML=Object.entries(values).map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>').join('');
 $('flight').disabled=state.meta?.job.state==='running'||!e.exited||e.interaction_affects_flight;
 drawExposure();
}
function fitCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(2,devicePixelRatio||1);const w=Math.max(10,Math.round(rect.width*dpr)),h=Math.max(10,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return [ctx,rect.width,rect.height];}
function drawExposure(){const canvas=$('exposure'),[ctx,w,h]=fitCanvas(canvas);ctx.clearRect(0,0,w,h);const e=state.event;if(!e?.track)return;const ih=e.interaction_history,local=e.interaction_affects_flight&&ih?.time_s?.length>1;const origin=local?ih.time_s[0]:e.injection_s,ts=local?ih.time_s.map(t=>t-origin):e.track.time_s,cs=local?ih.methanol:e.track.methanol,max=Math.max(1e-6,ts.at(-1));ctx.font='9px Segoe UI';ctx.strokeStyle='#2b4652';ctx.beginPath();ctx.moveTo(25,12);ctx.lineTo(25,h-22);ctx.lineTo(w-8,h-22);ctx.stroke();ctx.fillStyle='#6d99ac';ctx.fillText('100%',0,15);ctx.fillText('0',11,h-22);ctx.fillText((local?' Local  +':'')+fmt(max,2)+' s',w-75,h-6);ctx.strokeStyle=local?'#f98fbe':'#eeae66';ctx.lineWidth=1.5;ctx.beginPath();ts.forEach((t,i)=>{const x=25+t/max*(w-35),y=h-22-clamp(cs?.[i]||0,0,1)*(h-36);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();const age=clamp(state.time-origin,0,max),x=25+age/max*(w-35);ctx.strokeStyle='#48d8c2';ctx.beginPath();ctx.moveTo(x,10);ctx.lineTo(x,h-22);ctx.stroke();}
function drawGates(){const [ctx,w,h]=fitCanvas($('gates'));ctx.clearRect(0,0,w,h);if(!state.meta||!state.frame)return;
 const timing=state.meta.timing,cycle=timing.record_cycle_ms*.001,fill=timing.actual_injection_ms*.001,micro=timing.microscan_period_ms*.001;
 const span=Math.max(.25,cycle*3.5),start=state.time-span*.42,end=start+span,x=t=>(t-start)/span*w;
 ctx.fillStyle='#142932';ctx.fillRect(0,3,w,12);for(let i=Math.floor(start/cycle);i<=Math.ceil(end/cycle);i++){for(let j=0;j<state.meta.acquisition.microscans;j++){const a=i*cycle+j*micro;ctx.fillStyle='#206c5e';ctx.fillRect(x(a),3,Math.max(1,x(a+fill)-x(a)),12);}}
 const width=state.frame.acquisition.packet_width_s;for(const p of state.frame.packets){if(p.onset_s+width<start||p.onset_s>end)continue;ctx.fillStyle=p.event_id===state.selected?'#f1bb64':p.delivered?'#488fb5':'#5d6b73';ctx.fillRect(x(p.onset_s),21+(p.event_id%2)*7,Math.max(2,width/span*w),5);}
 const center=x(state.time);ctx.strokeStyle='#c5f2e8';ctx.beginPath();ctx.moveTo(center,0);ctx.lineTo(center,h);ctx.stroke();ctx.font='8px Segoe UI';ctx.fillStyle='#7cabbc';ctx.fillText(fmt(start,3)+' s',2,h-1);ctx.fillText(fmt(end,3)+' s',w-56,h-1);
}
async function nextEvent(trip=false){state.studyPlaybackEnd=null;const nextTime=state.time>=state.meta.duration_s-.2?0:state.time;const e=await api('/api/event?'+(trip?'trip=1':'next='+nextTime));await selectEvent(e.event_id);state.time=clamp(trip?e.injection_s:e.nozzle_s-.05,0,state.meta.duration_s);state.playing=true;state.follow=trip;state.autoTrip=trip;$('play').textContent=' Pause ';$('follow').classList.toggle('active',trip);if(trip){$('speed').value='10';renderer.focus(e.track.positions[0],4.5);}else{$('speed').value='.1';cameraPreset('tip');}await requestFrame();}
function nativeRequest(){const get=id=>Number($(id).value);return {length_cm:get('length'),sample_pressure_mbar:get('pressure'),sample_control:'pressure',sheath_flow_ul_min:get('sheath'),voltage_kv:get('voltage'),gas_pressure_kpa:get('gaspressure'),diameter_um:get('diameter'),concentration_per_ml:get('concentration'),cell_zeta_mv:get('cellzeta'),current_mode:$('currentmode').value,spray_current_na:$('current').value===''?null:get('current')*1000,sheath_conductivity_s_m:get('conductivity'),quartz_zeta_mv:get('quartzzeta'),brownian_enabled:$('brownian').value==='auto'?null:$('brownian').value==='on',cells:get('cells'),bend_radius_mm:get('bend'),up_fraction:get('upfraction')/100,path_kind:$('custompath').checked?'custom':'default',points_mm:$('waypoints').value.trim().split('\n').filter(Boolean).map(row=>row.trim().split(/[,\s]+/).map(Number)),phase_enabled:$('phaseenabled').checked,phase:{duration_us:get('phaseduration'),max_steps:get('phasesteps')},volume:{refinement:get('refinement'),axial_cells:get('axialcells'),flow_steps:get('flowsteps'),coupling_iterations:get('coupling'),gas_asymmetry:get('asymmetry')}};}
const requestWithGeometry=nativeRequest;
nativeRequest=function(){const body=requestWithGeometry();body.geometry={};document.querySelectorAll('[data-geometry]').forEach(el=>body.geometry[el.dataset.geometry]=Number(el.value));body.contact_distance_mm=Number($('contact-distance').value);body.bulk_wall_capture_rate_s=Number($('bulk-wall-rate').value);body.bulk_wall_capture_gap_um=Number($('bulk-wall-gap').value);return body;};
async function initialize(){
 renderer=new SceneRenderer($('scene'),{onPick:id=>{if(Number.isInteger(id))safeRun(()=>selectEvent(id))();}});renderer.setTransform({cellExaggeration:2});
 cutView=new SliceView({state,api,changed:loadGeometry,onSelect:id=>safeRun(()=>selectEvent(id))(),toast});
 document.querySelectorAll('[data-camera]').forEach(el=>el.addEventListener('click',()=>cameraPreset(el.dataset.camera)));
 document.querySelectorAll('[data-layer]').forEach(el=>el.addEventListener('change',applyLayers));
 document.querySelectorAll('[data-current]').forEach(el=>el.addEventListener('click',()=>{$('current').value=el.dataset.current;}));
 $('currentmode').addEventListener('change',()=>{$('current').disabled=$('currentmode').value==='scaling';});
 $('opacity').addEventListener('input',applyLayers);
 $('magnify').addEventListener('input',()=>{$('magvalue').textContent=$('magnify').value+'×';renderer.setTransform({cellExaggeration:Number($('magnify').value)});requestFrame().catch(e=>toast(e.message));});
 $('lens').addEventListener('change',()=>{setMapping();cameraPreset('whole');});
 $('field').addEventListener('change',safeRun(async()=>{state.field=$('field').value;await loadGeometry();}));
 $('fieldclock').addEventListener('change',safeRun(async()=>{state.fieldClock=$('fieldclock').value;await loadGeometry();}));
 $('showcharge').addEventListener('change',safeRun(requestFrame));
 $('export-png').addEventListener('click',()=>{renderer.render();const source=$('scene'),canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height;const ctx=canvas.getContext('2d');ctx.drawImage(source,0,0);const ratio=source.width/source.clientWidth;ctx.scale(ratio,ratio);const w=source.clientWidth,h=source.clientHeight;ctx.fillStyle='#071219ee';ctx.fillRect(0,h-74,w,74);ctx.fillStyle='#dceaf0';ctx.font='11px Segoe UI';ctx.fillText('CyESI · t='+state.time.toFixed(3)+' s · '+$('legend-title').textContent+' ['+$('legend-low').textContent+'–'+$('legend-high').textContent+']',14,h-54);ctx.fillStyle='#9ebac8';ctx.fillText(($('lens').checked?' Axial display lens ':' Equal physical scale ')+' ·  Cell magnification  '+$('magnify').value+'× · '+$('phase-caption').textContent,14,h-34);ctx.fillText('source='+state.meta.source_id+' ·  Cartesian tip  /  Capillary transport  ·  Research workspace ',14,h-14);const a=document.createElement('a');a.download='CyESI_3D_'+state.time.toFixed(3)+'s.png';a.href=canvas.toDataURL('image/png');a.click();});
 $('export-csv').addEventListener('click',()=>{const e=state.event;if(!e)return toast(' Select a cell event ');const ih=e.interaction_history,local=e.interaction_affects_flight&&ih?.positions?.length,charge=e.charge_history;const header=local?['event_id','absolute_time_s','x_m','y_m','z_m','methanol_fraction','interaction_status','model_scope']:['event_id','age_s','x_m','y_m','z_m','methanol_fraction','q_zeta_c','q_ep_effective_c','membrane_pole_voltage_v'];const rows=local?ih.time_s.map((t,i)=>[e.event_id,t,...ih.positions[i],ih.methanol?.[i]??'',ih.status?.[i]??'','local_continuous_interaction_scenario']):e.track.time_s.map((t,i)=>[e.event_id,t,...e.track.positions[i],e.track.methanol[i],charge?.q_zeta_c?.[i]??'',charge?.q_electrophoretic_effective_c?.[i]??'',charge?.membrane_pole_voltage_v?.[i]??'']);const blob=new Blob(['\ufeff'+[header,...rows].map(r=>r.join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.download='CyESI_cell_'+e.event_id+(local?'_interaction':'')+'.csv';a.href=url;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);});
 $('charge-parameters').addEventListener('click',safeRun(async()=>{await api('/api/charge-parameters',{membrane_capacitance_f_m2:Number($('membranecap').value),intracellular_conductivity_s_m:Number($('internalconductivity').value)});await refreshMeta();await requestFrame();}));
 $('phasetime').addEventListener('change',safeRun(async()=>{state.phase=Number($('phasetime').value);await loadGeometry();}));
 $('play').addEventListener('click',()=>{if(!state.playing&&state.meta&&state.time>=state.meta.duration_s)state.time=0;state.playing=!state.playing;$('play').textContent=state.playing?' Pause ':' Play ';});
 $('speed').addEventListener('change',()=>{state.autoTrip=false;});
 $('time').addEventListener('input',safeRun(async()=>{state.studyPlaybackEnd=null;state.time=Number($('time').value);state.playing=false;$('play').textContent=' Play ';await requestFrame();}));
 $('selectevent').addEventListener('click',safeRun(()=>selectEvent(Number($('eventid').value),true)));
 $('seek-event').addEventListener('click',safeRun(async()=>{if(state.event?.nozzle_s==null)return toast(' Select an event with an outlet time ');state.time=clamp(state.event.nozzle_s+Number($('event-offset').value)*.001,0,state.meta.duration_s);state.playing=false;state.autoTrip=false;$('play').textContent=' Play ';cameraPreset('tip');await requestFrame();}));
 $('next').addEventListener('click',safeRun(()=>nextEvent(false)));$('wholetrip').addEventListener('click',safeRun(()=>nextEvent(true)));
 $('follow').addEventListener('click',()=>{state.follow=!state.follow;$('follow').classList.toggle('active',state.follow);const c=state.frame?.cells.find(c=>c.id===state.selected);if(state.follow&&c)renderer.focus(c.position,4);});
 $('run').addEventListener('click',safeRun(async()=>{await api('/api/run3d',nativeRequest());await refreshMeta();}));
 $('cancel').addEventListener('click',safeRun(async()=>{await api('/api/cancel',{});$('job').textContent=' Stopping at the next solver checkpoint …';}));
 $('compare-voltage').addEventListener('click',safeRun(async()=>{await api('/api/compare-voltage',{});$('voltage-info').textContent=' Computing at matched supply flow  0 kV  Compare ,  Comparing electroosmosis and interface evolution .';await refreshMeta();}));
 $('interaction-start').addEventListener('input',()=>{state.interactionStartTouched=true;});
 $('interaction-now').addEventListener('click',()=>{$('interaction-start').value=Number(state.time.toFixed(3));state.interactionStartTouched=true;});
 $('interaction-run').addEventListener('click',safeRun(async()=>{await api('/api/interactions',{start_s:Number($('interaction-start').value),duration_s:Number($('interaction-duration').value),dt_s:Number($('interaction-dt').value)*.001,hydrodynamics:$('pair-hydro').checked,contact_gap_um:Number($('contact-gap').value),cell_adhesion_on_rate_s:Number($('cell-bind-rate').value),wall_adhesion_on_rate_s:Number($('wall-bind-rate').value),outlet_adhesion_on_rate_s:Number($('outlet-bind-rate').value),adhesion_off_rate_s:Number($('release-rate').value),adhesion_force_scale_pn:Number($('adhesion-force').value)});$('interaction-info').textContent=' Computing local interactions ,  Current 3D background field .';await refreshMeta();}));
 $('interaction-jump').addEventListener('click',safeRun(async()=>{const study=state.meta?.interaction_study,p=study?.parameters;if(!p)return;state.time=clamp(Number(p.start_s),0,state.meta.duration_s);state.studyPlaybackEnd=Number(study.diagnostics?.end_s??Number(p.start_s)+Number(p.duration_s));state.playing=true;state.autoTrip=false;$('speed').value='.1';$('play').textContent=' Pause ';cameraPreset('tip');await requestFrame();}));
 $('interaction-clear').addEventListener('click',safeRun(async()=>{state.studyPlaybackEnd=null;await api('/api/interactions',{clear:true});await refreshMeta();await requestFrame();}));
 $('load').addEventListener('click',safeRun(async()=>{await api('/api/load',{id:$('records').value});await refreshMeta();}));
 $('method').addEventListener('click',safeRun(async()=>{await api('/api/method',{resolution:Number($('resolution').value),actual_it_ms:Number($('it').value),packet_width_ms:Number($('packet').value)});await refreshMeta();}));
 $('flight').addEventListener('click',safeRun(async()=>{if(state.selected==null)return;state.pendingFlight=state.selected;await api('/api/flight',{id:state.selected});$('flight-info').textContent=' Computing carrier transfer . Carrier size ,  Charge ,  Editable membrane parameters .';await refreshMeta();}));
 window.addEventListener('resize',()=>{renderer.resize();drawExposure();drawGates();});
 const records=await api('/api/records');for(const record of records){const option=document.createElement('option');option.value=record.id;option.textContent=record.label;$('records').append(option);}
 await refreshMeta();renderer.resize();requestAnimationFrame(tick);
}
let lastTick=0;
function tick(now){const dt=Math.min(.1,(now-lastTick)/1000||0);lastTick=now;if(state.playing&&state.meta){let increment=dt*Number($('speed').value);if(state.autoTrip&&state.event){const remaining=state.event.nozzle_s-state.time,rate=remaining<.3?.01:remaining<3?1:10;$('speed').value=String(rate);increment=dt*rate;if(remaining>.3&&increment>remaining-.3)increment=remaining-.3;if(remaining<-.03){state.playing=false;$('play').textContent=' Play ';} $('scene-state').textContent=' Follow  #'+state.selected+' · '+rate+'×  Playback speed  ·  Continuous simulation clock ';}state.time+=increment;if(state.studyPlaybackEnd!=null&&state.time>=state.studyPlaybackEnd){state.time=state.studyPlaybackEnd;state.studyPlaybackEnd=null;state.playing=false;$('play').textContent=' Play ';}if(state.time>=state.meta.duration_s){state.time=state.meta.duration_s;state.playing=false;$('play').textContent=' Play ';}}
 $('clock').textContent=fmt(state.time,3)+' s';$('time').value=state.time;
 const frameChanged=!state.frame||Math.abs(state.frame.time_s-state.time)>1e-6||state.frame.selected_id!==state.selected||state.frame.revision!==state.revision;
 if(frameChanged&&now-state.lastFrame>(state.follow?50:100)){state.lastFrame=now;requestFrame().catch(e=>toast(e.message));}
 if(now-state.lastMeta>1500){state.lastMeta=now;refreshMeta().catch(e=>toast(e.message));}
 renderer.render();updateLabels();requestAnimationFrame(tick);
}
initialize().catch(e=>{toast(e.message);$('job').textContent=' Startup failed : '+e.message;});
