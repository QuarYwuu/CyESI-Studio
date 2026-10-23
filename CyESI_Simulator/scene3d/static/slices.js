// Slice cell-constant fields and finite spheres at their physical coordinates.
const $=id=>document.getElementById(id);
const fmt=(v,d=2)=>Number(v).toLocaleString('en-GB',{maximumFractionDigits:d});
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const palettes={methanol:[[.05,.38,.82],[.1,.73,.72],[.94,.68,.12],[.91,.28,.12]],diverging:[[.16,.35,.75],[.9,.94,.98],[.83,.2,.19]],sequential:[[.12,.16,.38],[.08,.45,.68],[.12,.72,.61],[.97,.86,.27]]};
function color(value,scale){const knots=palettes[scale.palette]||palettes.sequential,t=clamp((value-scale.min)/(scale.max-scale.min),0,1)*(knots.length-1),i=Math.min(knots.length-2,Math.floor(t)),f=t-i;return 'rgb('+knots[i].map((v,k)=>Math.round(255*(v+(knots[i+1][k]-v)*f))).join(',')+')';}
function locate(edges,value){let a=0,b=edges.length-1;if(value<edges[a]||value>edges[b])return -1;while(b-a>1){const m=(a+b)>>1;if(value>=edges[m])a=m;else b=m;}return Math.min(a,edges.length-2);}
export class SliceView{
 constructor({state,api,changed,onSelect,toast}){
  Object.assign(this,{state,api,changed,onSelect,toast,axis:'x',position_m:0,visible:false,payload:null,epoch:0,zoom:null});
  $('cut-toggle').addEventListener('click',()=>{if(this.state.meta?.mode!=='native_cartesian_tip')return this.toast(' Load or compute a Cartesian volume field ,  to inspect physical cross-sections .');this.visible=!this.visible;$('cut-panel').hidden=!this.visible;$('cut-toggle').classList.toggle('active',this.visible);this.changed().catch(e=>this.toast(e.message));});
  $('cut-close').addEventListener('click',()=>$('cut-toggle').click());
  $('cut-axis').addEventListener('change',()=>{this.axis=$('cut-axis').value;this.position_m=0;this.zoom=null;this.position_m=this.clampedPosition(this.position_m);this.commit();});
  $('cut-position').addEventListener('change',()=>{this.position_m=this.clampedPosition(Number($('cut-position').value)*1e-6);this.commit();});
  $('cut-slider').addEventListener('input',()=>{this.position_m=Number($('cut-slider').value);$('cut-position').value=fmt(this.position_m*1e6,4).replaceAll(',','');clearTimeout(this.timer);this.timer=setTimeout(()=>this.commit(),130);});
  $('cut-equal').addEventListener('change',()=>this.draw());$('cut-vectors').addEventListener('change',()=>this.draw());
  $('cut-reset').addEventListener('click',()=>{this.zoom=null;this.draw();});
  $('cut-selected').addEventListener('click',()=>{const c=this.state.frame?.cells.find(c=>c.id===this.state.selected);if(!c?.local_position_m||c.stage==='long_pipe')return this.toast(' Select a cell in the resolved domain ,  Move the section through its centre .');this.position_m=this.clampedPosition(c.local_position_m['xyz'.indexOf(this.axis)]);this.commit();});
  $('cut-full').addEventListener('click',()=>{$('cut-panel').classList.toggle('expanded');$('cut-full').textContent=$('cut-panel').classList.contains('expanded')?' Reconstructed ':' Expand ';this.draw();});
  $('cut-export').addEventListener('click',()=>{this.draw();const source=$('cut-canvas'),canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height+76;const c=canvas.getContext('2d');c.fillStyle='#091720';c.fillRect(0,0,canvas.width,canvas.height);c.drawImage(source,0,0);c.fillStyle='#adccd7';c.font='12px Segoe UI';c.fillText('source='+this.payload.source_id+' · main t='+this.state.time.toFixed(6)+' s · '+this.payload.data_clock+(this.payload.field_time_s==null?'':' t='+this.payload.field_time_s+' s'),16,source.height+22);c.fillText($('cut-aspect').textContent,16,source.height+44);c.fillText('True finite-sphere cuts; no cell magnification; field values from the containing FV slab.',16,source.height+64);const a=document.createElement('a');a.download='CyESI_cut_'+this.axis+'_'+fmt(this.position_m*1e6,2)+'um.png';a.href=canvas.toDataURL('image/png');a.click();});
  const canvas=$('cut-canvas');canvas.addEventListener('mousemove',event=>this.hover(event));canvas.addEventListener('mouseleave',()=>$('cut-tooltip').hidden=true);
  canvas.addEventListener('click',event=>{const p=this.pointer(event);if(!p)return;for(const c of this.intersections||[]){const dx=(c.h-p.h)/this.plot.hs,dy=(c.v-p.v)/this.plot.vs;if(Math.hypot(dx,dy)<=Math.max(5,c.radius/Math.min(this.plot.hs,this.plot.vs))){this.onSelect(c.id);break;}}});
  canvas.addEventListener('wheel',event=>{if(!this.plot)return;event.preventDefault();const p=this.pointer(event);if(!p)return;const factor=event.deltaY>0?1.25:.8,b=this.zoom||this.fullBounds();this.zoom=[p.h+(b[0]-p.h)*factor,p.h+(b[1]-p.h)*factor,p.v+(b[2]-p.v)*factor,p.v+(b[3]-p.v)*factor];this.draw();},{passive:false});
  canvas.addEventListener('dblclick',()=>{this.zoom=null;this.draw();});
  this.observer=new ResizeObserver(()=>this.draw());this.observer.observe($('cut-panel'));
 }
 updateMeta(meta){$('cut-toggle').disabled=meta.mode!=='native_cartesian_tip';if(meta.mode!=='native_cartesian_tip'&&this.visible){this.visible=false;$('cut-panel').hidden=true;$('cut-toggle').classList.remove('active');} }
 clampedPosition(value){const bounds=this.payload?.axis_ranges_m?.[this.axis];return bounds?clamp(value,...bounds):value;}
 async commit(){this.epoch++;try{await this.changed();}catch(e){this.toast(e.message);}}
 parameters(){return this.visible?{slice_axis:this.axis,slice_position_m:this.position_m}:{slice_axis:'both',slice_position_m:0};}
 async refresh(){
  if(!this.visible||this.state.meta?.mode!=='native_cartesian_tip')return;
  const s=this.state,key=[s.revision,s.field,s.fieldClock,s.phase,this.axis,this.position_m].join('|');
  if(key===this.key){this.draw();return;}
  const epoch=++this.epoch;
  $('cut-summary').textContent=' Loading a section through the 3D mesh …';
  const q=new URLSearchParams({field:s.field,axis:this.axis,position:String(this.position_m),clock:s.fieldClock,phase:String(s.phase)});
  let data;try{data=await this.api('/api/slice?'+q);}catch(error){if(!String(error.message).includes('outside'))throw error;this.position_m=0;q.set('position','0');data=await this.api('/api/slice?'+q);this.toast(' Field domain changed ,  Section reset to  0 µm.');}
  if(epoch!==this.epoch)return;
  if(this.payload?.source_id!==data.source_id)this.zoom=null;
  this.payload=data;this.key=key;
  const bounds=data.axis_ranges_m[this.axis],slider=$('cut-slider');slider.min=bounds[0];slider.max=bounds[1];slider.step=(bounds[1]-bounds[0])/2000;slider.value=this.position_m;
  $('cut-position').min=bounds[0]*1e6;$('cut-position').max=bounds[1]*1e6;$('cut-position').value=String(Number((this.position_m*1e6).toFixed(4)));
  this.draw();
 }
 fullBounds(){const p=this.payload;return[p.horizontal_edges_m[0],p.horizontal_edges_m.at(-1),p.vertical_edges_m[0],p.vertical_edges_m.at(-1)];}
 updateFrame(){if(this.visible)this.draw();}
 pointer(event){const p=this.plot;if(!p)return null;const rect=$('cut-canvas').getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;if(x<p.x||x>p.x+p.w||y<p.y||y>p.y+p.h)return null;return {x,y,h:p.bounds[0]+(x-p.x)*p.hs,v:p.bounds[3]-(y-p.y)*p.vs};}
 hover(event){const point=this.pointer(event),tip=$('cut-tooltip'),p=this.payload;if(!point||!p){tip.hidden=true;return;}const col=locate(p.horizontal_edges_m,point.h),row=locate(p.vertical_edges_m,point.v);if(row<0||col<0){tip.hidden=true;return;}const value=p.values[row][col],kind=[' Gas ',' Liquid ',' Quartz '][p.material[row][col]];tip.textContent=p.horizontal_axis+'='+fmt(point.h*1e6,2)+' µm; '+p.vertical_axis+'='+fmt(point.v*1e6,2)+' µm · '+kind+(value==null?' ·  Field masked in this material ':' · '+fmt(value,4)+' '+p.scale.unit);tip.hidden=false;}
 draw(){
  if(!this.visible||!this.payload)return;
  const canvas=$('cut-canvas'),rect=canvas.getBoundingClientRect();if(rect.width<10||rect.height<10)return;
  const dpr=Math.min(2,devicePixelRatio||1),w=rect.width,h=rect.height;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.fillStyle='#091720';c.fillRect(0,0,w,h);
  const p=this.payload,b=this.zoom||this.fullBounds();let box={x:57,y:28,w:w-82,h:h-89};
  if($('cut-equal').checked){const scale=Math.min(box.w/(b[1]-b[0]),box.h/(b[3]-b[2])),ww=(b[1]-b[0])*scale,hh=(b[3]-b[2])*scale;box.x+=(box.w-ww)/2;box.y+=(box.h-hh)/2;box.w=ww;box.h=hh;}
  this.plot={...box,bounds:b,hs:(b[1]-b[0])/box.w,vs:(b[3]-b[2])/box.h};
  const X=v=>box.x+(v-b[0])/(b[1]-b[0])*box.w,Y=v=>box.y+(b[3]-v)/(b[3]-b[2])*box.h;
  c.save();c.beginPath();c.rect(box.x,box.y,box.w,box.h);c.clip();c.fillStyle='#0c1c25';c.fillRect(box.x,box.y,box.w,box.h);
  for(let row=0;row<p.shape[0];row++)for(let col=0;col<p.shape[1];col++){
   const x0=X(p.horizontal_edges_m[col]),x1=X(p.horizontal_edges_m[col+1]),y0=Y(p.vertical_edges_m[row+1]),y1=Y(p.vertical_edges_m[row]);if(x1<box.x||x0>box.x+box.w||y1<box.y||y0>box.y+box.h)continue;
   c.fillStyle=p.values[row][col]==null?(p.material[row][col]===2?'#637d8a':p.material[row][col]===0?'#102b3b':'#133c3c'):color(p.values[row][col],p.scale);c.fillRect(x0,y0,x1-x0+.25,y1-y0+.25);if(p.material[row][col]===2&&p.values[row][col]!=null){c.strokeStyle='#e3f1fa66';c.lineWidth=.5;c.beginPath();c.moveTo(x0,y1);c.lineTo(x1,y0);c.stroke();}
  }
  if($('cut-vectors').checked){const max=Math.max(1e-30,...p.vectors.map(v=>Math.hypot(v[2]/this.plot.hs,v[3]/this.plot.vs)));c.strokeStyle='#f7f0bdce';c.lineWidth=1;for(const [h0,v0,dh,dv] of p.vectors){const magnitude=Math.hypot(dh,dv);if(magnitude<=0)continue;const x=X(h0),y=Y(v0),dx=dh/this.plot.hs/max*16,dy=-dv/this.plot.vs/max*16;c.beginPath();c.moveTo(x,y);c.lineTo(x+dx,y+dy);const a=Math.atan2(dy,dx);c.moveTo(x+dx-Math.cos(a-.5)*4,y+dy-Math.sin(a-.5)*4);c.lineTo(x+dx,y+dy);c.lineTo(x+dx-Math.cos(a+.5)*4,y+dy-Math.sin(a+.5)*4);c.stroke();}}
  this.intersections=[];let selectedOffPlane=null;const normal='xyz'.indexOf(this.axis),horizontal='xyz'.indexOf(p.horizontal_axis),vertical='xyz'.indexOf(p.vertical_axis);
  for(const cell of this.state.frame?.cells||[]){const xyz=cell.local_position_m;if(!xyz)continue;const R=cell.radius_m,d=xyz[normal]-this.position_m;if(cell.id===this.state.selected&&Math.abs(d)>R)selectedOffPlane=Math.abs(d);if(Math.abs(d)>R)continue;const radius=Math.sqrt(Math.max(0,R*R-d*d)),hc=xyz[horizontal],vc=xyz[vertical];if(hc+radius<b[0]||hc-radius>b[1]||vc+radius<b[2]||vc-radius>b[3])continue;const selected=cell.id===this.state.selected,bound=String(cell.interaction_status||'').includes('bound');c.beginPath();c.ellipse(X(hc),Y(vc),Math.max(.5,radius/this.plot.hs),Math.max(.5,radius/this.plot.vs),0,0,2*Math.PI);c.fillStyle=bound?'#f789bdc4':selected?'#ffe07cbf':'#dcfff985';c.fill();c.strokeStyle=selected?'#ffe994':bound?'#ff86c2':'#ebffff';c.lineWidth=selected?1.8:1.;c.stroke();if(selected||bound){c.font='9px Segoe UI';c.fillStyle=selected?'#ffe994':'#ffb7d8';c.fillText('#'+cell.id+(bound?'  Binding ':''),X(hc)+5,Y(vc)-5);}this.intersections.push({id:cell.id,h:hc,v:vc,radius});}
  c.restore();c.strokeStyle='#688d9b';c.lineWidth=.8;c.strokeRect(box.x,box.y,box.w,box.h);c.font='9px Segoe UI';c.fillStyle='#9fc2cf';
  for(let i=0;i<=4;i++){const x=box.x+box.w*i/4,y=box.y+box.h*i/4;c.textAlign='center';c.fillText(fmt((b[0]+(b[1]-b[0])*i/4)*1e6,1),x,box.y+box.h+15);c.textAlign='right';c.fillText(fmt((b[3]-(b[3]-b[2])*i/4)*1e6,1),box.x-6,y+3);}
  c.textAlign='center';c.fillText(p.horizontal_axis+' (µm)',box.x+box.w/2,box.y+box.h+31);c.save();c.translate(12,box.y+box.h/2);c.rotate(-Math.PI/2);c.fillText(p.vertical_axis+' (µm)',0,0);c.restore();
  c.textAlign='left';c.fillStyle='#d8e9eb';c.font='10px Segoe UI';c.fillText(p.scale.label+' ('+p.scale.unit+') · '+this.axis+'='+fmt(this.position_m*1e6,2)+' µm',12,16);
  const barY=h-20,barX=57,barW=Math.max(30,w-150);for(let i=0;i<barW;i++){c.fillStyle=color(p.scale.min+(p.scale.max-p.scale.min)*i/barW,p.scale);c.fillRect(barX+i,barY,1.5,5);}c.fillStyle='#9fc2cf';c.font='9px Segoe UI';c.fillText(fmt(p.scale.min,3),barX,barY+16);c.textAlign='right';c.fillText(fmt(p.scale.max,3),barX+barW,barY+16);c.textAlign='left';c.fillStyle='#637d8a';c.fillRect(w-79,barY-1,8,8);c.fillStyle='#9fc2cf';c.fillText(' Quartz ',w-67,barY+6);
  const clock=p.data_clock==='phase'?' Interface transient  t='+fmt(p.field_time_s*1e6,4)+' µs':' Quasi-steady Cartesian field ';
  $('cut-summary').textContent=clock+' · FV '+this.axis+'  Interval  ['+p.cell_slab_m.map(v=>fmt(v*1e6,2)).join(', ')+'] µm ·  Plane intersections  '+this.intersections.length+'  Visible cells '+(selectedOffPlane!=null?' ·  Selected cell off plane  '+fmt(selectedOffPlane*1e6,2)+' µm':'')+(this.state.frame?.omitted_count?' ·  Display sampling active ':'');
  $('cut-aspect').textContent=$('cut-equal').checked?' Equal physical scale ;  Circular sphere sections ':' Independent axis scales ;  Physical units  µm,  Sphere sections appear elliptical ';
 }
}
