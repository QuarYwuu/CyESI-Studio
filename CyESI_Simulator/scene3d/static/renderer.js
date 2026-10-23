/* Offline, dependency-free WebGL renderer. Input coordinates are always SI:
 * [x_m, y_m, z_m], with z along the coaxial instrument. Geometry stays connected
 * under the continuous axial lens; it never creates separate scene segments.
 * This module renders supplied solver data and does not integrate any physics.
 */

const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const finite = v => Number.isFinite(Number(v));
const vsub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const vdot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const vcross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const vnorm = a => { const n=Math.hypot(...a)||1; return a.map(v=>v/n); };

function perspective(fov, aspect, near, far) {
  const f=1/Math.tan(fov/2), r=1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*r,-1, 0,0,2*far*near*r,0]);
}
function lookAt(eye, target) {
  const z=vnorm(vsub(eye,target));
  let x=vnorm(vcross([0,1,0],z));
  if(Math.hypot(...x)<1e-8) x=[1,0,0];
  const y=vcross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -vdot(x,eye),-vdot(y,eye),-vdot(z,eye),1]);
}
function multiply(a,b) {
  const o=new Float32Array(16);
  for(let c=0;c<4;c++) for(let r=0;r<4;r++)
    for(let k=0;k<4;k++) o[c*4+r]+=a[k*4+r]*b[c*4+k];
  return o;
}
function project4(m,p) {
  return [m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12],m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13],m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14],m[3]*p[0]+m[7]*p[1]+m[11]*p[2]+m[15]];
}
function program(gl,vs,fs) {
  const p=gl.createProgram();
  for(const [kind,src] of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]) {
    const s=gl.createShader(kind); gl.shaderSource(s,src); gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)||'WebGL shader compilation failed');
    gl.attachShader(p,s); gl.deleteShader(s);
  }
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)||'WebGL shader link failed');
  return p;
}
const VERTEX = `attribute vec3 aPosition; attribute vec3 aNormal; attribute vec4 aColor;
uniform mat4 uVP; varying vec3 vNormal; varying vec4 vColor; varying vec3 vPosition;
void main(){vNormal=aNormal;vColor=aColor;vPosition=aPosition;gl_Position=uVP*vec4(aPosition,1.);}`;
const FRAGMENT = `precision mediump float; varying vec3 vNormal; varying vec4 vColor; varying vec3 vPosition;
uniform vec3 uEye; uniform float uLit; uniform float uOpacity;
void main(){vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
float diffuse=max(dot(n,normalize(vec3(-.35,.8,1.))),0.);
vec3 view=normalize(uEye-vPosition);float rim=pow(1.-abs(dot(n,view)),2.);
float light=mix(1.,.47+.51*diffuse+.12*rim,uLit);
gl_FragColor=vec4(vColor.rgb*light,vColor.a*uOpacity);}`;
const POINT_VERTEX = `attribute vec3 aPosition;attribute vec4 aColor;attribute float aRadius;
uniform mat4 uVP;uniform float uPointScale;uniform float uMaxPointSize;
varying vec4 vColor;void main(){gl_Position=uVP*vec4(aPosition,1.);vColor=aColor;
gl_PointSize=clamp(2.*aRadius*uPointScale/max(gl_Position.w,.001),2.2,uMaxPointSize);}`;
const POINT_FRAGMENT = `precision mediump float;varying vec4 vColor;uniform float uOpacity;
void main(){vec2 q=gl_PointCoord*2.-1.;float r=dot(q,q);if(r>1.)discard;
float z=sqrt(max(0.,1.-r));vec3 n=normalize(vec3(q.x,-q.y,z));
float light=.4+.6*max(dot(n,normalize(vec3(-.45,.65,1.))),0.);
float border=smoothstep(1.,.8,r);gl_FragColor=vec4(vColor.rgb*light,vColor.a*uOpacity*border);}`;

function rgba(colors, fallback, n) {
  const out=new Float32Array(n*4), c=fallback||[.2,.7,.9,1];
  const source=colors||[], stride=source.length===n*3?3:4;
  for(let i=0;i<n;i++)for(let j=0;j<4;j++) out[i*4+j]=source.length>=n*stride?(j===3&&stride===3?1:Number(source[i*stride+j])):(c[j]??1);
  return out;
}
function normals(positions, indices) {
  const out=new Float32Array(positions.length);
  const at=i=>[positions[3*i],positions[3*i+1],positions[3*i+2]];
  const n=indices?indices.length:positions.length/3;
  for(let k=0;k+2<n;k+=3) {
    const a=indices?indices[k]:k,b=indices?indices[k+1]:k+1,c=indices?indices[k+2]:k+2;
    const cross=vcross(vsub(at(b),at(a)),vsub(at(c),at(a)));
    for(const j of [a,b,c])for(let q=0;q<3;q++)out[j*3+q]+=cross[q];
  }
  for(let i=0;i<out.length;i+=3){const n=Math.hypot(out[i],out[i+1],out[i+2])||1;out[i]/=n;out[i+1]/=n;out[i+2]/=n;}
  return out;
}

export class SceneRenderer {
  constructor(canvas, options={}) {
    this.canvas=canvas; this.options=options;
    const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true})||canvas.getContext('experimental-webgl');
    if(!gl)throw new Error(' This browser could not create  WebGL. Enable hardware acceleration ,  or use a browser supporting  WebGL  rendering .');
    this.gl=gl;this.uint32=gl.getExtension('OES_element_index_uint');
    this.maxPointSize=gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.meshProgram=program(gl,VERTEX,FRAGMENT);this.pointProgram=program(gl,POINT_VERTEX,POINT_FRAGMENT);
    this.objects=[];this.cells=[];this.layers=new Map();this.opacities=new Map();this.cellBuffers=[];
    this.transform={axialLens:true,axialLength:.0015,axialScale:4,radialScale:4000,cellExaggeration:1};
    this.coordinateMapper=null;
    this.camera={target:[-9,0,0],distance:34,yaw:0.2,pitch:.2};
    this.background=[.035,.061,.106,1];this.disposed=false;this.needsRender=true;
    this.handlers=[];this._controls();
    this._contextLost=e=>{e.preventDefault();this.contextLost=true;options.onError?.('WebGL  context lost ,  Reload the page to restore rendering .');};
    canvas.addEventListener('webglcontextlost',this._contextLost);
    this.resizeObserver=typeof ResizeObserver!=='undefined'?new ResizeObserver(()=>this.resize()):null;
    this.resizeObserver?.observe(canvas);this.resize();
  }
  _listen(name,fn,opts){this.canvas.addEventListener(name,fn,opts);this.handlers.push([name,fn,opts]);}
  _controls(){
    this.canvas.style.touchAction='none';
    let drag=null;
    this._listen('contextmenu',e=>e.preventDefault());
    this._listen('pointerdown',e=>{
      if(e.button>2)return;this.canvas.setPointerCapture(e.pointerId);
      drag={id:e.pointerId,x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.button===2||e.shiftKey,moved:false};
    });
    this._listen('pointermove',e=>{
      if(!drag||drag.id!==e.pointerId)return;
      const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4)drag.moved=true;
      drag.x=e.clientX;drag.y=e.clientY;
      if(drag.pan||e.shiftKey){
        const scale=this.camera.distance*.0015;
        const [right,up]=this._cameraBasis();
        for(let i=0;i<3;i++)this.camera.target[i]+=-dx*scale*right[i]+dy*scale*up[i];
      }else{this.camera.yaw-=dx*.007;this.camera.pitch=clamp(this.camera.pitch+dy*.007,-1.48,1.48);}
      this.invalidate();this.options.onCameraChange?.(this.camera);
    });
    this._listen('pointerup',e=>{
      if(!drag||drag.id!==e.pointerId)return;
      if(!drag.moved&&e.button===0)this.pick(e.clientX,e.clientY);
      drag=null;
    });
    this._listen('pointercancel',()=>{drag=null;});
    this._listen('wheel',e=>{
      e.preventDefault();this.camera.distance=clamp(this.camera.distance*Math.exp(clamp(e.deltaY,-300,300)*.0012),.08,50000);
      this.invalidate();this.options.onCameraChange?.(this.camera);
    },{passive:false});
    this._listen('dblclick',()=>this.setCameraPreset('whole'));
  }
  _cameraBasis(){const eye=this.eye(),forward=vnorm(vsub(this.camera.target,eye)),right=vnorm(vcross(forward,[0,1,0]));return [right,vnorm(vcross(right,forward))];}
  eye(){const c=this.camera,cp=Math.cos(c.pitch);return[c.target[0]+c.distance*Math.sin(c.yaw)*cp,c.target[1]+c.distance*Math.sin(c.pitch),c.target[2]+c.distance*Math.cos(c.yaw)*cp];}
  world(position){
    if(this.coordinateMapper)return this.coordinateMapper(position);
    const t=this.transform,z=Number(position[2]);
    return[t.axialLens?t.axialScale*Math.asinh(z/t.axialLength):z*t.radialScale,Number(position[0])*t.radialScale,Number(position[1])*t.radialScale];
  }
  unworld(position){const t=this.transform;return[position[1]/t.radialScale,position[2]/t.radialScale,t.axialLens?t.axialLength*Math.sinh(position[0]/t.axialScale):position[0]/t.radialScale];}
  setCoordinateMapper(mapper=null){
    if(mapper!==null&&typeof mapper!=='function')throw new TypeError('Coordinate mapper must be a function or null');
    this.coordinateMapper=mapper;
    for(const o of this.objects)this._uploadObject(o);
    this._uploadCells();this.invalidate();
  }
  setTransform(options={}) {
    const before=this.transform;
    this.transform={...before,...options};
    for(const k of ['axialLength','axialScale','radialScale','cellExaggeration'])if(!(finite(this.transform[k])&&this.transform[k]>0))this.transform[k]=before[k];
    this.transform.axialLens=Boolean(this.transform.axialLens);
    if(['axialLens','axialLength','axialScale','radialScale'].some(k=>before[k]!==this.transform[k]))
      for(const o of this.objects)this._uploadObject(o);
    this._uploadCells();this.invalidate();
  }
  setLayer(layer,visible){this.layers.set(layer,Boolean(visible));this.invalidate();}
  setOpacity(layer,value){this.opacities.set(layer,clamp(Number(value)||0,0,1));this.invalidate();}
  setBackground(color){this.background=color;this.invalidate();}
  isVisible(layer){return this.layers.get(layer)!==false;}
  _buffer(data, target=this.gl.ARRAY_BUFFER,usage=this.gl.STATIC_DRAW){const gl=this.gl,b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,usage);return b;}
  _deleteBuffers(o){for(const b of o.buffers||[])this.gl.deleteBuffer(b);o.buffers=[];}
  _uploadObject(o){
    const gl=this.gl;this._deleteBuffers(o);
    const source=o.source,n=source.positions.length/3, p=new Float32Array(n*3);
    for(let i=0;i<n;i++)p.set(this.world(source.positions.slice(i*3,i*3+3)),i*3);
    o.center=[0,0,0];for(let i=0;i<n;i++)for(let q=0;q<3;q++)o.center[q]+=p[3*i+q]/Math.max(1,n);
    const allColors=rgba(source.colors,source.color,n);
    o.minimumAlpha=1;for(let i=3;i<allColors.length;i+=4)o.minimumAlpha=Math.min(o.minimumAlpha,allColors[i]);
    let indices=source.indices?Array.from(source.indices):null;
    // WebGL 1 can lack 32-bit element indices: expand instead of dropping data.
    if(indices&&!this.uint32&&n>65535){
      const expanded=new Float32Array(indices.length*3),colors=new Float32Array(indices.length*4);
      for(let i=0;i<indices.length;i++){expanded.set(p.subarray(indices[i]*3,indices[i]*3+3),i*3);colors.set(allColors.subarray(indices[i]*4,indices[i]*4+4),i*4);}
      o.position=this._buffer(expanded);o.normal=this._buffer(o.type==='mesh'?normals(expanded,null):new Float32Array(expanded.length));o.color=this._buffer(colors);o.count=indices.length;o.indices=null;
    }else{
      o.position=this._buffer(p);o.normal=this._buffer(o.type==='mesh'?normals(p,indices):new Float32Array(p.length).fill(1));o.color=this._buffer(allColors);o.count=indices?indices.length:n;
      if(indices){const use32=n>65535,arr=use32?new Uint32Array(indices):new Uint16Array(indices);o.indices=this._buffer(arr,gl.ELEMENT_ARRAY_BUFFER);o.indexType=use32?gl.UNSIGNED_INT:gl.UNSIGNED_SHORT;}else o.indices=null;
    }
    o.buffers=[o.position,o.normal,o.color,...(o.indices?[o.indices]:[])];
  }
  setScene(scene={}){
    for(const o of this.objects)this._deleteBuffers(o);this.objects=[];
    for(const [type,sources] of [['mesh',scene.meshes||[]],['line',scene.lines||[]]])for(const source of sources){
      if(!source.positions||source.positions.length<6||source.positions.length%3)continue;
      if(!Array.from(source.positions).every(finite))continue;
      const n=source.positions.length/3;
      if(source.indices&&Array.from(source.indices).some(i=>!Number.isInteger(i)||i<0||i>=n))continue;
      const o={source,type,id:source.id,layer:source.layer||(type==='mesh'?'geometry':'fields'),buffers:[]};
      this._uploadObject(o);this.objects.push(o);
    }
    this.invalidate();
  }
  setCells(cells=[]){
    this.cells=cells.filter(c=>c.position?.length===3&&c.position.every(finite));this._uploadCells();this.invalidate();
  }
  _uploadCells(){
    const gl=this.gl;
    const n=this.cells.length,p=new Float32Array(n*3),colors=new Float32Array(n*4),sizes=new Float32Array(n);
    for(let i=0;i<n;i++){
      const c=this.cells[i];p.set(this.world(c.position),i*3);const color=c.color||[.98,.75,.22,1];colors.set([color[0],color[1],color[2],color[3]??1],i*4);
      sizes[i]=Math.max(1e-9,Number(c.radius_m)||4e-6)*this.transform.radialScale*this.transform.cellExaggeration;
    }
    const upload=(buffer,data)=>{
      if(!buffer)return this._buffer(data,gl.ARRAY_BUFFER,gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);return buffer;
    };
    this.cellPosition=upload(this.cellPosition,p);this.cellColor=upload(this.cellColor,colors);this.cellRadius=upload(this.cellRadius,sizes);
    this.cellBuffers=[this.cellPosition,this.cellColor,this.cellRadius];
  }
  setCamera(options={}){this.camera={...this.camera,...options,target:options.target?[...options.target]:this.camera.target};this.invalidate();this.options.onCameraChange?.(this.camera);}
  setCameraPreset(name='whole',bounds={}){
    if(name==='tip'||name==='nozzle')this.setCamera({target:this.world([0,0,-.001]),distance:11,yaw:.12,pitch:.18});
    else if(name==='inlet'||name==='ms')this.setCamera({target:this.world([0,0,.0018]),distance:9,yaw:.35,pitch:.2});
    else if(this.coordinateMapper&&this.objects.length){
      const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
      for(const object of this.objects)for(let k=0;k<object.source.positions.length;k+=3){
        const p=this.world(object.source.positions.slice(k,k+3));for(let j=0;j<3;j++){min[j]=Math.min(min[j],p[j]);max[j]=Math.max(max[j],p[j]);}
      }
      this.setCamera({target:min.map((v,i)=>(v+max[i])/2),distance:Math.max(8,Math.hypot(...max.map((v,i)=>v-min[i]))*1.18),yaw:.12,pitch:.16});
    }else{
      const z0=bounds.z_min??-.8,z1=bounds.z_max??.004;
      const left=this.world([0,0,z0]),right=this.world([0,0,z1]);
      this.setCamera({target:[(left[0]+right[0])/2,0,0],distance:Math.max(8,Math.abs(right[0]-left[0])*1.18),yaw:.12,pitch:.16});
    }
  }
  focus(position,distance=5){this.setCamera({target:this.world(position),distance:Math.max(.08,distance)});}
  resize(){
    if(this.disposed)return;const rect=this.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(rect.width*dpr)),height=Math.max(1,Math.round(rect.height*dpr));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    this.cssWidth=Math.max(1,rect.width);this.cssHeight=Math.max(1,rect.height);this.dpr=dpr;this.invalidate();
  }
  _matrices(){const eye=this.eye(),far=Math.max(200,this.camera.distance*4),near=Math.max(.001,this.camera.distance/10000);return{eye,vp:multiply(perspective(Math.PI/4,this.canvas.width/this.canvas.height,near,far),lookAt(eye,this.camera.target))};}
  project(position){
    const {vp}=this._matrices(),p=project4(vp,this.world(position)),w=p[3]||1;
    return{x:(p[0]/w*.5+.5)*this.cssWidth,y:(.5-p[1]/w*.5)*this.cssHeight,depth:p[2]/w,visible:w>0&&Math.abs(p[0]/w)<=1&&Math.abs(p[1]/w)<=1&&Math.abs(p[2]/w)<=1};
  }
  pick(clientX,clientY){
    if(!this.isVisible('cells'))return null;
    const rect=this.canvas.getBoundingClientRect(),x=clientX-rect.left,y=clientY-rect.top;
    let best=null,distance=16;
    for(const cell of this.cells){const p=this.project(cell.position);if(!p.visible)continue;const d=Math.hypot(p.x-x,p.y-y);if(d<distance){best=cell;distance=d;}}
    if(best)this.options.onPick?.(best.id,best);return best;
  }
  invalidate(){this.needsRender=true;}
  _attrib(p,name,buffer,size){const gl=this.gl,location=gl.getAttribLocation(p,name);if(location<0)return;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,0,0);}
  render(){
    if(this.disposed||this.contextLost)return;this.needsRender=false;
    const gl=this.gl,{eye,vp}=this._matrices();
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(...this.background);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.meshProgram);gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProgram,'uVP'),false,vp);gl.uniform3fv(gl.getUniformLocation(this.meshProgram,'uEye'),eye);
    const visible=this.objects.filter(o=>this.isVisible(o.layer));
    const alpha=o=>(o.source.opacity??1)*(this.opacities.get(o.layer)??1)*o.minimumAlpha;
    const opaque=visible.filter(o=>o.type==='mesh'&&alpha(o)>=.995);
    const transparent=visible.filter(o=>o.type==='mesh'&&alpha(o)<.995).sort((a,b)=>vdot(vsub(b.center,eye),vsub(b.center,eye))-vdot(vsub(a.center,eye),vsub(a.center,eye)));
    const draw=(o,depthWrite)=>{
      gl.depthMask(depthWrite);const source=o.source;
      this._attrib(this.meshProgram,'aPosition',o.position,3);this._attrib(this.meshProgram,'aNormal',o.normal,3);this._attrib(this.meshProgram,'aColor',o.color,4);
      gl.uniform1f(gl.getUniformLocation(this.meshProgram,'uLit'),o.type==='mesh'&&source.unlit!==true?1:0);gl.uniform1f(gl.getUniformLocation(this.meshProgram,'uOpacity'),(source.opacity??1)*(this.opacities.get(o.layer)??1));
      const mode=o.type==='mesh'?gl.TRIANGLES:(source.mode==='segments'?gl.LINES:gl.LINE_STRIP);
      if(o.type==='line')gl.lineWidth(Math.min(2,source.width||1));
      if(o.indices){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,o.indices);gl.drawElements(mode,o.count,o.indexType,0);}else gl.drawArrays(mode,0,o.count);
    };
    for(const o of opaque)draw(o,true);
    // Cells write depth before transparent shells, keeping transparent quartz
    // and sampled solvent visible without hiding cells inside the tube.
    if(this.cells.length&&this.isVisible('cells')){
      const p=this.pointProgram;gl.useProgram(p);gl.depthMask(true);
      this._attrib(p,'aPosition',this.cellPosition,3);this._attrib(p,'aColor',this.cellColor,4);this._attrib(p,'aRadius',this.cellRadius,1);
      gl.uniformMatrix4fv(gl.getUniformLocation(p,'uVP'),false,vp);gl.uniform1f(gl.getUniformLocation(p,'uPointScale'),this.canvas.height/(2*Math.tan(Math.PI/8)));
      gl.uniform1f(gl.getUniformLocation(p,'uMaxPointSize'),Math.min(80*this.dpr,this.maxPointSize));gl.uniform1f(gl.getUniformLocation(p,'uOpacity'),this.opacities.get('cells')??1);gl.drawArrays(gl.POINTS,0,this.cells.length);gl.useProgram(this.meshProgram);
    }
    for(const o of transparent)draw(o,false);
    for(const o of visible.filter(o=>o.type==='line'))draw(o,false);
    gl.depthMask(true);
    this.options.onRender?.(this);
  }
  dispose(){
    if(this.disposed)return;this.disposed=true;this.resizeObserver?.disconnect();
    for(const [name,fn,opts]of this.handlers)this.canvas.removeEventListener(name,fn,opts);
    this.canvas.removeEventListener('webglcontextlost',this._contextLost);
    for(const o of this.objects)this._deleteBuffers(o);for(const b of this.cellBuffers)this.gl.deleteBuffer(b);
    this.gl.deleteProgram(this.meshProgram);this.gl.deleteProgram(this.pointProgram);
  }
}

/** Revolve measured (z_m,r_m) points around the physical z axis. Vertex colors
 * may be supplied per profile point as RGB(A), e.g. sampled methanol fractions.
 */
export function revolve(profile,segments=64,options={}){
  segments=Math.max(8,Math.min(128,Math.round(segments)));
  const positions=[],indices=[],colors=[],profileColors=options.profileColors;
  for(let j=0;j<profile.length;j++){
    const [z,r]=profile[j];
    for(let i=0;i<=segments;i++){
      const angle=i/segments*TAU;positions.push(r*Math.cos(angle),r*Math.sin(angle),z);
      if(profileColors)colors.push(...(profileColors[j]||[.2,.7,.9,1]).slice(0,3),profileColors[j]?.[3]??1);
    }
  }
  for(let j=0;j<profile.length-1;j++)for(let i=0;i<segments;i++){
    const a=j*(segments+1)+i,b=a+segments+1;indices.push(a,b,a+1,a+1,b,b+1);
  }
  return{positions,indices,...(profileColors?{colors}:{}),...Object.fromEntries(Object.entries(options).filter(([k])=>k!=='profileColors'))};
}

/** Hollow quartz cylinder with optional physical end annuli. Use more axial
 * samples around a large length range so the continuous axial lens is smooth.
 */
export function tube(z0,z1,rInner,rOuter,options={}){
  const count=options.axialSamples||80,profile=[];
  for(let i=0;i<=count;i++){
    // asinh-spaced points resolve the nozzle and the long upstream tube in
    // physical coordinates without moving or separating either endpoint.
    const a=Math.asinh(z0/.0015),b=Math.asinh(z1/.0015);
    profile.push([.0015*Math.sinh(a+(b-a)*i/count),rOuter]);
  }
  if(options.caps!==false)profile.push([z1,rInner]);
  for(let i=count;i>=0;i--)profile.push([profile[i][0],rInner]);
  if(options.caps!==false)profile.push([z0,rOuter]);
  return revolve(profile,options.segments||64,options);
}

/** SI-coordinate line arrow. The head is built from the supplied direction;
 * arrow length must already express the caller's stated visualization scale.
 */
export function arrow(start,end,options={}){
  const d=vsub(end,start),length=Math.hypot(...d);
  if(!length)return{positions:[],...options};
  const axis=vnorm(d),side=vnorm(vcross(axis,Math.abs(axis[2])>.8?[1,0,0]:[0,0,1]));
  const head=length*.2,base=end.map((v,i)=>v-axis[i]*head),left=base.map((v,i)=>v+side[i]*head*.45),right=base.map((v,i)=>v-side[i]*head*.45);
  return{positions:[...start,...end,...end,...left,...end,...right],mode:'segments',...options};
}

SceneRenderer.revolve=revolve;
SceneRenderer.tube=tube;
SceneRenderer.arrow=arrow;

export default SceneRenderer;
