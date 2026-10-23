"""Loopback HTTP server for the 2D and 3D workspace."""
from __future__ import annotations

from copy import deepcopy
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import json
import mimetypes
import secrets
import traceback
import webbrowser
import numpy as np

from ..scene3d.server import Handler as SceneHandler, STATIC as SHARED_STATIC, _number
from ..scene3d.fields import _palette
from ..scene3d.data import _clean
from .service import StudioService
from .slices import field_slice, FIELD_INFO

STATIC=Path(__file__).with_name('static')


def geometry(service,scene,field='methanol',phase=-1,clock='volume',axis='both',position=0.):
    native=hasattr(scene,'volume')
    extra=field in ('shear','charge_density','liquid_fraction')
    if native and not extra:
        return scene.geometry(field=field,phase_frame=phase,data_clock=clock,
            slice_axis=axis,slice_position_m=position)
    key=(scene.source_id,field,int(phase),clock,axis,float(position))
    if not hasattr(service,'geometry_cache'):service.geometry_cache={}
    if key in service.geometry_cache:return deepcopy(service.geometry_cache[key])
    result=deepcopy(scene.geometry(field='methanol',phase_frame=phase,data_clock=clock,slice_axis='none')
                    if native else scene.geometry(phase_frame=phase))
    result['meshes']=[m for m in result['meshes'] if not m['id'].startswith('methanol_meridian_')]
    if not native and clock=='phase':
        # The old adapter's arrows belong to the steady internal solution,
        # never to a selected microsecond free-surface snapshot.
        result['lines']=[line for line in result.get('lines',[]) if line.get('id') not in ('velocity_vectors','field_vectors')]
        result.setdefault('notes',[]).append(' Interface clock uses transient velocities / Electric-field vectors ;  Vectors use the selected interface frame .')
    cuts=[];scale=None
    for cut_axis in (['x','y'] if axis=='both' else [] if axis=='none' else [axis]):
        cut=field_slice(scene,field,cut_axis,position,clock,phase)
        scale=cut['scale'];cuts.append({k:cut[k] for k in ('mode','axis','position_m','cell_slab_m','sampling')})
        values=cut['values'];he=cut['horizontal_edges_m'];ve=cut['vertical_edges_m']
        positions=[];colors=[];indices=[]
        for i,row in enumerate(values):
            for j,v in enumerate(row):
                if v is None:continue
                corners=np.zeros((4,3));corners[:,'xyz'.index(cut_axis)]=position
                corners[:,'xyz'.index(cut['horizontal_axis'])]=[he[j],he[j+1],he[j+1],he[j]]
                corners[:,'xyz'.index(cut['vertical_axis'])]=[ve[i],ve[i],ve[i+1],ve[i+1]]
                if scene.path is not None:corners=scene._global(corners)
                k=len(positions)//3;positions.extend(corners.ravel().tolist())
                color=_palette((v-scale['min'])/(scale['max']-scale['min']),scale['palette']).tolist()+[1.]
                colors.extend(color*4);indices.extend([k,k+1,k+2,k,k+2,k+3])
        result['meshes'].append(dict(id='studio_'+field+'_'+cut_axis,positions=positions,colors=colors,
            indices=indices,color=[1,1,1,1],opacity=.72,doubleSided=True,layer='fluid'))
    result['scales']=[] if scale is None else [scale]
    result['metadata']=dict(mode='native_cartesian_volume' if native else 'axisymmetric_field_reconstruction',
        dimensions=3 if native else 2,field=field,slices=cuts,
        available_fields=[dict(name=k,label=v[0],unit=v[1]) for k,v in FIELD_INFO.items()],
        scope='Native Cartesian finite-volume field' if native else 'Axisymmetric solved values reconstructed in the device; not native 3-D CFD')
    if native:result['native_field']=dict(scales=result['scales'],metadata=result['metadata'])
    result['data_clock']=clock
    service.geometry_cache[key]=deepcopy(result)
    if len(service.geometry_cache)>8:service.geometry_cache.pop(next(iter(service.geometry_cache)))
    return result


class Handler(SceneHandler):
    def do_GET(self):
        if not self._same_origin():return self._send(403,{'error':'Local same-origin access only'})
        parsed=urlsplit(self.path);query=parse_qs(parsed.query)
        one=lambda name,default=None:query.get(name,[default])[0]
        try:
            assets={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/shell.js':'shell.js',
                    '/style.css':'style.css','/slices.js':'slices.js','/process2d.js':'process2d.js','/ordering.js':'ordering.js','/ordering.css':'ordering.css'}
            shared={'/shared/renderer.js':'renderer.js','/shared/slices.js':'slices.js'}
            if parsed.path in assets or parsed.path in shared:
                path=(STATIC/assets[parsed.path]) if parsed.path in assets else SHARED_STATIC/shared[parsed.path]
                mime={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8'}[path.suffix]
                return self._send(200,path.read_bytes(),mime)
            if parsed.path=='/api/studio':return self._send(200,self.service.studio_metadata())
            if parsed.path in ('/api/ordering','/api/ordering/frame','/api/ordering/event'):
                from . import ordering
                if parsed.path=='/api/ordering':value=ordering.overview(self.service)
                elif parsed.path.endswith('/frame'):value=ordering.frame(self.service,_number(one('t',0.),'study time',0.,3600.))
                else:
                    study=ordering.current(self.service)
                    if study is None:raise ValueError("Run a population study")
                    value=study.history(one('id'))
                return self._send(200,_clean(value))
            if parsed.path=='/api/analysis':
                from .analysis import get_analysis
                return self._send(200,get_analysis(self.service,one('kind','overview')))
            if parsed.path=='/api/exports':return self._send(200,self.service.export_records)
            if parsed.path=='/api/study-records':
                from .artifacts import study_catalog
                return self._send(200,study_catalog(self.service))
            if parsed.path=='/api/classic':
                from .classic import catalog
                return self._send(200,catalog(self.service))
            if parsed.path=='/api/demo':
                from .classic import demonstration
                return self._send(200,demonstration(self.service,{k:v[0] for k,v in query.items()}))
            if parsed.path=='/api/classic-scene.png':
                from .classic import render_scene
                return self._send(200,render_scene(self.service,{k:v[0] for k,v in query.items()}),'image/png')
            if parsed.path=='/api/config':
                scene,_=self.service.snapshot()
                return self._send(200,scene.data.config.to_dict())
            if parsed.path in ('/api/geometry','/api/slice'):
                scene,revision=self.service.snapshot()
                field=one('field','methanol');phase=int(_number(one('phase',-1),'phase frame',-1,10000))
                if parsed.path.endswith('geometry'):
                    value=geometry(self.service,scene,field,phase,one('clock','volume'),one('slice_axis','both'),
                        _number(one('slice_position_m',0.),'slice position',-10,10))
                else:
                    value=field_slice(scene,field,one('axis','x'),
                        _number(one('position',0.),'slice position',-10,10),one('clock','volume'),phase,
                        None if one('t') is None else _number(one('t'),'transport time',0.,scene.model.parameters.duration_s))
                value['revision']=revision;return self._send(200,_clean(value))
            if parsed.path.startswith('/research/'):
                from .artifacts import study_catalog
                components=unquote(parsed.path).split('/')
                if len(components)<4:return self._send(404,{'error':'Unknown study file'})
                record=next((x for x in study_catalog(self.service) if x['id']==components[2]),None)
                if record is None:return self._send(404,{'error':'Unknown study'})
                root=Path(record['path']).resolve();relative='/'.join(components[3:])
                if ':' in relative or '\\' in relative:return self._send(404,{'error':'Unknown study file'})
                path=(root/relative).resolve()
                if not path.is_relative_to(root) or not path.is_file():return self._send(404,{'error':'Unknown study file'})
                return self._file(path)
            if parsed.path.startswith('/classic-files/'):
                from .classic import catalog
                components=unquote(parsed.path).split('/')
                if len(components)!=4 or components[3] in ('.','..') or ':' in components[3] or '\\' in components[3]:
                    return self._send(404,{'error':'Unknown classic file'})
                catalog(self.service)
                root=getattr(self.service,'classic_sources',{}).get(components[2])
                if root is None:return self._send(404,{'error':'Unknown classic source'})
                path=(Path(root)/components[3]).resolve()
                if not path.is_relative_to(root) or not path.is_file():return self._send(404,{'error':'Unknown classic file'})
                return self._file(path)
            if parsed.path.startswith('/ordering-files/'):
                parts=unquote(parsed.path).split('/')
                if len(parts)!=4 or not parts[2].startswith('collective_') or any('\\' in p or ':' in p or p in ('.','..') for p in parts[2:]):
                    return self._send(404,{'error':'Unknown population file'})
                root=(self.service.studio_root/parts[2]).resolve()
                if not root.is_relative_to(self.service.studio_root.resolve()) or not (root/'record.json').is_file():
                    return self._send(404,{'error':'Unknown population record'})
                path=root/parts[3]
                if not path.is_file():return self._send(404,{'error':'Unknown population file'})
                return self._file(path)
            if parsed.path.startswith('/downloads/'):
                # Restrict downloads to generated report files.
                components=unquote(parsed.path).split('/')
                if len(components)!=4 or any(x in ('.','..') or '\\' in x for x in components[2:]):
                    return self._send(404,{'error':'Unknown report file'})
                record=next((x for x in self.service.export_records if x['id']==components[2]),None)
                if record is None:return self._send(404,{'error':'Unknown report'})
                path=Path(record['path'])/components[3]
                if not path.is_file():return self._send(404,{'error':'Unknown report file'})
                return self._file(path)
            return super().do_GET()
        except (ValueError,TypeError,KeyError) as exc:return self._send(400,{'error':str(exc)})
        except Exception as exc:
            traceback.print_exc();return self._send(500,{'error':str(exc)})

    def _file(self,path):
        mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        data=path.read_bytes()
        self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8' if mime.startswith('text/') else mime)
        self.send_header('Content-Length',str(len(data)));self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self' 'unsafe-inline'; object-src 'none'; frame-ancestors 'none'")
        self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

    def do_POST(self):
        path=urlsplit(self.path).path
        routes={'/api/run2d','/api/mode','/api/export','/api/study','/api/state-parameters','/api/pin','/api/classic-render','/api/ordering'}
        if path not in routes:return super().do_POST()
        if not self._same_origin() or not secrets.compare_digest(self.headers.get('X-CyESI-Session',''),self.service.token):
            return self._send(403,{'error':'A local studio session is required'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=65536:raise ValueError('JSON request body must be 1..65536 bytes')
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):raise ValueError('Expected a JSON object')
            if path=='/api/run2d':result=self.service.run2d(body)
            elif path=='/api/mode':result=self.service.mode(body)
            elif path=='/api/state-parameters':result=self.service.state_parameters(body)
            elif path=='/api/pin':
                from .analysis import pin
                result=pin(self.service,body)
            elif path=='/api/classic-render':
                from .classic import render_gallery
                result=render_gallery(self.service,body)
            elif path=='/api/ordering':
                from .ordering import run
                result=run(self.service,body)
            elif path=='/api/export':
                from .analysis import export_report
                result=export_report(self.service,body)
            else:
                from .studies import run_study
                result=run_study(self.service,body)
            return self._send(202,result)
        except RuntimeError as exc:return self._send(409,{'error':str(exc)})
        except (ValueError,TypeError,KeyError) as exc:return self._send(400,{'error':str(exc)})
        except Exception as exc:
            traceback.print_exc();return self._send(500,{'error':str(exc)})


def create_server(source=None,scene=None,port=0,output_root=None):
    server=ThreadingHTTPServer(('127.0.0.1',int(port)),Handler);server.daemon_threads=True
    try:
        server.service=StudioService(source=source,scene=scene,output_root=output_root)
        server.service.studio_root=server.service.output_root
        server.service.export_records=[];server.service.study_results={}
        reports=server.service.studio_root/'reports'
        for root in sorted(reports.glob('*'),reverse=True) if reports.exists() else []:
            if (root/'report.html').is_file():server.service.export_records.append(dict(id=root.name,path=str(root),report_url=f'/downloads/{root.name}/report.html',zip_url=f'/downloads/{root.name}/report.zip'))
    except Exception:
        server.server_close();raise
    return server


def launch(source=None,port=0,open_browser=True):
    server=create_server(source=source,port=port)
    url=f'http://127.0.0.1:{server.server_port}'
    print('CyESI Studio · unified 2-D / 3-D workspace: '+url,flush=True)
    print("Press Ctrl+C in this terminal to stop the server.",flush=True)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:pass
    finally:server.server_close()
