"""Dependency-aware reuse of the existing solvers, without global patching.

The pressure-controlled flow stored with a liquid solution is an output. Every
consumer receives that solved flow, never the stale input guess. Caches belong
to this workbench instance; legacy modules and files are not modified.
"""
from collections import OrderedDict,Counter
from copy import deepcopy
import hashlib
import json
import threading
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def physical_inputs(config):
    d=deepcopy(config.to_dict())
    d['cell'].pop('concentration_per_ml')
    # Threshold and manual controls can describe the same physical choice.
    d['cell']['brownian_enabled']=config.cell.use_brownian
    d['cell'].pop('brownian_below_um')
    d.pop('events')
    for key in ('workers','particle_backend'):d['numerics'].pop(key)
    if d['supply']['sample_control']=='pressure':d['fluid'].pop('sample_flow_ul_min')
    return d


def physical_key(config):return digest(physical_inputs(config))


def reusable_record(data):
    """Whether an archived particle response satisfies the current contract.

    Historical records stay inspectable, and their carrier fields can still be
    restored. The former legacy long-pipe path used a different finite-size
    wall/time boundary; silently reusing that response would hide its repair.
    """
    config=data.config;particles=data.summary.get('particles',{})
    if config.supply.orientation=='legacy' and config.geometry.bulk_length>0:
        return all(key in particles for key in ('bulk_wall_boundary','bulk_time_censoring'))
    return True


def reusable_output(data):
    """Complete requested output, including the current spray charge model."""
    if not reusable_record(data):return False
    if data.config.spray.enabled:
        if data.spray is None:return False
        if data.config.spray.free_charge_enabled:
            from ..spray.charge import CHARGE_MODEL_VERSION
            diag=data.summary.get('spray') or {}
            if ((diag.get('electric_coupling') or {}).get('model_version')!=CHARGE_MODEL_VERSION or not diag.get('species_coupling')):return False
    return True


def stage_keys(config):
    d=physical_inputs(config);n=d['numerics']
    fluid_n={k:v for k,v in n.items() if k not in ('particle_dt_s','bulk_dt_s','bulk_timeout_s',
        'particle_tip_timeout_s','saved_trajectories','seed')}
    supply={k:v for k,v in d['supply'].items() if k not in ('wall_capture_gap_um','wall_capture_rate_s',
        'saffman_enabled','lubrication_enabled','shear_damage_coefficient','shear_damage_exponent')}
    liquid_g={k:v for k,v in d['geometry'].items() if k in ('inner_id_um','inner_od_um','sheath_id_um',
        'sample_length_mm','tip_resolved_length_mm','inner_recess_mm')}
    liquid_f={k:v for k,v in d['fluid'].items() if not k.startswith('gas_') and k!='surface_tension'}
    liquid_e={k:v for k,v in d['electrical'].items() if k!='quartz_relative_permittivity'}
    liquid=digest(dict(geometry=liquid_g,fluid=liquid_f,chemistry=d['chemistry'],
        electrical=liquid_e,numerics=fluid_n,supply=supply,legacy_gravity_z=d['cell']['gravity_z']))
    electric=digest(dict(liquid=liquid,geometry=d['geometry'],electrical=d['electrical'],radial_cells=n['radial_cells']))
    # All particle force samples are inside the liquid. Conducting mode has
    # zero bulk DC field; resistive mode samples its liquid ionic grid, which
    # is guarded against external-field DEP interpolation in ElectricResult.
    particle_field=electric if config.electrical.liquid_model=='dielectric' else liquid
    particles=digest(dict(liquid=liquid,electric=particle_field,cell=d['cell'],supply=d['supply'],numerics=n))
    spray=digest(dict(liquid=liquid,electric=electric,spray=d['spray'],fluid=d['fluid'],geometry=d['geometry']))
    events=digest(dict(particles=particles,concentration=config.cell.concentration_per_ml,events=config.events.__dict__))
    return dict(liquid=liquid,electric=electric,particles=particles,spray=spray,events=events)


class StageCache:
    def __init__(self,capacity=4):
        self.capacity=capacity;self.values=OrderedDict();self.counts=Counter()

    def remember(self,stage,key,value):
        token=(stage,key);self.values[token]=value;self.values.move_to_end(token)
        peers=[k for k in self.values if k[0]==stage]
        limit=self.capacity*2 if stage in ('particles','events') else self.capacity
        while len(peers)>limit:self.values.pop(peers.pop(0))

    def get_or_compute(self,stage,key,compute,progress):
        token=(stage,key)
        if token in self.values:
            self.counts[stage+'_hit']+=1;self.values.move_to_end(token)
            progress('复用缓存 · '+stage)
            return self.values[token]
        progress('计算 · '+stage);value=compute()
        self.counts[stage+'_solve']+=1;self.remember(stage,key,value)
        return value


class WorkbenchSolver:
    def __init__(self,capacity=4):
        self.cache=StageCache(capacity);self.lock=threading.RLock()

    def seed_saved(self,data):
        """Restore saved carrier samplers for particle-only changes.

        No face flux or FEM coefficients are invented. These restored objects
        are used by the existing particle/spray samplers only. Any changed
        carrier condition requires a new liquid solve under a different key.
        Resistive results require the actual exported ionic archive.
        """
        from pathlib import Path
        from types import SimpleNamespace
        from ..chemistry.solvent_transport import SolventResult
        from ..electro.electrostatics import ElectricResult
        from ..electro.ionic_transport import IonicResult
        from ..geometry.coaxial_geometry import classify
        c=deepcopy(data.config);f=data.fields;summary=data.summary;ionic=None
        if not all(k in summary for k in ('flow','solvent','coupling')):return False
        if c.electrical.liquid_model=='resistive':
            path=Path(data.source)/'ions.npz'
            if not path.exists():return False
            with np.load(path,allow_pickle=False) as arrays:
                ionic=IonicResult(arrays['radial_edges_m'],arrays['axial_edges_m'],arrays['salt_mol_m3'],arrays['potential_v'],
                    arrays['field_r_v_m'],arrays['field_z_v_m'],arrays['current_radial_a'],arrays['current_axial_a'],
                    arrays['conductivity_s_m'],arrays['active'],summary['electric'].get('ionic') or {})
        solvent=SolventResult(f['liquid_r_edges_m'],f['liquid_z_edges_m'],f['methanol_fraction'],f['liquid_active'],
            f['velocity_r_m_s'],f['velocity_z_m_s'],f['pressure_pa'],None,None,summary['solvent'])
        if ionic is not None:solvent.ionic=ionic
        for key in ('equivalent_stress_pa','axial_extension_stress_pa'):
            if key in f:setattr(solvent,key,f[key])
        er=(f['electric_r_edges_m'][:-1]+f['electric_r_edges_m'][1:])/2
        ez=(f['electric_z_edges_m'][:-1]+f['electric_z_edges_m'][1:])/2
        R,Z=np.meshgrid(er,ez,indexing='ij');domains=f.get('electric_domains',classify(R,Z,c.geometry))
        electric=ElectricResult(f['electric_r_edges_m'],f['electric_z_edges_m'],f['potential_v'],f['field_r_v_m'],
            f['field_z_v_m'],domains,summary['electric'],c,ionic)
        keys=stage_keys(c)
        with self.lock:
            previous=self.cache.values.get(('liquid',keys['liquid']))
            previous_electric=self.cache.values.get(('electric',keys['electric']))
            changed=False
            if previous is not None:
                old=previous[1]
                changed=previous[3]!=c.fluid.sample_flow_ul_min or any(not np.array_equal(getattr(old,key),getattr(solvent,key),equal_nan=True)
                    for key in ('radial_edges','axial_edges','concentration','active','velocity_r','velocity_z','pressure'))
                for key in ('equivalent_stress_pa','axial_extension_stress_pa'):
                    a,b=getattr(old,key,None),getattr(solvent,key,None)
                    if (a is None)!=(b is None) or (a is not None and not np.array_equal(a,b,equal_nan=True)):changed=True
            if previous_electric is not None:
                changed=changed or any(not np.array_equal(getattr(previous_electric,key),getattr(electric,key),equal_nan=True)
                    for key in ('radial_edges','axial_edges','domains','potential','field_r','field_z'))
                old_ionic=previous_electric.ionic
                if (old_ionic is None)!=(ionic is None):changed=True
                elif ionic is not None:
                    changed=changed or any(not np.array_equal(getattr(old_ionic,key),getattr(ionic,key),equal_nan=True)
                        for key in ('radial_edges','axial_edges','active','salt_mol_m3','conductivity','potential','field_r','field_z'))
            if changed:
                for token in list(self.cache.values):
                    if token[0] in ('particles','events','spray'):self.cache.values.pop(token)
                self.cache.counts['source_revision_invalidation']+=1
            for stage,value in [('liquid',(SimpleNamespace(diagnostics=summary['flow']),solvent,summary['coupling'],c.fluid.sample_flow_ul_min)),
                                ('electric',electric)]:
                token=(stage,keys[stage])
                if changed or token not in self.cache.values:self.cache.remember(stage,keys[stage],value);self.cache.counts[stage+'_restore']+=1
            from ..spray.charge import CHARGE_MODEL_VERSION
            phase_compatible=(not c.spray.free_charge_enabled or
                ((summary.get('spray') or {}).get('electric_coupling') or {}).get('model_version')==CHARGE_MODEL_VERSION)
            if data.spray is not None and summary.get('spray') is not None and phase_compatible:
                sp=data.spray
                restored=SimpleNamespace(radial_edges=sp['radial_edges_m'],axial_edges=sp['axial_edges_m'],times=sp['time_s'],
                    phase=sp['liquid_fraction'],active=sp['active'],velocity_r=sp['velocity_r_m_s'],velocity_z=sp['velocity_z_m_s'],
                    pressure=sp['pressure_pa'],velocity_r_history=sp['velocity_r_history_m_s'],velocity_z_history=sp['velocity_z_history_m_s'],
                    pressure_history=sp['pressure_history_pa'],diagnostics=summary['spray'])
                self.cache.remember('spray',keys['spray'],restored)
                for key,name in {'charge_density_c_m3_history':'charge_density_history','potential_v_history':'potential_history',
                    'field_r_v_m_history':'field_r_history','field_z_v_m_history':'field_z_history',
                    'supply_current_a_history':'supply_current_history','inlet_voltage_v_history':'inlet_voltage_history',
                    'methanol_fraction_history':'methanol_history','salt_mol_m3_history':'salt_history'}.items():
                    if key in sp:setattr(restored,name,sp[key])
                from ..spray.droplet import identify
                from ..numerics import metric
                rc,zc,_,_,vol=metric(restored.radial_edges,restored.axial_edges)
                restored.droplets=identify(restored.phase[-1],vol,rc,zc)
        return True

    def run(self,config,progress=lambda s:None):
        from ..engine import liquid_solution,SimulationResult
        from ..electro.electrostatics import solve as electric_solve
        from ..particle.cell_transport import simulate
        from ..particle.events import simulate_events
        from ..spray.level_set import simulate as spray_solve
        with self.lock:
            c=deepcopy(config).validate();keys=stage_keys(c)
            def liquid(point):
                def compute():
                    f,s,co=liquid_solution(point,progress)
                    return f,s,co,point.fluid.sample_flow_ul_min
                f,s,co,q=self.cache.get_or_compute('liquid',stage_keys(point)['liquid'],compute,progress)
                point.fluid.sample_flow_ul_min=q
                return f,s,co
            flow,solvent,coupling=liquid(c)
            electric=self.cache.get_or_compute('electric',keys['electric'],
                lambda:electric_solve(c,getattr(solvent,'ionic',None)),progress)
            particles=self.cache.get_or_compute('particles',keys['particles'],
                lambda:simulate(c,solvent,electric,progress=progress),progress)
            if c.electrical.liquid_model=='conducting' or c.electrical.voltage_kv==0:zero=particles
            else:
                zc=deepcopy(c);zc.electrical.voltage_kv=0.;zc.electrical.spray_current_na=0.
                _,zs,_=liquid(zc);zk=stage_keys(zc)
                ze=self.cache.get_or_compute('electric',zk['electric'],
                    lambda:electric_solve(zc,getattr(zs,'ionic',None)),progress)
                zero=self.cache.get_or_compute('particles',zk['particles'],
                    lambda:simulate(zc,zs,ze,progress=progress),progress)
            events=self.cache.get_or_compute('events',keys['events'],lambda:simulate_events(c,particles),progress)
            spray=None
            if c.spray.enabled:
                spray=self.cache.get_or_compute('spray',keys['spray'],lambda:spray_solve(c,solvent,electric,progress),progress)
            warnings=[
                '使用现有有限元,  Species ,  Electric field , 有限球与两相求解器; 缓存只复用相同依赖条件的结果.',
                '基础有限球运输为独立稀相响应; 细胞间作用和可逆黏附可在共用物理模块的局部连续流页另算.细胞变形与真实裂解率未解析.',
                '甲醇暴露是外液浓度, 不等于提取率;  Material properties , 壁面捕获和膜参数需校准.',
                '稳态管内出口采用零牵引边界, 尚未接收两相瞬态背压.',
                '喷口出管, 条件滴入孔, 纯来源采集机会与实际质谱检出率是不同指标.',
                '当前场与出口轨迹仍需网格和时间步收敛; 短时两相回放不代表稳定喷雾.']
            if not coupling['converged']:warnings.append('载液 /  Species  / 离子耦合未达到设定容差.')
            ionic=getattr(solvent,'ionic',None)
            if ionic is not None:warnings.extend(ionic.diagnostics.get('limitations',[]))
            if c.electrical.spray_current_na is None:warnings.append('喷雾电流未知, 远端供电采用开路极限.')
            if spray is not None:warnings.extend(spray.diagnostics.get('limitations',[]))
            return SimulationResult(c,flow,solvent,electric,particles,zero,events,spray,coupling,warnings)


def save_numeric(result,directory):
    """Write a new compatible numeric record; encoding views is a separate action."""
    from pathlib import Path
    from ..visualization.explorer_data import ExplorerData
    from ..report.export import clean,write_csv
    from ..engine import derived
    path=Path(directory).resolve();path.mkdir(parents=True,exist_ok=True)
    if (path/'fields.npz').exists():
        raise FileExistsError('A numeric record already exists here; save to a new result directory.')
    def document(name,value):
        temporary=path/(name+'.partial.json')
        temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(path/(name+'.json'))
    # A result is published to the library only after every numeric companion
    # has been committed. Interrupted writes remain visibly incomplete.
    document('record',dict(format_version=1,status='writing'))
    data=ExplorerData.from_result(result,str(path));result.config.save(path/'config.json')
    def archive(name,**arrays):
        temporary=path/(name+'.partial.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path/(name+'.npz'))
    summary=dict(software='CyESI Workbench',derived=derived(result),flow=result.flow.diagnostics,
        solvent=result.solvent.diagnostics,electric=result.electric.diagnostics,particles=result.particles.diagnostics,
        events={k:v for k,v in result.events.items() if not isinstance(v,np.ndarray)},
        coupling=result.coupling,spray=None if result.spray is None else result.spray.diagnostics,warnings=result.warnings)
    document('summary',summary)
    ionic=getattr(result.solvent,'ionic',None)
    if ionic is not None:
        archive('ions',radial_edges_m=ionic.radial_edges,axial_edges_m=ionic.axial_edges,
            salt_mol_m3=ionic.salt_mol_m3,potential_v=ionic.potential,field_r_v_m=ionic.field_r,field_z_v_m=ionic.field_z,
            current_radial_a=ionic.current_r_a,current_axial_a=ionic.current_z_a,conductivity_s_m=ionic.conductivity,active=ionic.active)
    if data.spray is not None:archive('spray',**data.spray)
    p=result.particles
    write_csv(path/'cells.csv',({'cell_id':i,'status':int(p.status[i]),'residence_s':p.residence_time[i],
        'outlet_r_um':p.outlet_radius[i]*1e6,'outlet_methanol_fraction':p.outlet_methanol[i]} for i in range(result.config.cell.count)))
    # Legacy result discovery uses fields.npz as its index. Commit it last so
    # those readers also see all companions, even without understanding record.json.
    archive('fields',**data.fields)
    data.summary=deepcopy(summary);data._transport_summary=deepcopy(summary)
    document('record',dict(format_version=1,status='complete',files=[p.name for p in path.iterdir() if p.name!='record.json']))
    return data
