"""One-at-a-time and full-factorial parameter sweeps.

Voltage changes recompute resistive and electroosmotic fields. Internal liquid
flow excludes gas-pressure feedback; stress is reported as a physical observable.
"""
from copy import deepcopy
from itertools import product
import numpy as np
from ..engine import liquid_solution
from ..electro.electrostatics import solve as solve_electric, EPS0
from ..fluid.navier_stokes import annular_gas
from ..particle.cell_transport import simulate

SWEEPS = {"sheath_flow_ul_min": [5., 10., 20., 30., 50.],
          "sample_flow_ul_min": [.1, .5, 1., 2., 5.],
          "voltage_kv": [0., 2., 3., 4.5, 6.],
          "gas_pressure_kpa": [.1, .4, 1., 2.]}


def scan(config, full=False, cells=512, progress=lambda x: None):
    names = list(SWEEPS)
    base = [getattr(config.electrical if k == "voltage_kv" else config.fluid, k) for k in names]
    cases = []
    if full:
        cases = [("factorial", dict(zip(names, values))) for values in product(*(SWEEPS[k] for k in names))]
    else:
        for name in names:
            for value in SWEEPS[name]:
                params = dict(zip(names, base)); params[name] = value
                cases.append((name, params))
    from ..performance import worker_budget,process_jobs
    groups={}
    for index,(axis,params) in enumerate(cases):
        point=_point(config,axis,params,cells)
        groups.setdefault(_flow_key(point),[]).append((index,axis,params))
    workers=worker_budget(config.numerics.workers,len(groups))
    if workers>1:
        progress(f'Parallel parameter scan: {workers} processes, {len(groups)} unique flow groups')
        completed={}
        jobs=[(deepcopy(config),group,cells) for group in groups.values()]
        for _,result in process_jobs(_scan_job,jobs,workers,progress):
            completed.update(result);progress(f'Parallel scan completed {len(completed)}/{len(cases)} cases')
        rows=[completed[index] for index in range(len(cases))]
    else:
        cache={};rows=[]
        for index,(axis,params) in enumerate(cases):
            progress(f'Scan {index+1}/{len(cases)}: {axis}, {params}')
            rows.append(_evaluate(config,axis,params,cells,cache,progress))
    for row in rows:row['execution_workers']=workers
    # Pareto candidates only among independent sample/sheath changes (or all factorial cases).
    good = [r for r in rows if r["status"] == "ok" and r["coupling_converged"] and r["mean_residence_s"] is not None
            and r["sweep"] in ("factorial", "sample_flow_ul_min", "sheath_flow_ul_min")]
    for row in rows:
        row["pareto_candidate"] = False
    for row in good:
        vector = np.array([-row["transport_fraction"], row["outlet_r95_um"], row["mean_residence_s"]])
        dominated = False
        for other in good:
            ov = np.array([-other["transport_fraction"], other["outlet_r95_um"], other["mean_residence_s"]])
            if np.all(ov <= vector) and np.any(ov < vector): dominated = True; break
        row["pareto_candidate"] = not dominated
    return rows


def _point(config,axis,params,cells):
    point=deepcopy(config);point.cell.count=cells;point.numerics.saved_trajectories=min(8,cells)
    for k,v in params.items():setattr(point.electrical if k=='voltage_kv' else point.fluid,k,v)
    if axis in ('sample_flow_ul_min','factorial'):point.supply.sample_control='flow'
    return point


def _flow_key(point):
    return (point.fluid.qs,point.fluid.qh,point.electrical.voltage_kv,point.supply.sample_control)


def _scan_job(payload):
    from ..performance import emit
    config,group,cells=payload;config.numerics.workers=1;cache={};rows={}
    for index,axis,params in group:
        emit(f'Parallel scan case {index+1}: {axis}')
        rows[index]=_evaluate(config,axis,params,cells,cache,emit)
    return rows


def _evaluate(config,axis,params,cells,cache,progress):
    point=_point(config,axis,params,cells);key=_flow_key(point)
    try:
        if key not in cache:
            flow,solvent,coupling=liquid_solution(point,progress)
            electric=solve_electric(point,getattr(solvent,'ionic',None))
            particles=simulate(point,solvent,electric,progress=progress)
            local_e=float(np.hypot(electric.sample(np.array(0.),np.array(50e-6),'field_r'),electric.sample(np.array(0.),np.array(50e-6),'field_z')))
            cache[key]=(particles.diagnostics,coupling,solvent.diagnostics,local_e,point.fluid.sample_flow_ul_min)
        p,coupling,solvent_diag,E,actual_q=cache[key]
        point.fluid.sample_flow_ul_min=actual_q;gas=annular_gas(point)
        rate=point.cell.concentration_per_ml*actual_q/60000*p['transport_fraction']
        return {'sweep':axis,**params,'cells':cells,'status':'ok',
            'sample_control':point.supply.sample_control,'actual_sample_flow_ul_min':actual_q,
            'transport_fraction':p['transport_fraction'],'focusing_fraction':p['focusing_fraction_of_exits'],
            'outlet_r95_um':p['outlet_r95_um'],'mean_residence_s':p['mean_residence_s'],
            'mean_outlet_methanol_fraction':p['mean_outlet_methanol_fraction'],
            'steady_nozzle_rate_cells_s':rate,'ideal_poisson_one_second_cv':1/np.sqrt(rate) if rate>0 else None,
            'gas_weber':point.fluid.gas_density*gas['mean_velocity_m_s']**2*2*point.geometry.c/point.fluid.surface_tension,
            'electric_bond_at_axis_50um':EPS0*E**2*point.geometry.c/point.fluid.surface_tension,
            'coupling_converged':coupling['converged'],
            'max_flux_projection_correction':solvent_diag['max_fem_flux_projection_relative_correction'],
            'detection_efficiency':None,'note':'Upstream metrics only; gas/voltage stability is not inferred from stress numbers.'}
    except (ValueError,RuntimeError) as exc:
        return {'sweep':axis,**params,'cells':cells,'status':'failed','error':str(exc)}
