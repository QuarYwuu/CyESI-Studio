"""Read simulation arrays and provide unit conversions for display.

Field and trajectory views use solved data. Mechanism illustrations use
separately specified display parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import csv
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ..config import Config
from ..numerics import metric
from ..spray.droplet import identify


@dataclass
class CellSnapshot:
    time_s: float
    r_m: float
    z_m: float
    methanol: float
    dose_fraction_s: float


@dataclass
class ExplanationSettings:
    """Parameters for the optional mechanism illustration."""
    jet_diameter_um: float = 8.0
    lysis_rate_s: float = 20.0
    extraction_rate_s: float = 50.0
    extra_exposure_ms: float = 100.0
    log_radius_spread: float = .12

    def validate(self):
        values = list(vars(self).values())
        if not np.isfinite(values).all() or min(values) < 0 or self.jet_diameter_um <= 0:
            raise ValueError("Assumption parameters must be finite and nonnegative; jet diameter > 0")
        return self


def illustrative_cell_response(time_s, concentration, settings):
    """Phenomenological, uncalibrated membrane/extraction model.

    dI/dt=-k_l C^2 I; dM/dt=-k_e C(1-I)M. I is a normalized
    membrane-integrity variable, not a measured probability of lysis.
    M + released = 1 exactly. Default rigid-sphere simulation never uses this.
    """
    settings.validate()
    t, C = np.asarray(time_s, float), np.asarray(concentration, float)
    if t.ndim != 1 or t.shape != C.shape or not len(t) or np.any(np.diff(t) < 0):
        raise ValueError("Exposure history must be a nondecreasing 1D time series")
    if not np.isfinite(t).all() or not np.isfinite(C).all() or np.any((C < 0) | (C > 1)):
        raise ValueError("Invalid exposure history")
    dt = np.diff(t, prepend=t[0])
    midpoint = (C+np.r_[C[0], C[:-1]])/2
    integrity = np.exp(-settings.lysis_rate_s*np.cumsum(midpoint**2*dt))
    midpoint_integrity = (integrity+np.r_[integrity[0], integrity[:-1]])/2
    remaining = np.exp(-settings.extraction_rate_s*np.cumsum(midpoint*(1-midpoint_integrity)*dt))
    return integrity, remaining, 1-remaining


def rayleigh_reference(jet_diameter_um, liquid_flow_m3_s):
    """Inviscid uncharged cylinder reference, not an ESI drop-size prediction.

    Fastest reference wavelength = 9.02 R; one wavelength's cylinder volume
    is assigned to a sphere. Electric stress, viscosity, satellites and charge
    fission are absent. Animation timing is deliberately not inferred here.
    """
    if jet_diameter_um <= 0 or liquid_flow_m3_s < 0:
        raise ValueError("Invalid cylinder reference inputs")
    radius = jet_diameter_um*.5e-6
    wavelength = 9.02*radius
    volume = np.pi*radius**2*wavelength
    drop_radius = (3*volume/(4*np.pi))**(1/3)
    return {"jet_radius_m": radius, "wavelength_m": wavelength,
            "drop_radius_m": drop_radius, "drop_volume_m3": volume,
            "reference_frequency_s": liquid_flow_m3_s/volume,
            "reference_jet_speed_m_s": liquid_flow_m3_s/(np.pi*radius**2)}


class ExplorerData:
    def __init__(self, config, fields, summary=None, spray=None, source=""):
        self.config = deepcopy(config).validate()
        self.rate_config = deepcopy(self.config)
        self.rate_preview = False
        self.fields = fields
        self.summary = deepcopy(summary or {})
        self._transport_summary = deepcopy(self.summary)
        self.spray = spray
        self.source = str(source)
        self._interpolators = {}
        self._tracks = {}
        self._droplets = {}
        self.r = self._centers(fields["liquid_r_edges_m"])
        self.z = self._centers(fields["liquid_z_edges_m"])
        self.er = self._centers(fields["electric_r_edges_m"])
        self.ez = self._centers(fields["electric_z_edges_m"])
        self.ids = np.asarray(fields["trajectory_ids"], dtype=int)
        shape = (len(self.r), len(self.z))
        for name in ["liquid_active", "velocity_r_m_s", "velocity_z_m_s", "pressure_pa", "methanol_fraction"]:
            if fields[name].shape != shape: raise ValueError(f"Invalid field dimensions: {name}")
        ts = fields["trajectory_time_s"]
        if ts.ndim != 2 or ts.shape[1] != len(self.ids) or len(self.ids) == 0:
            raise ValueError("No readable saved cell trajectories")
        for name in ["trajectory_r_m", "trajectory_z_m", "trajectory_methanol_fraction"]:
            if fields[name].shape != ts.shape: raise ValueError(f"Invalid trajectory dimensions: {name}")
        for edges in [fields["liquid_r_edges_m"], fields["liquid_z_edges_m"],
                      fields["electric_r_edges_m"], fields["electric_z_edges_m"]]:
            if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError("Grid edges must be finite and strictly increasing")
        if spray is not None:
            expected = (len(spray["time_s"]), len(spray["radial_edges_m"])-1, len(spray["axial_edges_m"])-1)
            if spray["liquid_fraction"].shape != expected:
                raise ValueError("Invalid phase dimensions")

    @staticmethod
    def _centers(edges): return (edges[:-1]+edges[1:])/2

    @classmethod
    def load(cls, directory):
        path = Path(directory).resolve()
        config = Config.load(path/"config.json")
        with np.load(path/"fields.npz", allow_pickle=False) as arrays:
            fields = {key: arrays[key].copy() for key in arrays.files}
        summary = json.loads((path/"summary.json").read_text(encoding="utf-8")) if (path/"summary.json").exists() else {}
        spray = None
        if (path/"spray.npz").exists():
            with np.load(path/"spray.npz", allow_pickle=False) as arrays:
                spray = {key: arrays[key].copy() for key in arrays.files}
        return cls(config, fields, summary, spray, path)

    @classmethod
    def from_result(cls, result, source=" Current results "):
        s, e, p = result.solvent, result.electric, result.particles
        fields = dict(liquid_r_edges_m=s.radial_edges, liquid_z_edges_m=s.axial_edges,
            liquid_active=s.active, velocity_r_m_s=s.velocity_r, velocity_z_m_s=s.velocity_z,
            pressure_pa=s.pressure, methanol_fraction=s.concentration,
            electric_r_edges_m=e.radial_edges, electric_z_edges_m=e.axial_edges,
            potential_v=e.potential, field_r_v_m=e.field_r, field_z_v_m=e.field_z,
            trajectory_ids=p.saved_ids, trajectory_time_s=p.trajectory_time,
            trajectory_r_m=p.trajectory_r, trajectory_z_m=p.trajectory_z,
            trajectory_methanol_fraction=p.trajectory_methanol,
            response_residence_time_s=p.residence_time, response_exited=p.exited)
        ionic=getattr(s,"ionic",None)
        if getattr(p,'bulk_history',None):
            fields.update({f'bulk_trajectory_{key}':value for key,value in p.bulk_history.items()})
        if getattr(p,'stress',None):
            fields.update({f'cell_{key}':value for key,value in p.stress.items()})
        if hasattr(s,'equivalent_stress_pa'):
            fields['equivalent_stress_pa']=s.equivalent_stress_pa
            fields['axial_extension_stress_pa']=s.axial_extension_stress_pa
        if ionic is not None:
            fields.update(liquid_field_r_v_m=ionic.field_r,liquid_field_z_v_m=ionic.field_z,liquid_potential_v=ionic.potential)
        spray = None
        if result.spray is not None:
            sp = result.spray
            spray = dict(radial_edges_m=sp.radial_edges, axial_edges_m=sp.axial_edges,
                time_s=sp.times, liquid_fraction=sp.phase, active=sp.active,
                velocity_r_m_s=sp.velocity_r, velocity_z_m_s=sp.velocity_z, pressure_pa=sp.pressure,
                velocity_r_history_m_s=sp.velocity_r_history,velocity_z_history_m_s=sp.velocity_z_history,
                pressure_history_pa=sp.pressure_history)
            from ..spray.level_set import electrical_arrays
            spray.update(electrical_arrays(sp))
        # Live views need the same provenance as numeric exports: otherwise a
        # solver residual, incomplete spray or unconverged carrier disappears
        # merely because the result was opened without writing it to disk.
        summary={"particles":p.diagnostics,"electric":e.diagnostics}
        for name in ("flow", "solvent", "spray"):
            item = getattr(result, name, None)
            if item is not None and hasattr(item, "diagnostics"):
                summary[name] = item.diagnostics
        if hasattr(result, "coupling"):
            summary["coupling"] = result.coupling
        if hasattr(result, "warnings"):
            summary["warnings"] = result.warnings
        if hasattr(result,"events"):
            summary['events']={k:v for k,v in result.events.items() if not isinstance(v,np.ndarray)}
        return cls(result.config, fields, summary, spray, source)

    def with_concentration(self, value):
        """Resample continuous events, sharing unchanged dilute CFD responses.

        Existing files and source objects are immutable. Older exports provide
        the complete response pool through cells.csv, not just animated tracks.
        """
        from ..particle.events import simulate_events
        config = deepcopy(self.config)
        config.cell.concentration_per_ml = float(value)
        config.validate()
        if config.cell.concentration_per_ml == self.config.cell.concentration_per_ml:
            return self
        fields = self.fields
        if "response_residence_time_s" not in fields:
            path = Path(self.source)/"cells.csv"
            if not path.exists():
                raise ValueError("Complete transport responses require cells.csv. Run a simulation before updating concentration.")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            fields = dict(fields, response_residence_time_s=np.array([float(x["residence_s"]) for x in rows]),
                          response_exited=np.array([int(x["status"]) == 1 for x in rows]))
        responses = SimpleNamespace(residence_time=fields["response_residence_time_s"],
                                    exited=fields["response_exited"])
        events = simulate_events(config, responses)
        summary = deepcopy(self.summary)
        summary["events"] = {k:v for k,v in events.items() if not isinstance(v,np.ndarray)}
        return ExplorerData(config, fields, summary, self.spray, self.source)

    def with_rate_inputs(self,edited):
        """Live source-rate preview while preserving the solved flow's inputs.

        Concentration alone can reuse transport exactly under the dilute model.
        Changing sample flow invalidates transit times and possibly exit fraction:
        only expected rates are previewed, explicitly using the last exit fraction.
        Finite-window observations are withheld until a fresh solve.
        """
        data=self.with_concentration(edited.cell.concentration_per_ml)
        if data.rate_config.fluid.sample_flow_ul_min==edited.fluid.sample_flow_ul_min:return data
        updated=ExplorerData(data.config,data.fields,data._transport_summary,data.spray,data.source)
        updated.rate_config.fluid.sample_flow_ul_min=edited.fluid.sample_flow_ul_min
        updated.rate_config.validate()
        updated.rate_preview=updated.config.fluid.sample_flow_ul_min!=edited.fluid.sample_flow_ul_min
        if updated.rate_preview:
            updated.summary['events']={'summaries':[],
                'note':'Sample flow edited; expected rate preview uses the last modeled exit fraction. Transit/finite-window statistics require a new CFD solve.'}
        return updated

    def sample(self, r, z, name):
        electric = name in ("potential_v", "field_r_v_m", "field_z_v_m")
        axes = (self.er, self.ez) if electric else (self.r, self.z)
        if name not in self._interpolators:
            values = self.fields[name]
            if not electric and name != "liquid_active":
                from ..chemistry.solvent_transport import filled
                values = filled(values, self.fields["liquid_active"])
            self._interpolators[name] = RegularGridInterpolator(axes, values, bounds_error=False, fill_value=None)
        r, z = np.broadcast_arrays(r, z)
        points = np.c_[np.clip(np.abs(r.ravel()), axes[0][0], axes[0][-1]),
                       np.clip(z.ravel(), axes[1][0], axes[1][-1])]
        values = self._interpolators[name](points).reshape(r.shape)
        if name in ("velocity_r_m_s", "field_r_v_m"): values = values*np.where(r < 0, -1., 1.)
        if name.startswith("field_") and self.config.electrical.liquid_model == "conducting":
            values = np.where((np.abs(r) < self.config.geometry.c) & (z <= 0), 0., values)
        local_name={"field_r_v_m":"liquid_field_r_v_m","field_z_v_m":"liquid_field_z_v_m","potential_v":"liquid_potential_v"}.get(name)
        if local_name in self.fields:
            local=self.sample(r,z,local_name)
            if name=="field_r_v_m":local*=np.where(r<0,-1.,1.)
            values=np.where((abs(r)<self.config.geometry.c)&(z<=0),local,values)
        return values

    def track(self, index):
        index = int(index) % len(self.ids)
        if index not in self._tracks:
            f = self.fields
            t = f["trajectory_time_s"][:, index]
            valid = np.isfinite(t) & np.isfinite(f["trajectory_z_m"][:, index]) & np.isfinite(f["trajectory_r_m"][:, index])
            t = t[valid]
            if len(t) < 2: raise ValueError("Trajectory has too few finite frames")
            # Keep the last sample at duplicated timestamps (resting/terminal frames).
            indices = np.flatnonzero(np.r_[np.diff(t) > 1e-12, True])
            t = t[indices]
            r = f["trajectory_r_m"][valid, index][indices]
            z = f["trajectory_z_m"][valid, index][indices]
            C = np.clip(f["trajectory_methanol_fraction"][valid, index][indices], 0, 1)
            bulk_key='bulk_trajectory_time_s'
            if bulk_key in f:
                matches=np.flatnonzero(f['bulk_trajectory_ids']==self.ids[index])
                if len(matches):
                    j=int(matches[0]);bt=f[bulk_key][:,j]
                    keep=np.r_[np.diff(bt)>1e-12,True];bt=bt[keep]
                    br=np.hypot(f['bulk_trajectory_x_m'][:,j],f['bulk_trajectory_y_m'][:,j])[keep]
                    bz=(self.config.geometry.inner_tip-self.config.geometry.sample_length_mm*1e-3
                        +f['bulk_trajectory_distance_m'][:,j])[keep]
                    after=t>bt[-1]+1e-12
                    t=np.r_[bt,t[after]];r=np.r_[br,r[after]];z=np.r_[bz,z[after]];C=np.r_[np.zeros(len(bt)),C[after]]
            if np.any(np.diff(t) < 0): raise ValueError("Trajectory time decreases")
            dose = np.r_[0., np.cumsum((C[1:]+C[:-1])*.5*np.diff(t))]
            bulk_time = float(f["trajectory_time_s"][1, index])
            lip = self.config.geometry.inner_tip
            cross = np.flatnonzero(z >= lip)
            inner_time = float(t[-1])
            if len(cross):
                k = cross[0]
                inner_time = float(t[k]) if k == 0 else float(np.interp(lip, z[k-1:k+1], t[k-1:k+1]))
            self._tracks[index] = dict(t=t, r=r, z=z, C=C, dose=dose,
                bulk_time=max(0., min(bulk_time, t[-1])), inner_time=inner_time, end_time=float(t[-1]))
        return self._tracks[index]

    def snapshot(self, index, time_s):
        track = self.track(index)
        t = np.clip(time_s, track["t"][0], track["t"][-1])
        return CellSnapshot(float(t), *(float(np.interp(t, track["t"], track[key])) for key in ["r", "z", "C", "dose"]))

    def progress_time(self, index, progress):
        track = self.track(index)
        # Allocate screen time to the long tube, recessed tip, and mixing region.
        return float(np.interp(np.clip(progress, 0, 1), [0., .32, .70, 1.],
            [0., track["bulk_time"], track["inner_time"], track["end_time"]]))

    def progress_at_time(self, index, time_s):
        track = self.track(index)
        return float(np.interp(time_s, [0., track["bulk_time"], track["inner_time"], track["end_time"]], [0., .32, .70, 1.]))

    def continuous_cells(self, time_s):
        """Stationary Poisson feed at the resolved-region entrance, no recycling.

        Saved trajectories are response samples. Every arrival has its own ID;
        crossing the lip transfers a cell to the downstream unresolved ledger.
        """
        if not hasattr(self,"_source"):
            from ..particle.events import ContinuousCellStream
            durations=np.array([self.track(i)["end_time"]-self.track(i)["bulk_time"] for i in range(len(self.ids))])
            exited=np.array([self.snapshot(i,self.track(i)["end_time"]).z_m>=-1e-9 for i in range(len(self.ids))])
            rate=self.rate_config.cell.concentration_per_ml*self.rate_config.fluid.sample_flow_ul_min/60000
            self._source=ContinuousCellStream(rate,durations,exited,self.config.numerics.seed+13,-float(durations.max()))
        records=self._source.extend(max(0.,time_s)); visible=[]
        in_pipe=(records["nozzle_s"]>time_s)|np.isnan(records["nozzle_s"])
        for k in np.flatnonzero(in_pipe):
            index=int(records["template_id"][k]); age=self.track(index)["bulk_time"]+time_s-records["injection_s"][k]
            if age>self.track(index)["end_time"]:continue  # terminal unresolved, counted in ledger
            visible.append(dict(event_id=int(records["event_id"][k]),index=index,snapshot=self.snapshot(index,age)))
        return visible, {"created":len(records["event_id"]),"in_pipe":int(in_pipe.sum()),
            "downstream":int(np.sum(np.isfinite(records["nozzle_s"])&(records["nozzle_s"]<=time_s))),
            "downstream_window":int(np.sum(np.isfinite(records["nozzle_s"])&(records["nozzle_s"]>=0)&(records["nozzle_s"]<=time_s))),
            "rate_cells_s":self._source.rate}

    def spray_sample(self,r,z,name,frame):
        sp=self.spray
        if sp is None or name not in sp:return np.full(np.broadcast(r,z).shape,np.nan)
        r,z=np.broadcast_arrays(r,z); values=sp[name]
        if values.ndim==3:values=values[int(frame)]
        key=("spray",name,int(frame) if sp[name].ndim==3 else -1)
        if key not in self._interpolators:
            self._interpolators[key]=RegularGridInterpolator((self._centers(sp["radial_edges_m"]),self._centers(sp["axial_edges_m"])),values,bounds_error=False,fill_value=np.nan)
        out=self._interpolators[key](np.c_[abs(r.ravel()),z.ravel()]).reshape(r.shape)
        if name.startswith("velocity_r"):out*=np.where(r<0,-1.,1.)
        # Bound history interpolator memory during extended playback.
        if len(self._interpolators)>40:
            for old in list(self._interpolators):
                if isinstance(old,tuple) and old!=key:del self._interpolators[old]
        return out

    def explanation_response(self, index, fraction, settings):
        track = self.track(index)
        # Only a hypothetical post-nozzle hold is appended to the actual exposure.
        dt = settings.extra_exposure_ms*1e-3*np.clip(fraction, 0, 1)
        time = np.r_[track["t"], track["t"][-1]+np.linspace(0, dt, 41)[1:]]
        concentration = np.r_[track["C"], np.full(40, track["C"][-1])]
        integrity, remaining, released = illustrative_cell_response(time, concentration, settings)
        return float(integrity[-1]), float(remaining[-1]), float(released[-1])

    def droplet_frame(self, frame):
        if self.spray is None: return []
        frame = int(np.clip(frame, 0, len(self.spray["time_s"])-1))
        if frame not in self._droplets:
            sp = self.spray
            r,z,_,_,volumes = metric(sp["radial_edges_m"], sp["axial_edges_m"])
            self._droplets[frame] = identify(sp["liquid_fraction"][frame], volumes, r, z)
            h=min(np.diff(sp["radial_edges_m"]).min(),np.diff(sp["axial_edges_m"]).min())
            for drop in self._droplets[frame]:
                drop["resolved"] = bool(drop["cells"]>=4 and drop["equivalent_radius_um"]*1e-6>=2*h and not drop["touches_outer_boundary"])
        return self._droplets[frame]

    @property
    def default_cell(self):
        times = np.array([self.track(i)["end_time"] for i in range(len(self.ids))])
        return int(np.argmin(abs(times-np.median(times))))


def find_results(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    candidates = []
    outputs = root/"outputs"
    if outputs.exists(): candidates += list(outputs.iterdir())
    valid=[p for p in candidates if (p/"fields.npz").exists() and (p/"config.json").exists()]
    # Analysis subfolders change a directory's mtime without a new CFD solve.
    # Prefer the newest numeric result, whether or not spray was enabled.
    return max(valid,key=lambda p:(p/'fields.npz').stat().st_mtime) if valid else None
