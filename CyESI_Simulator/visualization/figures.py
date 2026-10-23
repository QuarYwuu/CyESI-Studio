"""Source-backed plots. Labels explicitly separate prediction from assumptions."""
from pathlib import Path
from functools import wraps
from threading import RLock
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from ..geometry.coaxial_geometry import classify

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 150})
BLUE, GREEN, ORANGE = "#2764a5", "#149788", "#cf6a28"
_PLOT_LOCK=RLock()


def serialized_plotting(function):
    """Pyplot's process-wide state must not race between GUI workers."""
    @wraps(function)
    def render(*args,**kwargs):
        with _PLOT_LOCK:return function(*args,**kwargs)
    return render


def _map(ax, re, ze, field, label, cmap="viridis", active=None, zlim=None):
    values = np.ma.masked_where(~active, field) if active is not None else field
    p = ax.pcolormesh(ze*1e3, re*1e6, values, shading="flat", cmap=cmap)
    ax.set(xlabel="z (mm), middle lip = 0", ylabel="r (um)")
    if zlim is not None: ax.set_xlim(*zlim)
    ax.figure.colorbar(p, ax=ax, label=label, pad=.02)
    return p


def _new(title, rows=1, cols=1, size=(10, 5)):
    fig, ax = plt.subplots(rows, cols, figsize=size, layout="constrained", squeeze=False)
    fig.suptitle(title, fontweight="bold", fontsize=14)
    return fig, ax.ravel()


def geometry(result):
    c = result.config; g = c.geometry
    fig, axes = _new("Figure 1 | CyESI geometry and axial offsets", cols=2, size=(12, 4.5))
    r = np.unique(np.r_[np.linspace(0, g.gas_od_um*.5e-6*1.1, 200), g.a, g.b, g.c, g.d])
    z = np.linspace(-max(g.inner_recess_mm, g.gas_recess_mm, 1)*1.5e-3, .3e-3, 300)
    R, Z = np.meshgrid((r[:-1]+r[1:])/2, (z[:-1]+z[1:])/2, indexing="ij")
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([BLUE, GREEN, "#c4d9ed", "#aaaeb6", "#f4f4f4"])
    p = axes[0].pcolormesh(z*1e3, r*1e6, classify(R, Z, g), cmap=cmap,
                         norm=BoundaryNorm(np.arange(.5, 6), 5), shading="flat")
    axes[0].set(xlabel="z (mm)", ylabel="r (um)", title="Axisymmetric cross-section; physical dimensions")
    for position, name in [(g.inner_tip*1e3, "inner"), (-g.gas_recess_mm, "gas"), (0, "sheath")]:
        axes[0].axvline(position, color="#555", lw=.7, ls="--")
        axes[0].text(position, r[-1]*1e6, name, rotation=90, va="top", ha="right")
    cb = fig.colorbar(p, ax=axes[0], ticks=np.arange(1, 6), pad=.02)
    cb.ax.set_yticklabels(["sample", "sheath", "N2", "quartz", "air"])
    axes[1].axis("off")
    axes[1].text(.03, .93,
        f"Sample liquid: {c.chemistry.sample_medium}\nSample tube traversed: {g.sample_length_mm:g} mm\n"
        f"Developed segment: {g.bulk_length*1e3:g} mm\nTip FEM: {g.inner_tip*1e3:g} to {g.liquid_start*1e3:g} mm + mixing gap\n\n"
        f"Sample / sheath flow: {c.fluid.sample_flow_ul_min:g} / {c.fluid.sheath_flow_ul_min:g} uL/min\n"
        f"Methanol inlet fraction: {c.chemistry.sheath_methanol_fraction:.0%}\n"
        f"Inner / gas recess: {g.inner_recess_mm:g} / {g.gas_recess_mm:g} mm\n"
        f"MS plane: +{g.ms_gap_mm:g} mm\n\n"
        "Quartz is a dielectric. Conducting liquid is the HV electrode.\nGas tube OD and supply length are assumed inputs.", va="top", linespacing=1.7)
    return fig


def velocity(result):
    s = result.solvent
    fig, axes = _new("Figure 2 | Solved liquid velocity and pressure", cols=2, size=(12, 4.7))
    _map(axes[0], s.radial_edges, s.axial_edges, np.hypot(s.velocity_r, s.velocity_z)*1e3,
         "Speed (mm/s)", active=s.active, zlim=(-1.4, 0))
    _map(axes[1], s.radial_edges, s.axial_edges, s.pressure, "Pressure (Pa)", "magma", s.active)
    axes[1].set_title("Resolved tip; developed bulk pressure drop reported separately")
    return fig


def focusing(result):
    s, p = result.solvent, result.particles
    fig, axes = _new("Figure 3 | Hydrodynamic focusing and shear", cols=2, size=(12, 4.7))
    _map(axes[0], s.radial_edges, s.axial_edges, s.concentration, "Methanol fraction", "YlGnBu", s.active, (-1.3, 0))
    # The conservative streamfunction is cumulative axial volume flux / (2 pi).
    stream = np.cumsum(s.axial_flux[:, :-1], axis=0)
    R, Z = np.meshgrid(s.r*1e6, s.z*1e3, indexing="ij")
    axes[0].contour(Z, R, np.ma.masked_where(~s.active, stream), levels=12, colors="#333", linewidths=.5)
    mu = np.exp((1-s.concentration)*np.log(result.config.fluid.sample_viscosity)+s.concentration*np.log(result.config.fluid.sheath_viscosity))
    from ..chemistry.solvent_transport import filled
    from ..fluid.stress import axisymmetric_stress
    stress = axisymmetric_stress(s.r,s.z,filled(s.velocity_r,s.active),filled(s.velocity_z,s.active),mu)
    _map(axes[1], s.radial_edges, s.axial_edges, stress['equivalent_stress_pa'], "Equivalent viscous stress (Pa)", "magma", s.active, (-1.3, 0))
    axes[0].set_title(f"Particle outlet r95: {p.diagnostics['outlet_r95_um'] or float('nan'):.2f} um")
    return fig


def trajectories(result):
    p, g = result.particles, result.config.geometry
    fig, axes = _new("Figure 4 | Finite-sphere cell transport", cols=2, size=(12, 4.7))
    for j in range(min(40, len(p.saved_ids))):
        axes[0].plot(p.trajectory_z[:, j]*1e3, p.trajectory_r[:, j]*1e6, lw=.65, alpha=.6, color=BLUE)
    axes[0].set(xlim=(-1.4, 0), ylim=(0, g.c*1e6), xlabel="z (mm)", ylabel="Cell center r (um)")
    axes[0].axvline(g.inner_tip*1e3, ls="--", color="#888")
    axes[0].axhline(result.config.cell.diameter_um*.5, ls=":", color="#bbb", label="Cell radius")
    axes[1].hist(p.initial_radius*1e6, bins=24, density=True, alpha=.45, color=ORANGE, label="inlet, flux-weighted")
    if p.exited.any(): axes[1].hist(p.outlet_radius[p.exited]*1e6, bins=24, density=True, alpha=.6, color=GREEN, label="nozzle exits")
    axes[1].set(xlabel="r (um)", ylabel="Probability density (1/um)")
    axes[1].legend()
    return fig


def residence(result):
    p = result.particles
    fig, axes = _new("Figure 5 | Residence time and event arrivals", cols=2, size=(12, 4.7))
    if p.exited.any():
        axes[0].hist(p.residence_time[p.exited], bins=45, color=BLUE, alpha=.8, label="Applied voltage")
    zero = result.particles_zero_voltage
    if zero is not p and zero.exited.any(): axes[0].hist(zero.residence_time[zero.exited], bins=45, histtype="step", color=ORANGE, label="0 kV")
    elif zero is p:
        axes[0].text(.97,.97,"0 kV and applied voltage coincide\nin conducting upstream liquid",ha="right",va="top",transform=axes[0].transAxes,fontsize=9)
    axes[0].set(xlabel="Time from sample entrance to nozzle (s)", ylabel="Cells")
    axes[0].axvline(p.diagnostics["sample_volume_over_flow_s"], color=ORANGE, ls="--", label="tube volume / flow")
    axes[0].legend()
    event = result.events; horizon = max(result.config.events.durations_s)
    bins = np.arange(0, horizon+5, 5.)
    exits = event["nozzle_s"]; exits = exits[np.isfinite(exits) & (exits>=0) & (exits <= horizon)]
    axes[1].hist(exits, bins=bins, color=GREEN)
    axes[1].set(xlabel="Laboratory time after starting injection (s)", ylabel="Nozzle events / 5 s",
                title=f"Continuous Poisson feed {event['rate_cells_s']:.2f} cells/s; {result.config.events.startup} startup")
    return fig


def exposure(result):
    p, g = result.particles, result.config.geometry
    fig, axes = _new("Figure 6 | Extracellular methanol exposure", cols=2, size=(12, 4.7))
    for j in range(min(24, len(p.saved_ids))):
        z, t = p.trajectory_z[:, j], p.trajectory_time[:, j]
        cross = np.flatnonzero(z >= g.inner_tip)
        if not len(cross): continue
        k = cross[0]
        if k == 0: origin = t[k]
        else: origin = np.interp(g.inner_tip, z[k-1:k+1], t[k-1:k+1])
        axes[0].plot((t[k:]-origin)*1e3, p.trajectory_methanol[k:, j]*100, alpha=.5, lw=.8)
    axes[0].set(xlabel="Time since cell center crossed inner lip (ms)", ylabel="Methanol fraction (%)", ylim=(0, 100))
    if p.exited.any(): axes[1].scatter(p.tip_residence_time[p.exited]*1e3, p.outlet_methanol[p.exited]*100,
                                      s=4, alpha=.2, color=BLUE)
    axes[1].set(xlabel="Resolved tip residence (ms)", ylabel="Methanol fraction at nozzle (%)", ylim=(0, 100),
                title="Exposure is not a fitted lysis / extraction yield")
    return fig


def electric_field(result):
    e = result.electric
    fig, axes = _new("Figure 7 | Electric potential and field", cols=2, size=(12, 4.7))
    _map(axes[0], e.radial_edges, e.axial_edges, e.potential/1000, "Potential (kV)", "plasma", zlim=(-1.3, result.config.geometry.ms_gap_mm))
    _map(axes[1], e.radial_edges, e.axial_edges, np.hypot(e.field_r, e.field_z)/1e6, "|E| (kV/mm)", "inferno", zlim=(-.3, .8))
    axes[1].set_ylim(0, 500)
    axes[0].set_title(f"Liquid: {result.config.electrical.liquid_model}; grounded MS plane")
    return fig


def spray(result):
    s = result.spray
    fig, axes = _new("Figure 8 | Computed two-phase transient (experimental)", cols=3, size=(12, 4.7))
    if s is None:
        for ax in axes: ax.axis("off")
        axes[1].text(.5, .5, "Two-phase simulation disabled.\nNo Taylor cone / jet result is inferred.", ha="center")
        return fig
    for ax, k in zip(axes, np.linspace(0, len(s.times)-1, 3, dtype=int)):
        _map(ax, s.radial_edges, s.axial_edges, s.phase[k], "Liquid fraction", "Blues", s.active, (-.05, result.config.spray.length_mm))
        rr = (s.radial_edges[:-1]+s.radial_edges[1:])/2*1e6
        zz = (s.axial_edges[:-1]+s.axial_edges[1:])/2*1e3
        ax.contour(zz, rr, s.phase[k], levels=[.5], colors=ORANGE, linewidths=1)
        ax.set(title=f"t = {s.times[k]*1e3:.4f} ms", ylim=(0, result.config.spray.radius_mm*1000))
    return fig


def optimization(result):
    fig, axes = _new("Figure 9 | Parameter sweeps; no calibrated MS efficiency model", rows=2, cols=2, size=(11, 7))
    names = ["sheath_flow_ul_min", "sample_flow_ul_min", "voltage_kv", "gas_pressure_kpa"]
    for ax, name in zip(axes, names):
        rows = [r for r in (result.scan or []) if r["status"] == "ok" and r["sweep"] in (name,"factorial")]
        if not rows:
            ax.text(.5, .5, "Sweep not run" if not result.scan else "Use scan.csv for factorial cases", ha="center", transform=ax.transAxes)
            ax.set_xlabel(name)
            continue
        metric = "outlet_r95_um" if "flow" in name else "electric_bond_at_axis_50um" if name == "voltage_kv" else "gas_weber"
        rows=[r for r in rows if r.get(metric) is not None]
        xs=sorted(set(r[name] for r in rows))
        groups=[np.array([r[metric] for r in rows if r[name]==x]) for x in xs]
        ax.plot(xs,[np.median(group) for group in groups],"o-",color=BLUE)
        if groups and any(len(np.unique(group))>1 for group in groups):
            ax.fill_between(xs,[group.min() for group in groups],[group.max() for group in groups],alpha=.15,color=BLUE,label="Range across other parameters")
            ax.legend(fontsize=8)
        ax.set(xlabel=name, ylabel=metric)
        if "flow" not in name: ax.set_title("Stress diagnostic; not a stability score", fontsize=10)
    return fig


def gas_fields(result):
    fig,axes=_new("Figure 10 | Solved nitrogen / external gas field",cols=3,size=(13,4.7))
    s=result.spray
    if s is None:
        for ax in axes:ax.axis("off")
        axes[1].text(.5,.5,"Two-phase/gas solver disabled",ha="center")
        return fig
    gas=s.active&(s.phase[-1]<.1)
    _map(axes[0],s.radial_edges,s.axial_edges,s.phase[-1],"Liquid fraction","Blues",s.active)
    _map(axes[1],s.radial_edges,s.axial_edges,np.hypot(s.velocity_r,s.velocity_z),"Gas speed (m/s)",active=gas)
    _map(axes[2],s.radial_edges,s.axial_edges,s.pressure,"Gas gauge pressure (Pa)","coolwarm",gas)
    for ax in axes:
        for lip in [result.config.geometry.inner_tip,-result.config.geometry.gas_recess_mm*1e-3,0.]:ax.axvline(lip*1e3,lw=.6,ls="--",color="#888")
        ax.set_title(f"t={s.times[-1]*1e3:.4f} ms; sharp quartz lips")
    return fig


BUILDERS = [geometry, velocity, focusing, trajectories, residence, exposure, electric_field, spray, optimization,gas_fields]
FILENAMES = ["figure_01_geometry", "figure_02_velocity_pressure", "figure_03_focusing_shear",
             "figure_04_trajectories", "figure_05_residence_events", "figure_06_methanol",
             "figure_07_electric", "figure_08_two_phase", "figure_09_optimization","figure_10_gas"]


@serialized_plotting
def save_figures(result, directory, progress=lambda x: None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for name, builder in zip(FILENAMES, BUILDERS):
        progress(f"Rendering {name}")
        fig = builder(result)
        fig.savefig(directory/f"{name}.png")
        fig.savefig(directory/f"{name}.svg")
        plt.close(fig)


def trajectory_animation_figure(result):
    """Full resolved region plus the original outlet zoom, using saved data."""
    p, g = result.particles, result.config.geometry
    fig, axes = plt.subplots(1,2,figsize=(13,4.3),layout='constrained')
    fig.suptitle('Finite-sphere trajectories | time since entering resolved tip')
    zoom=max(g.liquid_start*1e3,g.inner_tip*1e3-.3)
    for ax,left,title in [(axes[0],g.liquid_start*1e3-.02,'Full resolved tip'),
                           (axes[1],zoom,'Inner lip and outlet (zoom)')]:
        ax.set(xlim=(left,.05),ylim=(0,g.c*1e6),xlabel='z (mm)',ylabel='r (um)',title=title)
        ax.fill_between([g.liquid_start*1e3,g.inner_tip*1e3],g.a*1e6,g.b*1e6,color='#e0e5e9')
        ax.axvline(g.inner_tip*1e3,ls='--',color='#999',lw=.7)
        ax.axvline(0,color=GREEN,lw=.8)
    R = result.config.cell.diameter_um*.5
    count=min(200,len(p.saved_ids));groups=[]
    for ax in axes:
        patches=[Ellipse((0,0),width=2*R/1000,height=2*R,facecolor=BLUE,alpha=.65) for _ in range(count)]
        for patch in patches:ax.add_patch(patch)
        trails=[ax.plot([],[],color=BLUE,lw=.55,alpha=.3)[0] for _ in range(count)]
        crossings=ax.plot([],[],ls='',marker='x',ms=4,color=GREEN)[0]
        overlap=ax.text(.98,.96,'',transform=ax.transAxes,ha='right',va='top',fontsize=9,color=BLUE,
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        groups.append((patches,trails,crossings,overlap))
    stamp=axes[0].text(.02,.96,'',transform=axes[0].transAxes,va='top',fontsize=9)
    axes[1].text(.02,.96,'Age-aligned responses, not simultaneous arrivals\nGreen x = recorded lip crossing',
        transform=axes[1].transAxes,va='top',fontsize=8,color=GREEN)
    frames = np.unique(np.linspace(2, len(p.trajectory_time)-1, min(70, len(p.trajectory_time)-2), dtype=int))
    def update(k):
        z=p.trajectory_z[k,:count];r=p.trajectory_r[k,:count]
        crossed=np.isfinite(z)&(z>=0);artists=[stamp]
        for ax,(patches,trails,crossings,overlap) in zip(axes,groups):
            for j,patch in enumerate(patches):
                patch.center=(z[j]*1e3,r[j]*1e6)
                patch.set_visible(bool(np.isfinite(z[j]) and z[j]<0))
                trails[j].set_data(p.trajectory_z[1:k+1,j]*1e3,p.trajectory_r[1:k+1,j]*1e6)
            crossings.set_data(z[crossed]*1e3,r[crossed]*1e6)
            bounds=ax.get_xlim();visible=np.isfinite(z)&(z<0)&(z*1e3>=bounds[0])&(z*1e3<=bounds[1])
            # Quantify exact spatial overlap without moving any trajectory.
            xy=np.c_[z[visible]*1e3,r[visible]*1e6]
            unique,multiplicity=np.unique(np.round(xy,8),axis=0,return_counts=True)
            largest=int(multiplicity.max()) if len(multiplicity) else 0
            overlap.set_text(f'{int(visible.sum())} responses / {len(unique)} locations\nLargest overlap: {largest}')
            artists.extend(patches+trails+[crossings,overlap])
        # Finished cells retain their terminal time; the median would freeze
        # after half the samples exit even while remaining cells keep moving.
        clock=float(np.nanmax(p.trajectory_time[k,:count]-p.trajectory_time[1,:count]))
        stamp.set_text(f'Resolved-tip clock: {clock:.3f} s\nRecorded crossings: {int(crossed.sum())}/{count}')
        return artists
    return fig,update,frames


@serialized_plotting
def save_trajectory_animation(result, directory):
    """Render the original finite-sphere animation from saved trajectories."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fig,update,frames=trajectory_animation_figure(result)
    if len(frames): FuncAnimation(fig, update, frames=frames).save(directory/"cell_trajectories.gif", writer=PillowWriter(fps=15))
    plt.close(fig)
    return directory/"cell_trajectories.gif" if len(frames) else None


@serialized_plotting
def save_animations(result, directory):
    directory = Path(directory)
    save_trajectory_animation(result,directory)
    if result.spray is not None:
        s = result.spray
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        image = ax.pcolormesh(s.axial_edges*1e3, s.radial_edges*1e6, s.phase[0], vmin=0, vmax=1, cmap="Blues", shading="flat")
        ax.set(xlim=(-.05, result.config.spray.length_mm), xlabel="z (mm)", ylabel="r (um)")
        fig.colorbar(image, ax=ax, label="Liquid fraction")
        def draw(k):
            image.set_array(s.phase[k]); ax.set_title(f"Computed two-phase transient | {s.times[k]*1e3:.4f} ms")
            return [image]
        frames = np.unique(np.linspace(0, len(s.times)-1, min(50, len(s.times)), dtype=int))
        FuncAnimation(fig, draw, frames=frames).save(directory/"two_phase.gif", writer=PillowWriter(fps=15))
        plt.close(fig)
