"""Configuration fields and their display units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass
class Geometry:
    inner_id_um: float = 50.0
    inner_od_um: float = 150.0
    sheath_id_um: float = 200.0
    sheath_od_um: float = 360.0
    gas_id_um: float = 540.0
    gas_od_um: float = 640.0  # Assumed: user did not specify gas tube OD.
    sample_length_mm: float = 700.0
    tip_resolved_length_mm: float = 5.0
    inner_recess_mm: float = 1.0
    gas_recess_mm: float = 1.0
    gas_supply_length_mm: float = 5.0  # Pressure-to-flow conversion needs this.
    ms_gap_mm: float = 2.0
    external_radius_mm: float = 1.5

    @property
    def a(self): return self.inner_id_um * 0.5e-6
    @property
    def b(self): return self.inner_od_um * 0.5e-6
    @property
    def c(self): return self.sheath_id_um * 0.5e-6
    @property
    def d(self): return self.sheath_od_um * 0.5e-6
    @property
    def inner_tip(self): return -self.inner_recess_mm * 1e-3
    @property
    def liquid_start(self):
        return self.inner_tip - min(self.tip_resolved_length_mm, self.sample_length_mm) * 1e-3
    @property
    def bulk_length(self):
        return max(0., self.sample_length_mm - self.tip_resolved_length_mm) * 1e-3


@dataclass
class Fluid:
    sample_flow_ul_min: float = 1.0
    sheath_flow_ul_min: float = 20.0
    gas_pressure_kpa: float = 0.4
    sample_density: float = 1000.0
    sheath_density: float = 792.0
    sample_viscosity: float = 1.0e-3
    sheath_viscosity: float = 0.544e-3
    gas_density: float = 1.165
    gas_viscosity: float = 1.76e-5
    surface_tension: float = 0.0226
    temperature_k: float = 298.15
    inertial: bool = True

    @property
    def qs(self): return self.sample_flow_ul_min * 1e-9 / 60
    @property
    def qh(self): return self.sheath_flow_ul_min * 1e-9 / 60


@dataclass
class Cell:
    diameter_um: float = 8.0
    density: float = 1050.0
    concentration_per_ml: float = 1e6
    count: int = 10000
    zeta_mv: float = -20.0
    clausius_mossotti: float = -0.5  # Assumed DC effective value, not a membrane model.
    brownian_below_um: float = 1.0
    brownian_enabled: bool | None = None  # None uses the default size cutoff.
    gravity_z: float = 0.0  # Orientation unspecified; sedimentation disabled.
    focusing_radius_um: float = 25.0

    @property
    def use_brownian(self):
        return self.diameter_um <= self.brownian_below_um if self.brownian_enabled is None else self.brownian_enabled


@dataclass
class Chemistry:
    sample_medium: str = "140 mM ammonium formate"
    ammonium_formate_mm: float = 140.0
    sheath_methanol_fraction: float = 1.0
    diffusivity_m2_s: float = 1.5e-9  # Constant illustrative mutual diffusivity.
    liquid_relative_permittivity: float = 60.0  # Effective value; editable.


@dataclass
class Electrical:
    voltage_kv: float = 4.5
    liquid_model: str = "resistive"  # conducting = ideal zero-resistance limit
    quartz_relative_permittivity: float = 3.8
    contact_distance_mm: float = 100.0
    contact_mode: str = "liquid_contact"  # User confirmed metal contact with sheath at -100 mm.
    spray_current_na: float | None = None  # None: open-circuit limiting case
    sheath_conductivity_s_m: float = 1e-5  # unmeasured supporting conductivity
    quartz_zeta_mv: float = -20.0  # independent of cell zeta; unmeasured
    electroosmosis: bool = True


@dataclass
class Numerics:
    workers: int = 0  # 0 = automatic with CPU/memory headroom; 1 = serial
    particle_backend: str = "auto"  # auto / numpy reference / numba compiled
    radial_cells: int = 48
    upstream_cells: int = 48
    mixing_cells: int = 80
    picard_iterations: int = 25
    flow_tolerance: float = 2e-6
    coupling_iterations: int = 3
    coupling_tolerance: float = 0.03
    particle_dt_s: float = 1.25e-4
    bulk_dt_s: float = 0.02
    bulk_timeout_s: float = 1800.0
    particle_tip_timeout_s: float = 5.0
    saved_trajectories: int = 100
    seed: int = 42


@dataclass
class Spray:
    enabled: bool = True
    free_charge_enabled: bool = True
    radial_cells: int = 32
    axial_cells: int = 72
    radius_mm: float = 0.4
    length_mm: float = 0.6
    duration_ms: float = 0.10
    max_steps: int = 6000
    save_every: int = 20
    electric_update_every: int = 10
    interface_cells: float = 1.0
    upstream_length_mm: float = 2.0
    extend_to_ms: bool = True


@dataclass
class Events:
    durations_s: list[float] = field(default_factory=lambda: [60., 600.])
    collection_efficiency: float | None = None
    resolving_time_ms: float = 5.0
    startup: str = "steady"  # steady feed or empty-tube startup


@dataclass
class Supply:
    # Flow control is the default; pressure control solves the supply network.
    sample_control: str = "flow"
    sample_pressure_mbar: float = 400.0  # reservoir gauge pressure, atmosphere reference
    outlet_pressure_pa: float = 0.0  # imposed external liquid traction offset
    reservoir_head_mm: float = 0.0  # reservoir liquid surface above tube entrance
    orientation: str = "legacy"
    # angle is measured downwards from horizontal; roll sets transverse gravity.
    segments: list[dict] = field(default_factory=lambda: [
        {"fraction": .5, "angle_deg": 0., "roll_deg": 0.},
        {"fraction": .5, "angle_deg": 90., "roll_deg": 0.}])
    wall_capture_gap_um: float = 0.10  # contact/roughness scenario, not measured adhesion
    wall_capture_rate_s: float = 0.0  # 0 = nonadhesive; conditional stochastic capture
    gravity_enabled: bool = True
    saffman_enabled: bool = True
    lubrication_enabled: bool = True
    shear_damage_coefficient: float | None = None  # hazard = k*(tau / 1 Pa)^exponent
    shear_damage_exponent: float = 2.0
    pressure_tolerance_pa: float = 2.0
    pressure_iterations: int = 6


@dataclass
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    fluid: Fluid = field(default_factory=Fluid)
    cell: Cell = field(default_factory=Cell)
    chemistry: Chemistry = field(default_factory=Chemistry)
    electrical: Electrical = field(default_factory=Electrical)
    numerics: Numerics = field(default_factory=Numerics)
    spray: Spray = field(default_factory=Spray)
    events: Events = field(default_factory=Events)
    supply: Supply = field(default_factory=Supply)

    def validate(self):
        for section in fields(self):
            for key, value in vars(getattr(self, section.name)).items():
                values = value if isinstance(value, list) else [value]
                if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in values):
                    raise ValueError(f"{section.name}.{key} must be finite")
        g, f, c, n, s = self.geometry, self.fluid, self.cell, self.numerics, self.spray
        if not isinstance(s.free_charge_enabled,bool):raise ValueError('Spray free-charge switch must be boolean')
        if c.brownian_enabled is not None and not isinstance(c.brownian_enabled,bool):
            raise ValueError('cell.brownian_enabled must be true, false or null')
        if isinstance(n.workers,bool) or not isinstance(n.workers,int) or not 0<=n.workers<=32:
            raise ValueError("workers must be an integer in [0,32]; 0 means automatic")
        if n.particle_backend not in ("auto","numpy","numba"):
            raise ValueError("particle_backend must be auto, numpy or numba")
        if not 0 < g.a < g.b < g.c < g.d < g.gas_id_um * 0.5e-6 < g.gas_od_um * 0.5e-6:
            raise ValueError("Need inner ID < inner OD < sheath ID < sheath OD < gas ID < gas OD")
        positive = [g.sample_length_mm, g.tip_resolved_length_mm, g.gas_supply_length_mm,
                    g.ms_gap_mm, g.external_radius_mm, f.qs, f.qh, f.sample_density,
                    f.sheath_density, f.sample_viscosity, f.sheath_viscosity,
                    f.gas_density, f.gas_viscosity, f.surface_tension, f.temperature_k,
                    c.diameter_um, c.density, self.chemistry.diffusivity_m2_s,
                    self.chemistry.liquid_relative_permittivity,
                    self.electrical.quartz_relative_permittivity, n.particle_dt_s,
                    n.particle_tip_timeout_s, n.bulk_dt_s, n.bulk_timeout_s, n.flow_tolerance, n.coupling_tolerance,
                    s.radius_mm, s.length_mm, s.duration_ms, s.interface_cells]
        if min(positive) <= 0:
            raise ValueError("Lengths, flows, material properties, time steps must be positive")
        if min(g.inner_recess_mm, g.gas_recess_mm, f.gas_pressure_kpa, c.concentration_per_ml,
               c.focusing_radius_um, c.brownian_below_um) < 0:
            raise ValueError("Recesses, gas pressure, concentration and radii cannot be negative")
        if c.diameter_um >= g.inner_id_um:
            raise ValueError("Cell must fit inside the sample capillary")
        if not -0.5 <= c.clausius_mossotti <= 1:
            raise ValueError("Clausius-Mossotti factor must be in [-0.5, 1]")
        if not 0 <= self.chemistry.sheath_methanol_fraction <= 1:
            raise ValueError("Methanol fraction must be in [0,1]")
        e = self.electrical
        if e.liquid_model not in ("conducting", "dielectric", "resistive"):
            raise ValueError("liquid_model must be conducting, resistive or dielectric")
        if e.contact_distance_mm <= 0 or e.sheath_conductivity_s_m <= 0:
            raise ValueError("Contact distance and supporting conductivity must be positive")
        if e.contact_mode not in ("liquid_contact", "external_quartz"):
            raise ValueError("Unknown electrical contact mode")
        if e.contact_mode=="external_quartz":
            raise ValueError("An electrode outside insulating quartz needs a capacitive/time-dependent electrode model; direct-liquid DC contact cannot be used")
        if e.liquid_model=="resistive" and e.contact_distance_mm < -g.liquid_start*1e3:
            raise ValueError("Remote contact must lie upstream of the resolved liquid domain; shorten tip_resolved_length_mm for a nearer contact")
        if e.spray_current_na is not None and e.spray_current_na < 0:
            raise ValueError("Spray current is a nonnegative magnitude")
        if self.events.startup not in ("steady", "empty"):
            raise ValueError("Event startup must be steady or empty")
        if s.upstream_length_mm <= 0:
            raise ValueError("Spray upstream length must be positive")
        supply = self.supply
        if supply.sample_control not in ("flow", "pressure"):
            raise ValueError("Sample supply must be flow or pressure controlled")
        if supply.orientation not in ("legacy", "horizontal", "down", "up", "horizontal_down", "down_horizontal", "custom"):
            raise ValueError("Unknown tube orientation")
        if supply.sample_pressure_mbar < 0 or supply.wall_capture_rate_s < 0 or supply.wall_capture_gap_um <= 0:
            raise ValueError("Pressure / wall capture inputs are invalid")
        if supply.wall_capture_gap_um >= (g.inner_id_um-c.diameter_um)/2:
            raise ValueError("Wall exclusion gap must leave a positive finite-cell inlet clearance")
        if supply.shear_damage_coefficient is not None and (not math.isfinite(supply.shear_damage_coefficient) or supply.shear_damage_coefficient < 0):
            raise ValueError("Shear damage coefficient must be null or nonnegative")
        if supply.shear_damage_exponent <= 0 or supply.pressure_tolerance_pa <= 0:
            raise ValueError("Pressure tolerance and damage exponent must be positive")
        if supply.orientation == "custom":
            if not supply.segments:
                raise ValueError("Custom orientation needs segments")
            for segment in supply.segments:
                if set(segment) - {"fraction", "angle_deg", "roll_deg"} or "fraction" not in segment or "angle_deg" not in segment:
                    raise ValueError("Segments use fraction, angle_deg and optional roll_deg")
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in segment.values()) or segment["fraction"] <= 0 or not -90 <= segment["angle_deg"] <= 90:
                    raise ValueError("Segment fractions must be positive and angles within [-90,90]")
            if not math.isclose(sum(x["fraction"] for x in supply.segments), 1., abs_tol=1e-8):
                raise ValueError("Segment fractions must sum to one")
        if self.events.collection_efficiency is not None and not 0 <= self.events.collection_efficiency <= 1:
            raise ValueError("Collection efficiency must be null or in [0,1]")
        if not self.events.durations_s or min(self.events.durations_s) <= 0 or self.events.resolving_time_ms < 0:
            raise ValueError("Invalid event horizon / resolving time")
        integers = [(c.count, 1), (n.radial_cells, 9), (n.upstream_cells, 4),
                    (n.mixing_cells, 4), (n.picard_iterations, 1), (n.coupling_iterations, 1),
                    (n.saved_trajectories, 1), (n.seed, 0), (s.radial_cells, 12),
                    (s.axial_cells, 16), (s.max_steps, 1), (s.save_every, 1),
                    (s.electric_update_every, 1)]
        integers.append((supply.pressure_iterations, 2))
        if any(isinstance(v, bool) or not isinstance(v, int) or v < low for v, low in integers):
            raise ValueError("Invalid integer count / resolution / seed")
        if g.external_radius_mm * 1e-3 <= g.gas_od_um * 0.5e-6:
            raise ValueError("External radius must exceed gas tube radius")
        if s.radius_mm * 1e-3 <= g.gas_id_um * 0.5e-6:
            raise ValueError("Spray radius must include the gas annulus")
        return self

    def to_dict(self): return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data):
        result = cls()
        for section, values in data.items():
            if section not in {f.name for f in fields(cls)}:
                raise ValueError(f"Unknown config section: {section}")
            target = getattr(result, section)
            for name, value in values.items():
                if name not in vars(target): raise ValueError(f"Unknown parameter: {section}.{name}")
                setattr(target, name, value)
        return result.validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
