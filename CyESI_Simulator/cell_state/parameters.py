from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math


@dataclass
class StateParameters:
    instrument: str = "Q Exactive"
    inlet_diameter_mm: float = .6  # Scenario bore; catalog value is 0.58 mm.
    transfer_tube_length_mm: float = 58.42  # Thermo part 70005-20606
    inlet_wall_temperature_c: float = 320.0  # Wall temperature.
    ambient_temperature_c: float = 25.0
    ambient_pressure_pa: float = 101325.0
    downstream_pressure_pa: float = 200.0  # approximate forevacuum, not Orbitrap UHV
    inlet_flow_l_min: float | None = None  # derive a compressible-tube estimate
    flow_factors: list[float] = field(default_factory=lambda: [.5, 1., 1.5])
    drop_radii_um: list[float] = field(default_factory=lambda: [10., 20., 40.])
    charge_fractions: list[float] = field(default_factory=lambda: [.05, .15, .3])
    methanol_permeabilities_m_s: list[float] = field(default_factory=lambda: [1e-7, 1e-6, 1e-5])
    dissolution_rates_s: list[float] = field(default_factory=lambda: [0., 20., 200.])
    solute_permeability_m_s: float = 1e-8  # hypothetical generic neutral tracer
    solvent_permeability_beta: float = 0.0  # optional calibrated C-dependence
    membrane_thickness_nm: float = 5.0
    minimum_membrane_fraction: float = .4  # continuum-shell validity, not a viability threshold
    maximum_pore_area_fraction: float = .05  # validity guard; not biological survival
    membrane_capacitance_f_m2: float = .01
    membrane_conductance_s_m2: float = 0.0
    cytosol_conductivity_s_m: float = .5
    intracellular_diffusivity_m2_s: float = 1e-9
    tracer_diffusivity_m2_s: float = 1e-10
    shells: int = 12
    angles: int = 8
    sample_tracks: int = 3
    max_pre_step_ms: float = 2.0
    max_flight_step_us: float = 5.0
    max_flight_time_ms: float = 30.0
    relative_humidity: float = .4
    hot_face_radius_mm: float = 5.0
    effective_emissivity: float = .3
    alignment_offset_um: float = 0.0
    initial_angle_deg: float = 0.0
    evaporation: bool = True
    electrical_screening: bool = True
    use_saved_nozzle_gas: bool = True  # frozen additive perturbation, not coupled NS
    pore_creation_rate_m2_s: float = 1e9
    resting_pore_density_m2: float = 1.5e9
    pore_characteristic_v: float = .258
    pore_exponent: float = 2.46
    pore_radius_nm: float = .8

    def validate(self):
        positive=[self.inlet_diameter_mm,self.transfer_tube_length_mm,self.ambient_pressure_pa,
            self.downstream_pressure_pa,self.membrane_thickness_nm,self.membrane_capacitance_f_m2,
            self.cytosol_conductivity_s_m,self.intracellular_diffusivity_m2_s,self.tracer_diffusivity_m2_s,
            self.max_pre_step_ms,self.max_flight_step_us,self.max_flight_time_ms,self.hot_face_radius_mm,
            self.pore_creation_rate_m2_s,self.resting_pore_density_m2,self.pore_characteristic_v,self.pore_radius_nm]
        for key,value in vars(self).items():
            values=value if isinstance(value,list) else [value]
            if any(isinstance(x,(int,float)) and not math.isfinite(x) for x in values):raise ValueError(f"Nonfinite state parameter: {key}")
        if min(positive)<=0:raise ValueError("Lengths, transport coefficients and integration steps must be positive")
        if self.inlet_flow_l_min is not None and self.inlet_flow_l_min<=0:raise ValueError("Inlet flow must be positive or null")
        if not 0<self.minimum_membrane_fraction<1 or not 0<self.maximum_pore_area_fraction<1 or not 0<=self.relative_humidity<=1 or not 0<=self.effective_emissivity<=1:
            raise ValueError("Invalid membrane fraction / humidity / emissivity")
        if not self.ambient_temperature_c>-17 or not self.inlet_wall_temperature_c>=self.ambient_temperature_c:
            raise ValueError("Use an ambient temperature above 256 K and a warmer inlet")
        if not 0<self.downstream_pressure_pa/self.ambient_pressure_pa<.1:raise ValueError("This approximate tube model requires a low-pressure receiver")
        for values in [self.flow_factors,self.drop_radii_um,self.methanol_permeabilities_m_s]:
            if not values or min(values)<=0:raise ValueError("Empty or nonpositive scenario range")
        if not self.charge_fractions or min(self.charge_fractions)<0 or max(self.charge_fractions)>=1:raise ValueError("Charge fractions must be in [0,1)")
        if not self.dissolution_rates_s or min(self.dissolution_rates_s)<0:raise ValueError("Invalid dissolution rate")
        if min(self.solute_permeability_m_s,self.solvent_permeability_beta,self.membrane_conductance_s_m2)<0:
            raise ValueError("Permeabilities and conductance must be nonnegative")
        if self.pore_exponent<=1:raise ValueError("Pore exponent must exceed one")
        for count,low in [(self.shells,3),(self.angles,2),(self.sample_tracks,1)]:
            if not isinstance(count,int) or isinstance(count,bool) or count<low:raise ValueError("Invalid state model resolution")
        if self.hot_face_radius_mm<=self.inlet_diameter_mm/2:raise ValueError("Hot face must surround the aperture")
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig"))).validate()

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
