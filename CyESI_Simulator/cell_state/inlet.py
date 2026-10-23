"""Fanno transfer-tube flow and paraxial suction near the inlet axis.

Q Exactive tube 70005-20606 has a catalog bore of 0.58 mm and length of
58.42 mm. Default scenario inputs use 0.60 mm and a 320 C wall. Uniform
effective gas temperatures provide heating-sensitivity cases.
"""
import numpy as np
from scipy.optimize import brentq

GAS_R=296.8  # N2-dominated surrounding gas; editable model assumption
GAMMA=1.4


def gas_viscosity(temperature):
    return 1.76e-5*(temperature/298.15)**1.5*(298.15+111.)/(temperature+111.)


def friction_factor(reynolds):
    """Smooth Churchill Darcy friction, hydraulically smooth circular tube."""
    re=max(float(reynolds),1e-8)
    a=(2.457*np.log(1/((7/re)**.9)))**16
    b=(37530/re)**16
    return 8*((8/re)**12+1/(a+b)**1.5)**(1/12)


def choked_orifice_mass_flow(area,pressure,temperature):
    return area*pressure/np.sqrt(temperature)*np.sqrt(GAMMA/GAS_R)*((GAMMA+1)/2)**(-(GAMMA+1)/(2*(GAMMA-1)))


def fanno_tube_flow(diameter,length,pressure,temperature):
    area=np.pi*diameter**2/4
    def mass(mach):
        return area*pressure/np.sqrt(temperature)*np.sqrt(GAMMA/GAS_R)*mach*(1+(GAMMA-1)/2*mach**2)**(-(GAMMA+1)/(2*(GAMMA-1)))
    def residual(mach):
        reynolds=mass(mach)/area*diameter/gas_viscosity(temperature)
        fanno=(1-mach**2)/(GAMMA*mach**2)+(GAMMA+1)/(2*GAMMA)*np.log((GAMMA+1)*mach**2/(2+(GAMMA-1)*mach**2))
        return fanno-friction_factor(reynolds)*length/diameter
    mach=brentq(residual,1e-5,.999999)
    return float(mass(mach)),float(mach)


def estimate_inlet(parameters):
    p=parameters.validate()
    ambient=p.ambient_temperature_c+273.15;wall=p.inlet_wall_temperature_c+273.15
    density=p.ambient_pressure_pa/(GAS_R*ambient)
    cases=[]
    for temperature in [ambient,(ambient+wall)/2,wall]:
        mass,mach=fanno_tube_flow(p.inlet_diameter_mm*1e-3,p.transfer_tube_length_mm*1e-3,p.ambient_pressure_pa,temperature)
        exit_temperature=temperature/(1+(GAMMA-1)/2)
        exit_pressure=mass/(np.pi*(p.inlet_diameter_mm*1e-3)**2/4)*np.sqrt(GAS_R*exit_temperature/GAMMA)
        if p.downstream_pressure_pa>=exit_pressure:raise ValueError("Receiver pressure is too high for the assumed choked tube")
        cases.append(dict(effective_gas_temperature_k=temperature,mass_flow_kg_s=mass,
            ambient_flow_l_min=mass/density*60000,inlet_mach=mach,sonic_exit_pressure_pa=float(exit_pressure)))
    baseline=cases[1]["ambient_flow_l_min"] if p.inlet_flow_l_min is None else p.inlet_flow_l_min
    upper=choked_orifice_mass_flow(np.pi*(p.inlet_diameter_mm*1e-3)**2/4,p.ambient_pressure_pa,ambient)/density*60000
    if baseline*max(p.flow_factors)>upper:raise ValueError("Scenario suction exceeds ideal cold-orifice choking ceiling")
    return {"instrument":p.instrument,"inlet_diameter_mm":p.inlet_diameter_mm,
        "transfer_tube_length_mm":p.transfer_tube_length_mm,"wall_temperature_c":p.inlet_wall_temperature_c,
        "baseline_ambient_flow_l_min":baseline,"temperature_sensitivity_cases":cases,
        "ideal_cold_orifice_ceiling_l_min":float(upper),
        "flow_input":"engineering estimate" if p.inlet_flow_l_min is None else "user override",
        "flow_reference":"actual volume at ambient temperature / pressure; not standard L/min",
        "sources":["https://tools.thermofisher.com/content/sfs/manuals/IonTransferTube.pdf",
                   "https://www.grc.nasa.gov/www/k-12/airplane/mflchk.html"],
        "limitations":["Fanno assumes constant effective total temperature; heated tube is not solved thermally",
            "entrance losses, surface roughness and actual cone geometry are unknown",
            "200 Pa downstream pressure is an approximate forevacuum condition, not an Orbitrap UHV pressure"]}


def suction_velocity(radial,z,gap,aperture_radius,flow_m3_s):
    """Uniform suction disk, exact on axis and paraxial radial approximation.

    u_z=Q/A*(1-s/sqrt(s^2+a^2)); u_r follows incompressible continuity to
    first order in r/a. Use for focused cells close to the inlet axis.
    """
    s=max(gap-z,0.);a=aperture_radius;u0=flow_m3_s/(np.pi*a*a)
    return -.5*radial*u0*a*a/(s*s+a*a)**1.5,u0*(1-s/np.sqrt(s*s+a*a))


def gap_gas_temperature(z,gap,aperture_radius,flow_m3_s,parameters):
    # High-Pe 1D advection/conduction hot-end approximation. The hot wall is
    # NOT assigned to the full 2 mm gap. Radial cone convection remains absent.
    p=parameters;s=max(gap-z,0.);a=aperture_radius
    ambient=p.ambient_temperature_c+273.15
    density=p.ambient_pressure_pa/(GAS_R*ambient);cp=GAMMA*GAS_R/(GAMMA-1)
    alpha=.026/(density*cp)
    integral=flow_m3_s/(np.pi*a*a)*(s+a-np.sqrt(s*s+a*a))
    return ambient+(p.inlet_wall_temperature_c-p.ambient_temperature_c)*np.exp(-min(700.,integral/alpha))
