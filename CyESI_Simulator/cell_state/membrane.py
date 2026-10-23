"""Conservative spherical intracellular diffusion and thin lipid-shell RC.

Methanol fraction uses additive-volume, equal-volume water countertransport;
this is a reduced incompressible mixture model. A generic neutral tracer is
mass conservative. Lipid dissolution kinetics are explicit uncalibrated inputs.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from scipy.linalg import solve_banded


class SphereTransport:
    def __init__(self,radius,shells):
        self.edges=np.linspace(0,radius,shells+1)
        self.centers=(self.edges[:-1]+self.edges[1:])/2
        self.volumes=4*np.pi/3*np.diff(self.edges**3)
        self.areas=4*np.pi*self.edges**2
        self.radius=radius

    def advance(self,concentration,diffusivity,permeability,dt,bath_concentration,bath_volume=None):
        """Implicit FV with Robin membrane; optional finite external compartment.

        Returns new cell concentration, external concentration, inward amount.
        Amount units = concentration units times m^3. Both sides use one flux.
        """
        c=np.asarray(concentration,float);n=len(c)
        if dt<=0:return c.copy(),float(bath_concentration),0.
        finite=bath_volume is not None
        if finite and bath_volume<=0:raise ValueError("Carrier liquid volume must be positive")
        size=n+int(finite);vol=np.r_[self.volumes,bath_volume] if finite else self.volumes
        diagonal=vol/dt;lower=np.zeros(size-1);upper=np.zeros(size-1)
        rhs=np.r_[c,bath_concentration]*vol/dt if finite else c*vol/dt
        k=diffusivity*self.areas[1:-1]/np.diff(self.centers)
        diagonal[:-1 if not finite else -2]+=k
        diagonal[1:n]+=k
        lower[:n-1]=-k;upper[:n-1]=-k
        km=0. if permeability==0 else self.areas[-1]/(1/permeability+(self.radius-self.centers[-1])/diffusivity)
        diagonal[n-1]+=km
        if finite:
            diagonal[-1]+=km;lower[-1]=-km;upper[-1]=-km
        else:rhs[-1]+=km*bath_concentration
        band=np.zeros((3,size));band[1]=diagonal;band[0,1:]=upper;band[2,:-1]=lower
        result=solve_banded((1,1),band,rhs,check_finite=False)
        amount=float(np.dot(result[:n]-c,self.volumes))
        return result[:n],float(result[-1] if finite else bath_concentration),amount


@dataclass
class MembraneState:
    methanol: np.ndarray
    tracer: np.ndarray
    voltage: np.ndarray
    pore_density: np.ndarray
    membrane_fraction: float = 1.
    lipid_released_fraction: float = 0.
    max_voltage_v: float = 0.
    max_pore_area_fraction: float = 0.

    def clone(self):return deepcopy(self)


class LipidShell:
    def __init__(self,radius,parameters,methanol_permeability,dissolution_rate):
        self.radius=radius;self.p=parameters
        self.transport=SphereTransport(radius,parameters.shells)
        self.cosine,self.weights=np.polynomial.legendre.leggauss(parameters.angles)
        self.weights=self.weights/2
        self.permeability=methanol_permeability;self.dissolution_rate=dissolution_rate

    def initial(self):
        p=self.p
        return MembraneState(np.zeros(p.shells),np.ones(p.shells),np.zeros(p.angles),
                             np.full(p.angles,p.resting_pore_density_m2))

    def electrical_step(self,state,field_v_m,conductivity,dt):
        p=self.p;fraction=state.membrane_fraction
        thickness=p.membrane_thickness_nm*1e-9*max(fraction,1e-3)
        capacitance=p.membrane_capacitance_f_m2/max(fraction,1e-3)
        pore_area=np.clip(state.pore_density*np.pi*(p.pore_radius_nm*1e-9)**2,0,1)
        conductance=p.membrane_conductance_s_m2+conductivity/thickness*pore_area
        tau=self.radius*capacitance*(1/p.cytosol_conductivity_s_m+1/(2*max(conductivity,1e-12)))
        decay_rate=1/tau+conductance/capacitance
        target=1.5*self.radius*field_v_m*self.cosine/(tau*decay_rate)
        state.voltage=target+(state.voltage-target)*np.exp(-decay_rate*dt)
        x=(state.voltage/p.pore_characteristic_v)**2
        # Exact frozen-voltage pore-density step, evaluated in log form. Bound
        # at area packing, not at an arbitrarily chosen voltage.
        max_density=1/(np.pi*(p.pore_radius_nm*1e-9)**2)
        rate=p.pore_creation_rate_m2_s/p.resting_pore_density_m2*np.exp(np.maximum(-745.,(1-p.pore_exponent)*x))
        coefficient=np.full_like(rate,dt)
        np.divide(-np.expm1(-rate*dt),rate,out=coefficient,where=rate>1e-200)
        log_increment=np.log(p.pore_creation_rate_m2_s)+x+np.log(coefficient)
        state.pore_density=np.minimum(max_density,state.pore_density*np.exp(-rate*dt)+
            np.exp(np.minimum(np.log(max_density),log_increment)))
        state.max_voltage_v=max(state.max_voltage_v,float(np.max(abs(state.voltage))))
        average=float(np.dot(np.minimum(state.pore_density*np.pi*(p.pore_radius_nm*1e-9)**2,1),self.weights))
        state.max_pore_area_fraction=max(state.max_pore_area_fraction,average)
        return average

    def step(self,state,dt,external_methanol,field_v_m,conductivity,external_tracer=0.,external_volume=None):
        p=self.p
        pore_area=self.electrical_step(state,field_v_m,conductivity,dt)
        old=state.membrane_fraction
        state.membrane_fraction*=np.exp(-self.dissolution_rate*external_methanol**2*dt)
        state.lipid_released_fraction+=old-state.membrane_fraction
        h=p.membrane_thickness_nm*1e-9*max(state.membrane_fraction,1e-3)
        modifier=np.exp(min(50.,p.solvent_permeability_beta*external_methanol))/max(state.membrane_fraction,1e-3)
        pm=self.permeability*modifier+p.intracellular_diffusivity_m2_s/h*pore_area
        ps=p.solute_permeability_m_s*modifier+p.tracer_diffusivity_m2_s/h*pore_area
        state.methanol,cout,amount=self.transport.advance(state.methanol,p.intracellular_diffusivity_m2_s,pm,dt,external_methanol,external_volume)
        state.tracer,sout,tracer_amount=self.transport.advance(state.tracer,p.tracer_diffusivity_m2_s,ps,dt,external_tracer,external_volume)
        return cout,sout,amount,tracer_amount

    def metrics(self,state):
        volumes=self.transport.volumes
        return {"intracellular_methanol_fraction":float(np.dot(state.methanol,volumes)/volumes.sum()),
            "intracellular_tracer_fraction":float(np.dot(state.tracer,volumes)/volumes.sum()),
            "membrane_lipid_remaining_fraction":state.membrane_fraction,
            "dissolved_lipid_fraction":state.lipid_released_fraction,
            "max_transmembrane_potential_v":state.max_voltage_v,
            "max_pore_area_fraction":state.max_pore_area_fraction,
            "shell_continuum_valid":bool(state.membrane_fraction>=self.p.minimum_membrane_fraction and state.max_pore_area_fraction<=self.p.maximum_pore_area_fraction)}
