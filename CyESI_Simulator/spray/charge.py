"""Conservative axisymmetric free-charge transport and a remote RC supply.

Implicit Ohmic relaxation solves (G_epsilon + dt G_sigma)V = q_adv Vol + b.
The unknown liquid inlet voltage is closed against conduction + displacement
current through the upstream sheath resistance. Quartz is a dielectric with
zero Ohmic conductivity. This is a leaky-dielectric research closure, not PNP.
"""
import numpy as np
from scipy.sparse.linalg import splu
from ..numerics import metric,elliptic
from ..electro.electrostatics import EPS0

CHARGE_MODEL_VERSION=2


def advect_charge(q,u,w,active,re,ze,dt):
    r,z,aa,ra,vol=metric(re,ze);fr=np.zeros_like(u);fz=np.zeros_like(w)
    fr[1:-1]=ra[1:-1]*u[1:-1]*np.where(u[1:-1]>=0,q[:-1],q[1:])*(active[:-1]&active[1:])
    fz[:,1:-1]=aa[:,None]*w[:,1:-1]*np.where(w[:,1:-1]>=0,q[:,:-1],q[:,1:])*(active[:,:-1]&active[:,1:])
    # Incoming feed and external gas carry zero net free charge; charge enters
    # the liquid through the separately conserved Ohmic electrode circuit.
    fr[-1]=ra[-1]*np.maximum(u[-1],0)*q[-1]*active[-1]
    fz[:,0]=aa*np.minimum(w[:,0],0)*q[:,0]*active[:,0]
    fz[:,-1]=aa*np.maximum(w[:,-1],0)*q[:,-1]*active[:,-1]
    delta=dt*(np.diff(fr,axis=0)+np.diff(fz,axis=1))
    new=q-delta/vol
    boundary=float(dt*(fr[-1].sum()+fz[:,-1].sum()-fz[:,0].sum()))
    return new,boundary


class AxisymmetricCharge:
    def __init__(self,config,electric,re,ze,active,phi):
        self.config=config;self.re=re;self.ze=ze;self.active=active
        self.r,self.z,self.aa,self.ra,self.vol=metric(re,ze)
        r,z=self.r,self.z
        self.boundary={'r1':electric.sample(np.full_like(z,re[-1]),z,'potential'),
            'z1':electric.sample(r,np.full_like(r,ze[-1]),'potential'),
            'z0':electric.sample(r,np.full_like(r,ze[0]),'potential').copy()}
        self.inlet=(phi[:,0]>.01)&active[:,0]
        self.ids=np.arange(phi.size).reshape(phi.shape)[:,0][self.inlet]
        self.voltage=float(config.electrical.voltage_kv*1000)
        distance=max(0.,config.electrical.contact_distance_mm*.001+ze[0])
        conduct=config.electrical.sheath_conductivity_s_m
        self.resistance=distance/max(conduct*np.pi*(config.geometry.c**2-config.geometry.b**2),1e-30)
        self.q=np.zeros_like(phi)
        eps=self.epsilon(phi);A,b=elliptic(re,ze,eps,boundary=self.boundary)
        V=splu(A.tocsc()).solve(b).reshape(phi.shape)
        self.initial_potential=V
        self.initial_er,self.initial_ez=[-a for a in np.gradient(V,self.r,self.z)]
        inlet_v=self.boundary['z0'][self.inlet]
        ge=eps[:,0][self.inlet]*self.aa[self.inlet]/(self.z[0]-ze[0])
        self.previous_d=float(np.sum(ge*(inlet_v-V[:,0][self.inlet])))
        self.initial_charge_c=0.;self.max_balance=0.;self.max_circuit=0.;self.advected_out=0.;self.ohmic_out=0.

    def epsilon(self,phi):
        return EPS0*np.where(self.active,1+(self.config.chemistry.liquid_relative_permittivity-1)*phi,
                            self.config.electrical.quartz_relative_permittivity)

    def predict(self,phi,u,w,dt,salt_mol_m3=None):
        eps=self.epsilon(phi)
        # Effective support K is the same editable unmeasured assumption as
        # the 3-D circuit; no assertion of a measured pure-methanol value.
        if salt_mol_m3 is None:sigma=self.config.electrical.sheath_conductivity_s_m*phi**2
        else:
            salt_conductivity=96485.33212**2*(1.96e-9+1.45e-9)*np.maximum(salt_mol_m3,0)/(8.314462618*self.config.fluid.temperature_k)
            sigma=phi*(self.config.electrical.sheath_conductivity_s_m+salt_conductivity)
        sigma=np.where(self.active,sigma,0.)
        bc={k:v.copy() for k,v in self.boundary.items()};bc['z0'][self.inlet]=0.
        Ae,be=elliptic(self.re,self.ze,eps,boundary=bc)
        As,bs=elliptic(self.re,self.ze,sigma,boundary=bc)
        qa,out=advect_charge(self.q,u,w,self.active,self.re,self.ze,dt)
        factor=splu((Ae+dt*As).tocsc());base=factor.solve(qa.ravel()*self.vol.ravel()+be+dt*bs)
        area=self.aa[self.inlet];distance=self.z[0]-self.ze[0]
        ge=eps[:,0][self.inlet]*area/distance;gs=sigma[:,0][self.inlet]*area/distance
        source=np.zeros(phi.size);source[self.ids]=ge+dt*gs
        response=factor.solve(source);weights=ge/dt+gs;R=self.resistance
        denominator=1+R*(weights.sum()-weights@response[self.ids])
        node=(self.voltage+R*(self.previous_d/dt+weights@base[self.ids]))/denominator
        V=base+response*node
        be=be.copy();bs=bs.copy();be[self.ids]+=ge*node;bs[self.ids]+=gs*node
        q=((Ae@V-be)/self.vol.ravel()).reshape(phi.shape)
        ohmic=As@V-bs
        error=(q-qa)*self.vol+dt*ohmic.reshape(phi.shape)
        D=float(np.sum(ge*(node-V[self.ids])));J=float(np.sum(gs*(node-V[self.ids])))
        total=J+(D-self.previous_d)/dt
        supply=(self.voltage-node)/R if R>0 else total
        V=V.reshape(phi.shape);er,ez=[-a for a in np.gradient(V,self.r,self.z)]
        e2=er*er+ez*ez
        trr=eps*(er*er-.5*e2);tzz=eps*(ez*ez-.5*e2);trz=eps*er*ez;tt=-.5*eps*e2
        force_r=np.gradient(trr,self.r,axis=0)+np.gradient(trz,self.z,axis=1)+(trr-tt)/self.r[:,None]
        force_z=np.gradient(trz,self.r,axis=0)+np.gradient(tzz,self.z,axis=1)+trz/self.r[:,None]
        force_r[~self.active]=0.;force_z[~self.active]=0.
        return dict(q=q,potential=V,er=er,ez=ez,force_r=force_r,force_z=force_z,
            displacement=D,node_voltage_v=float(node),supply_current_a=float(supply),
            circuit_error_a=float(total-supply),charge_balance_error_c=float(np.max(np.abs(error))),
            advected_out_c=out,ohmic_out_c=float(dt*np.sum(ohmic)))

    def commit(self,state):
        self.q=state['q'];self.previous_d=state['displacement']
        self.max_balance=max(self.max_balance,state['charge_balance_error_c'])
        self.max_circuit=max(self.max_circuit,abs(state['circuit_error_a']))
        self.advected_out+=state['advected_out_c'];self.ohmic_out+=state['ohmic_out_c']

    def diagnostics(self):
        return dict(method='axisymmetric conservative free charge + implicit Ohmic relaxation + remote R circuit with displacement current',
            model_version=CHARGE_MODEL_VERSION,
            max_local_charge_balance_error_c=self.max_balance,max_circuit_balance_error_a=self.max_circuit,
            global_charge_balance_error_c=float(np.sum(self.q*self.vol)-self.initial_charge_c+self.advected_out+self.ohmic_out),
            upstream_sheath_resistance_ohm=self.resistance,support_conductivity_s_m=self.config.electrical.sheath_conductivity_s_m,
            conductivity_scope='editable unmeasured support conductivity plus optional Nernst-Einstein advected salt; activity and temperature feedback remain approximate')
