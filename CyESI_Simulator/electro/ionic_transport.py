"""Electroneutral binary-electrolyte closure, not a resolved Debye-layer PNP.

NH4+ and HCOO- have equal concentrations. Their unequal diffusion rates
produce the ambipolar diffusion potential. Ohmic current is solved in 2D;
a 1D annular resistor connects the resolved domain to the remote HV contact.
Methanol has charge number zero and receives no electrophoretic drift.
Constant aqueous ion diffusivities and supporting conductivity need calibration.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ..numerics import elliptic, metric, dirichlet_solve
from ..chemistry.solvent_transport import filled

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
D_PLUS, D_MINUS = 1.96e-9, 1.45e-9


@dataclass
class IonicResult:
    radial_edges: np.ndarray
    axial_edges: np.ndarray
    salt_mol_m3: np.ndarray
    potential: np.ndarray
    field_r: np.ndarray
    field_z: np.ndarray
    current_r_a: np.ndarray
    current_z_a: np.ndarray
    conductivity: np.ndarray
    active: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        r,z,*_=metric(self.radial_edges,self.axial_edges)
        self.r,self.z=r,z;self._cache={}
        # Take derivatives of the ionic potential on the liquid grid.
        e2=filled(self.field_r**2+self.field_z**2,self.active)
        self.grad_e2_r,self.grad_e2_z=np.gradient(e2,r,z,edge_order=2)

    def sample(self, r, z, name="potential"):
        rc, zc = self.r,self.z
        if name not in self._cache:
            self._cache[name] = RegularGridInterpolator((rc, zc), filled(getattr(self, name), self.active),
                                                       bounds_error=False, fill_value=None)
        r, z = np.broadcast_arrays(r, z)
        points = np.c_[np.clip(r.ravel(), rc[0], rc[-1]), np.clip(z.ravel(), zc[0], zc[-1])]
        return self._cache[name](points).reshape(r.shape)


def remote_resistance(config):
    g, e = config.geometry, config.electrical
    # Upstream of the recessed inner lip the sheath is an annulus.
    length = max(e.contact_distance_mm*1e-3 + g.liquid_start, 0.)
    return length/(e.sheath_conductivity_s_m*np.pi*(g.c*g.c-g.b*g.b))


def solve(config, solvent):
    from ..chemistry.solvent_transport import scalar_transport
    g, e, ch = config.geometry, config.electrical, config.chemistry
    re, ze = solvent.radial_edges, solvent.axial_edges
    r, z, aa, ra, volume = metric(re, ze)
    active = solvent.active
    ambipolar = 2*D_PLUS*D_MINUS/(D_PLUS+D_MINUS)
    # Eliminating E from the two NP fluxes under local electroneutrality
    # yields ambipolar diffusion, rather than neutral-salt molecular diffusion.
    cin = np.where(r < g.a, ch.ammonium_formate_mm, 0.)
    salt, _ = scalar_transport(re, ze, active, solvent.radial_flux, solvent.axial_flux, ambipolar, cin)
    sigma_salt = FARADAY**2*(D_PLUS+D_MINUS)*salt/(GAS_CONSTANT*config.fluid.temperature_k)
    background = e.sheath_conductivity_s_m
    sigma = sigma_salt+background
    A, _ = elliptic(re, ze, sigma, active)
    # Diffusion current uses the SAME face conductances as charge conservation.
    jr_diff = np.zeros_like(solvent.radial_flux)
    jz_diff = np.zeros_like(solvent.axial_flux)
    factor = FARADAY*(D_MINUS-D_PLUS)
    jr_diff[1:-1] = factor*ra[1:-1]*np.diff(salt, axis=0)/np.diff(r)[:, None]*(active[:-1]&active[1:])
    jz_diff[:, 1:-1] = factor*aa[:, None]*np.diff(salt, axis=1)/np.diff(z)[None, :]*(active[:, :-1]&active[:, 1:])
    rhs = -(np.diff(jr_diff, axis=0)+np.diff(jz_diff, axis=1))
    signed = np.sign(e.voltage_kv)
    current = signed*(e.spray_current_na or 0.)*1e-9
    if e.contact_mode == "external_quartz" and current:
        raise ValueError("DC current through an external quartz electrode is unsupported; use a direct liquid contact")
    feed = (r > g.b)&active[:, 0]
    entry = sigma[:, 0]*aa*feed
    exit_weight = sigma[:, -1]*aa*active[:, -1]
    jin = current*entry/max(entry.sum(), 1e-300)
    jout = current*exit_weight/max(exit_weight.sum(), 1e-300)
    rhs[:, 0] += jin
    rhs[:, -1] -= jout
    fixed = np.zeros(active.shape, bool)
    fixed[tuple(np.argwhere(active)[0])] = True  # gauge only; net source is zero
    relative, residual = dirichlet_solve(A, rhs.ravel(), fixed, np.zeros(active.size))
    relative = relative.reshape(active.shape)
    resistance = remote_resistance(config)
    entry_potential = np.sum(relative[:, 0]*entry)/entry.sum()
    offset = e.voltage_kv*1000-current*resistance-entry_potential
    V = relative+offset
    lip = float(np.sum(V[:, -1]*exit_weight)/exit_weight.sum())
    if current and signed*lip < 0:
        raise ValueError("Specified spray current exceeds available HV through the 10 cm sheath resistance; measure conductivity/current")
    er = np.zeros_like(jr_diff); ez = np.zeros_like(jz_diff)
    er[1:-1] = -np.diff(relative, axis=0)/np.diff(r)[:, None]*(active[:-1]&active[1:])
    ez[:, 1:-1] = -np.diff(relative, axis=1)/np.diff(z)[None, :]*(active[:, :-1]&active[:, 1:])
    sr = 2*sigma[:-1]*sigma[1:]/(sigma[:-1]+sigma[1:])
    sz = 2*sigma[:, :-1]*sigma[:, 1:]/(sigma[:, :-1]+sigma[:, 1:])
    jr = jr_diff.copy(); jz = jz_diff.copy()
    jr[1:-1] += sr*er[1:-1]*ra[1:-1]
    jz[:, 1:-1] += sz*ez[:, 1:-1]*aa[:, None]
    jz[:, 0] = jin; jz[:, -1] = jout
    imbalance = np.diff(jr, axis=0)+np.diff(jz, axis=1)
    # Extend normal fields to inlet/outlet from specified total current.
    ez[:, 0] = jin/(np.maximum(sigma[:, 0], 1e-300)*aa)
    ez[:, -1] = jout/(np.maximum(sigma[:, -1], 1e-300)*aa)
    diag = {"method": "electroneutral binary NP ambipolar closure + conservative 2D Ohmic/diffusion current",
        "salt_diffusivity_m2_s": ambipolar, "ion_diffusivities_m2_s": [D_PLUS, D_MINUS],
        "max_cell_current_imbalance_a": float(np.max(np.abs(imbalance[active]))),
        "current_a": float(current), "relative_current_residual": residual,
        "remote_resistance_ohm": float(resistance), "remote_voltage_drop_v": float(current*resistance),
        "lip_potential_v": lip, "contact_distance_mm": e.contact_distance_mm,
        "current_boundary": "open-circuit limit; spray current not supplied" if e.spray_current_na is None else "prescribed user current; measurement provenance must be verified",
        "methanol_charge_number": 0,
        "limitations": ["electroneutral thin-double-layer approximation; no resolved space charge / electrode reactions",
            "constant ion diffusivities; supporting conductivity is an unmeasured input",
            "binary ambipolar salt closure omits concentration polarization from supporting impurity carriers",
            "current distribution at lip uses local conductivity weights; emission/current closure is not solved"]}
    return IonicResult(re, ze, salt, V, (er[1:]+er[:-1])/2, (ez[:, 1:]+ez[:, :-1])/2,
                       jr, jz, sigma, active, diag)
