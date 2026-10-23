"""Axisymmetric field adapter for the shared continuous finite-sphere model.

The background remains a solved (r,z) axisymmetric field. Physical cells have
three coordinates: a particle is never interpreted as an axisymmetric ring.
Azimuth is inherited from saved bulk x/y history; missing azimuth is reported.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .interactions import _TipField, simulate_interactions
from ..scene3d.data import SceneData


class AxisymmetricTipField(_TipField):
    model_name='continuous_common_clock_axisymmetric_tip'
    scope=('Solved axisymmetric r/z background; physical Cartesian finite spheres with inherited azimuth. '
           'Shared dilute neighbour/contact/bond laws; local tip only, not a full-length many-body CFD solve.')

    def __init__(self,scene):
        data=scene.data;cfg=data.config;self.cfg=cfg;f=data.fields
        self.axes=(data.r,data.z)
        if min(map(len,self.axes))<2:raise ValueError('Axisymmetric interaction fields need at least two cells per axis')
        self.inlet=float(f['liquid_z_edges_m'][0])
        self.min_grid=min(float(np.diff(f[key]).min()) for key in ('liquid_r_edges_m','liquid_z_edges_m'))
        self.lower=np.array([-cfg.geometry.c,-cfg.geometry.c,data.z[0]])
        self.upper=np.array([cfg.geometry.c,cfg.geometry.c,data.z[-1]])
        active=f['liquid_active'].astype(bool)
        if not active.any():raise ValueError('No solved liquid cells')
        nearest=distance_transform_edt(~active,return_distances=False,return_indices=True)
        def fill(a):return np.asarray(a)[tuple(nearest)]
        ur=fill(f['velocity_r_m_s']);uz=fill(f['velocity_z_m_s'])
        C=fill(f['methanol_fraction'])
        r,z=np.meshgrid(*self.axes,indexing='ij')
        er=fill(data.sample(r,z,'field_r_v_m'));ez=fill(data.sample(r,z,'field_z_v_m'))
        ur_r,ur_z=np.gradient(ur,*self.axes);uz_r,uz_z=np.gradient(uz,*self.axes)
        e2_r,e2_z=np.gradient(er*er+ez*ez,*self.axes)
        channels=np.stack([ur,uz,er,ez,C,ur_r,ur_z,uz_r,uz_z,e2_r,e2_z],axis=-1)
        self.sampler=RegularGridInterpolator(self.axes,channels,bounds_error=True)
        self.joint=self.cartesian_channels
        from ..fluid.supply import segments
        self.gravity=np.array([segments(cfg)[-1][k] for k in ('gx','gy','gz')]) if cfg.supply.gravity_enabled else np.zeros(3)
        self.salt_sampler=None
        ion_file=Path(data.source)/'ions.npz'
        if ion_file.is_file():
            with np.load(ion_file,allow_pickle=False) as a:
                ia=tuple(.5*(a[k][1:]+a[k][:-1]) for k in ('radial_edges_m','axial_edges_m'))
                near=distance_transform_edt(~a['active'].astype(bool),return_distances=False,return_indices=True)
                self.salt_sampler=RegularGridInterpolator(ia,a['salt_mol_m3'][tuple(near)],bounds_error=False,fill_value=None)
                self.salt_axes=ia
        self.salt_scope='saved ionic transport' if self.salt_sampler is not None else 'unresolved salt: explicit dilution proxy I0*(1-C), not a new salt solve'

    def cartesian_channels(self,xyz):
        xyz=np.asarray(xyz,float);r=np.hypot(xyz[:,0],xyz[:,1])
        n=np.divide(xyz[:,:2],r[:,None],out=np.zeros_like(xyz[:,:2]),where=r[:,None]>1e-25)
        n[r<1e-25]=[1.,0.]
        rz=np.c_[np.clip(r,self.axes[0][0],self.axes[0][-1]),np.clip(xyz[:,2],self.axes[1][0],self.axes[1][-1])]
        a=self.sampler(rz)
        radial_factor=np.minimum(1.,r/self.axes[0][0]) if self.axes[0][0]>0 else np.ones_like(r)
        ur=np.where(r<self.axes[0][0],a[:,5]*r,a[:,0])
        u=np.c_[ur[:,None]*n,a[:,1]];electric=np.c_[(a[:,2]*radial_factor)[:,None]*n,a[:,3]]
        # A regular axisymmetric field has u_r=a*r on the centreline. Use its
        # derivative there rather than divide the first off-axis cell by zero.
        over_r=np.divide(a[:,0],r,out=a[:,5].copy(),where=r>self.axes[0][0])
        grad=np.zeros((len(xyz),3,3))
        grad[:,:2,:2]=over_r[:,None,None]*np.eye(2)+(a[:,5]-over_r)[:,None,None]*n[:,:,None]*n[:,None,:]
        grad[:,:2,2]=a[:,6,None]*n;grad[:,2,:2]=a[:,7,None]*n;grad[:,2,2]=a[:,8]
        strain=.5*(grad+np.swapaxes(grad,1,2));strain-=np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
        omega=np.c_[grad[:,2,1]-grad[:,1,2],grad[:,0,2]-grad[:,2,0],grad[:,1,0]-grad[:,0,1]]
        ge=np.c_[(a[:,9]*radial_factor)[:,None]*n,a[:,10]]
        # At r=0 regular radial vectors vanish, while gradients remain finite.
        u[r<1e-25,:2]=0.;electric[r<1e-25,:2]=0.;ge[r<1e-25,:2]=0.
        return np.c_[u,electric,a[:,4],ge,strain.reshape(-1,9),omega]

    def charge_samples(self,points):
        a=self.cartesian_channels(points)
        if self.salt_sampler is None:I=self.cfg.chemistry.ammonium_formate_mm*(1-np.clip(a[:,6],0,1))
        else:
            rz=np.c_[np.hypot(points[:,0],points[:,1]),points[:,2]]
            rz=np.clip(rz,[x[0] for x in self.salt_axes],[x[-1] for x in self.salt_axes])
            I=self.salt_sampler(rz)
        return np.maximum(I,0.),a[:,3:6]


class AxisymmetricScene(SceneData):
    def __init__(self,data,model):
        # Reuse the already built continuous feed, not a second random train.
        self.data=data;self.model=model;self.path=None
        self._xyz={};self._geometry={};self._flights=OrderedDict()
        self._bulk_columns={int(k):i for i,k in enumerate(data.fields.get('bulk_trajectory_ids',[]))}
        self.source_id=hashlib.sha256((data.source+repr(data.config.to_dict())).encode()).hexdigest()[:16]
        from ..scene3d.cell_charge import ChargeParameters
        self.charge_parameters=ChargeParameters();self._charge_cache={}
        self.field_provider=AxisymmetricTipField(self)

    def charge_samples(self,points):return self.field_provider.charge_samples(points)


def simulate_axisymmetric(data,model,parameters,progress=lambda value:None):
    scene=AxisymmetricScene(data,model)
    result=simulate_interactions(scene,parameters,progress,field_provider=scene.field_provider)
    result.diagnostics['saved_bulk_azimuth']=bool('bulk_trajectory_x_m' in data.fields)
    if not result.diagnostics['saved_bulk_azimuth']:
        result.diagnostics['notes'].append('Archived responses lack azimuth; their meridian placement is an explicit initial-condition assumption. Recompute bulk histories for quantitative contacts.')
    return scene,result
