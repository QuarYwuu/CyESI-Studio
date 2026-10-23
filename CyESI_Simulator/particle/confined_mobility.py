"""Reduced many-body mobility in a locally straight circular tube.

Stokes collocation enforces no slip on a finite cylindrical wall. Blob mobility
is calibrated to the single-sphere wall mobility. Pair squeeze resistance is
added in resistance space with nonnegative dissipation. Membranes are unresolved.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from .forces import lubrication_factors


REFERENCES=[
    dict(title='Cortez: The method of regularized Stokeslets',url='https://doi.org/10.1137/S106482750038146X'),
    dict(title='Costa et al.: lubrication and contact for finite-size particles',url='https://arxiv.org/abs/1506.01880'),
    dict(title='Aouane et al.: soft particle pairing in confined flow',url='https://doi.org/10.1103/PhysRevFluids.2.063102'),
    dict(title='Humphry et al.: axial and lateral ordering',url='https://doi.org/10.1063/1.3478311')]


def green(target,source,mu,epsilon):
    """Cortez regularized force Green tensor, [target,source,3,3], SI."""
    d=np.asarray(target)[:,None,:]-np.asarray(source)[None,:,:]
    r2=np.sum(d*d,axis=-1);den=8*np.pi*mu*(r2+epsilon**2)**1.5
    return ((r2+2*epsilon**2)[...,None,None]*np.eye(3)+d[..., :,None]*d[...,None,:])/den[...,None,None]


def matrix(tensor):
    n,m=tensor.shape[:2]
    return tensor.transpose(0,2,1,3).reshape(3*n,3*m)


def stresslet_velocity(target,source,strain,radius,epsilon):
    """Leading force-free sphere stresslet, smoothed only at its centre."""
    d=np.asarray(target)[:,None,:]-np.asarray(source)[None,:,:]
    r2=np.sum(d*d,axis=-1)+epsilon**2
    de=np.einsum('tni,nij,tnj->tn',d,strain,d)
    return np.sum(-2.5*radius**3*de[...,None]*d/r2[...,None]**2.5,axis=1)


def self_mobility(xyz,radius,tube_radius,mu,roughness):
    r=np.linalg.norm(xyz[:,:2],axis=1);normal=np.c_[xyz[:,:2]/np.maximum(r[:,None],1e-30),np.zeros(len(r))]
    normal[r<1e-25]=[1.,0.,0.]
    gap=np.maximum(tube_radius-r-radius,roughness)
    fn,fp=lubrication_factors(radius,gap)
    zeta=6*np.pi*mu*radius
    return ((np.eye(3)[None,:,:]/fp[:,None,None])+
        (1/fn-1/fp)[:,None,None]*normal[:,:,None]*normal[:,None,:])/zeta


def squeeze_resistance(xyz,radius,mu,roughness,cutoff_gap=None):
    """Leading sphere-sphere squeeze divergence minus its cutoff value.

    F_i=-k nn.(v_i-v_j), F_j=-F_i. This changes relative motion and cannot
    propel a common translation or manufacture a preferred finite spacing.
    """
    xyz=np.asarray(xyz);n=len(xyz);out=np.zeros((3*n,3*n))
    cutoff_gap=radius if cutoff_gap is None else cutoff_gap
    pairs=cKDTree(xyz).query_pairs(2*radius+cutoff_gap,output_type='ndarray')
    for i,j in pairs:
        delta=xyz[i]-xyz[j];distance=np.linalg.norm(delta)
        direction=delta/max(distance,1e-30)
        if distance<1e-25:direction=np.array([1.,0.,0.])
        h=max(distance-2*radius,roughness)
        k=6*np.pi*mu*(radius/2)**2*max(0.,1/h-1/cutoff_gap)
        block=k*np.outer(direction,direction);a=slice(3*i,3*i+3);b=slice(3*j,3*j+3)
        out[a,a]+=block;out[b,b]+=block;out[a,b]-=block;out[b,a]-=block
    return out


def sqrt_matrix(a,inverse=False):
    values,vectors=np.linalg.eigh(.5*(a+a.T))
    threshold=max(float(np.max(values))*1e-12,1e-40)
    if np.min(values)<-threshold:raise ValueError('Mobility lost positive definiteness; refine wall collocation')
    values=np.maximum(values,threshold)
    return (vectors*((1/np.sqrt(values)) if inverse else np.sqrt(values)))@vectors.T


class TubeKernel:
    def __init__(self,tube_radius,radius,mu,theta=12,rings=7,half_length_radii=5.):
        self.a=float(tube_radius);self.R=float(radius);self.mu=float(mu)
        self.epsilon=1.5*self.R;self.length=half_length_radii*self.a
        angle=np.arange(theta)*2*np.pi/theta;z=np.linspace(-self.length,self.length,rings)
        t,Z=np.meshgrid(angle,z)
        self.wall=np.c_[self.a*np.cos(t.ravel()),self.a*np.sin(t.ravel()),Z.ravel()]
        self.ww=matrix(green(self.wall,self.wall,mu,self.epsilon))
        self.factor=cho_factor(self.ww,lower=True,check_finite=False)
        self.max_wall_residual=0.;self.evaluations=0

    def _local(self,xyz):
        x=np.array(xyz,float,copy=True);x[:,2]-=.5*(x[:,2].min()+x[:,2].max())
        if np.ptp(x[:,2])>self.length:
            raise ValueError('Neighbour cluster exceeds local wall length; increase wall half-length')
        return x

    def mobility(self,xyz,roughness,wall=True):
        x=self._local(xyz);n=len(x)
        H=matrix(green(x,x,self.mu,self.epsilon))
        if wall:
            cw=matrix(green(x,self.wall,self.mu,self.epsilon))
            H-=cw@cho_solve(self.factor,cw.T,check_finite=False)
        H=.5*(H+H.T)
        target=self_mobility(x,self.R,self.a,self.mu,roughness)
        transform=np.zeros_like(H)
        for i in range(n):
            s=slice(3*i,3*i+3)
            transform[s,s]=sqrt_matrix(target[i])@sqrt_matrix(H[s,s],inverse=True)
        M=transform@H@transform.T
        self.evaluations+=1
        return .5*(M+M.T)

    def disturbance(self,xyz,strain,wall=True,deformation=None):
        """Pair strain fields and wall reflections; rigid self lift is excluded.

        Optional deformation enters as a small-strain constitutive stresslet
        sensitivity, not a resolved Skalak/Helfrich membrane solution.
        """
        x=self._local(xyz);E=np.asarray(strain);n=len(x)
        extra=np.zeros_like(E)
        if deformation is not None:
            Q=np.asarray(deformation)
            extra=Q@E+E@Q
            extra-=np.trace(extra,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
        out=stresslet_velocity(x,x,E+extra,self.R,self.epsilon)
        if wall:
            # Subtract each sphere's self-image to avoid duplicate wall self-mobility.
            cw=matrix(green(x,self.wall,self.mu,self.epsilon))
            for j in range(n):
                uw=stresslet_velocity(self.wall,x[j:j+1],(E+extra)[j:j+1],self.R,self.epsilon)
                force=cho_solve(self.factor,-uw.ravel(),check_finite=False)
                correction=(cw@force).reshape(n,3)
                rigid_wall=stresslet_velocity(self.wall,x[j:j+1],E[j:j+1],self.R,self.epsilon)
                rigid_force=cho_solve(self.factor,-rigid_wall.ravel(),check_finite=False)
                correction[j]-=(cw[3*j:3*j+3]@rigid_force)
                out+=correction
                residual=self.ww@force+uw.ravel()
                self.max_wall_residual=max(self.max_wall_residual,float(np.max(np.abs(residual))))
        return out

    def born_inertia(self,xyz,strain,external_force,umean,rho):
        """One Born iteration of the linearized disturbance convection term.

        -mu Laplacian(u1)+grad(p1) = -rho[(U.grad)u0+(u0.grad)U].
        A low-order volume quadrature and the same no-slip side-wall projection
        are used. This is a switchable low-Re sensitivity, NOT a complete
        finite-Re particle wake / inertial train solver.
        """
        x=self._local(xyz);a=self.a
        nodes,weights=np.polynomial.legendre.leggauss(3)
        radii=(nodes+1)*a/2;rw=weights*a/2
        zn,zw=np.polynomial.legendre.leggauss(7);z=zn*self.length;zw*=self.length
        angles=np.arange(8)*2*np.pi/8
        rr,tt,zz=np.meshgrid(radii,angles,z,indexing='ij')
        q=np.c_[rr.ravel()*np.cos(tt.ravel()),rr.ravel()*np.sin(tt.ravel()),zz.ravel()]
        volume=(rw[:,None,None]*radii[:,None,None]*np.ones((1,8,1))*(2*np.pi/8)*zw[None,None,:]).ravel()
        force=np.asarray(external_force).ravel()
        def free(points):
            return (matrix(green(points,x,self.mu,self.epsilon))@force).reshape(-1,3)+stresslet_velocity(points,x,strain,self.R,self.epsilon)
        wall_force=cho_solve(self.factor,-free(self.wall).ravel(),check_finite=False)
        def u0(points):return free(points)+(matrix(green(points,self.wall,self.mu,self.epsilon))@wall_force).reshape(-1,3)
        eps=a*.01;offset=np.array([0.,0.,eps])
        uq=u0(q);du_dz=(u0(q+offset)-u0(q-offset))/(2*eps)
        uz=2*umean*(1-(q[:,0]**2+q[:,1]**2)/a**2)
        convective=uz[:,None]*du_dz
        convective[:,2]+=-4*umean/a**2*(q[:,0]*uq[:,0]+q[:,1]*uq[:,1])
        body=-rho*convective*volume[:,None]
        wc=matrix(green(self.wall,q,self.mu,self.epsilon))@body.ravel()
        reaction=cho_solve(self.factor,-wc,check_finite=False)
        return ((matrix(green(x,q,self.mu,self.epsilon))@body.ravel())+
                matrix(green(x,self.wall,self.mu,self.epsilon))@reaction).reshape(-1,3)


def membrane_step(Q,gradient,dt,relaxation_s):
    """Linear objective Kelvin relaxation, trace-free, exact frozen relaxation.

    Q is a dimensionless logarithmic shape tensor. tau=mu*R/Gs is an editable
    constitutive timescale. exp(eigenvalues(Q)) gives volume-preserving axes.
    It is an explicit small-strain sensitivity, not calibrated cell rheology.
    """
    E=.5*(gradient+gradient.swapaxes(-1,-2));W=.5*(gradient-gradient.swapaxes(-1,-2))
    omega=np.sqrt(np.maximum(0.,-.5*np.trace(W@W,axis1=-2,axis2=-1)))
    def rotate(time):
        angle=omega*time
        s=np.divide(np.sin(angle),omega,out=np.full_like(omega,time),where=omega>1e-30)
        c=np.divide(1-np.cos(angle),omega**2,out=np.full_like(omega,time*time/2),where=omega>1e-30)
        return np.eye(3)+s[:,None,None]*W+c[:,None,None]*(W@W)
    decay=np.exp(-dt/relaxation_s);rotation=rotate(dt)
    out=decay*(rotation@Q@rotation.swapaxes(-1,-2))
    # y=1-exp(-s/tau) makes the stiff relaxation quadrature uniform even
    # when dt >> tau. Rotation is integrated instead of an unstable Euler step.
    nodes,weights=np.polynomial.legendre.leggauss(8)
    for node,weight in zip(nodes,weights):
        y=.5*(node+1)*(1-decay);s=-relaxation_s*np.log1p(-y)
        rotation=rotate(s)
        out+=relaxation_s*(1-decay)*weight*(rotation@E@rotation.swapaxes(-1,-2))
    out-=np.trace(out,axis1=1,axis2=2)[:,None,None]*np.eye(3)/3
    return out


def solve_group(xyz,ambient,strain,force,kernel,roughness,*,coupling=True,wall=True,
                lubrication=True,deformation=None,inertia=False,umean=0.,rho=1000.,fixed=None):
    n=len(xyz)
    if coupling and n>1:
        mobility=kernel.mobility(xyz,roughness,wall)
        induced=kernel.disturbance(xyz,strain,wall,deformation)
    else:
        mobility=np.zeros((3*n,3*n));blocks=self_mobility(xyz,kernel.R,kernel.a,kernel.mu,roughness)
        for i in range(n):mobility[3*i:3*i+3,3*i:3*i+3]=blocks[i]
        induced=np.zeros_like(xyz)
        if deformation is not None and wall and np.max(np.abs(deformation))>0:
            for i in range(n):induced[i]=kernel.disturbance(xyz[i:i+1],strain[i:i+1],True,deformation[i:i+1])[0]
    v0=np.asarray(ambient)+induced+(mobility@np.asarray(force).ravel()).reshape(n,3)
    if inertia:
        v0+=kernel.born_inertia(xyz,strain,force,umean,rho)
    R=squeeze_resistance(xyz,kernel.R,kernel.mu,roughness) if lubrication else np.zeros_like(mobility)
    fixed=np.zeros(n,bool) if fixed is None else np.asarray(fixed,bool)
    if np.any(R) or fixed.any():
        # (M^-1+R)v=M^-1 v0; use a symmetric positive system, not a frozen
        # squeeze force which would destabilize dt >> Stokes relaxation time.
        inverse=np.linalg.inv(mobility)
        resistance=inverse+R;rhs=inverse@v0.ravel();free=np.repeat(~fixed,3)
        flat=np.zeros(n*3)
        if free.any():flat[free]=np.linalg.solve(resistance[np.ix_(free,free)],rhs[free])
        v=flat.reshape(n,3)
    else:v=v0
    return v,dict(max_induced_m_s=float(np.max(np.linalg.norm(induced,axis=1))) if n else 0.,
        lubrication_dissipation_w=float(v.ravel()@R@v.ravel()))
