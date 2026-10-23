"""Extract the phi=0.5 isosurface and connected liquid components.

Marching tetrahedra uses cell-centre samples within the stored domain.
Vertices use SI metres and can share the capillary display-path mapping.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label


_CORNERS=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]],int)
_TETS=((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6))


def isosurface(grid,values,level=.5,material=None,max_triangles=40000):
    """Piecewise-linear tetrahedral interpolation of actual scalar samples."""
    values=np.asarray(values,float)
    if values.shape!=grid.shape:raise ValueError('Phase values do not match the native grid')
    cube_shape=np.array(values.shape)-1
    vv=np.stack([values[c[0]:c[0]+cube_shape[0],c[1]:c[1]+cube_shape[1],c[2]:c[2]+cube_shape[2]]
                 for c in _CORNERS],axis=-1)
    active=np.isfinite(vv).all(axis=-1)&(vv.min(axis=-1)<level)&(vv.max(axis=-1)>=level)
    if material is not None:
        solid=np.asarray(material)==2
        # Quartz is masked; an interface does not pass through its interior.
        corners=np.stack([solid[c[0]:c[0]+cube_shape[0],c[1]:c[1]+cube_shape[1],c[2]:c[2]+cube_shape[2]]
                          for c in _CORNERS],axis=-1)
        active&=~corners.any(axis=-1)
    cells=np.argwhere(active)
    if not len(cells):return np.empty((0,3)),dict(triangles=0,display_decimated=False)
    vv=vv[active]
    coords=np.stack([np.stack([grid.centers[a][cells[:,a]+corner[a]] for a in range(3)],axis=-1)
                     for corner in _CORNERS],axis=1)
    output=[]
    for tet in _TETS:
        v=vv[:,tet];p=coords[:,tet];case=np.sum((v>=level)*(1<<np.arange(4)),axis=1)
        for code in range(1,15):
            selected=case==code
            if not selected.any():continue
            q=p[selected];w=v[selected]
            inside=[a for a in range(4) if code&(1<<a)]
            outside=[a for a in range(4) if not code&(1<<a)]
            def cross(a,b):
                f=(level-w[:,a])/np.maximum(np.abs(w[:,b]-w[:,a]),1e-30)*np.sign(w[:,b]-w[:,a])
                return q[:,a]+f[:,None]*(q[:,b]-q[:,a])
            if len(inside)==1:
                output.append(np.stack([cross(inside[0],a) for a in outside],axis=1))
            elif len(outside)==1:
                output.append(np.stack([cross(outside[0],a) for a in inside[::-1]],axis=1))
            else:
                a,b=inside;c,d=outside
                ac,ad,bc,bd=cross(a,c),cross(a,d),cross(b,c),cross(b,d)
                output.extend([np.stack([ac,ad,bd],axis=1),np.stack([ac,bd,bc],axis=1)])
    triangles=np.concatenate(output,axis=0) if output else np.empty((0,3,3))
    physical_triangles=len(triangles)
    if physical_triangles>max_triangles:
        # Only renderer load is bounded. Numerical fields/component volumes
        # remain complete; the explicit flag makes display omission visible.
        triangles=triangles[np.linspace(0,physical_triangles-1,max_triangles,dtype=int)]
    return triangles.reshape(-1,3),dict(triangles=len(triangles),
        display_decimated=physical_triangles>max_triangles,total_surface_triangles=physical_triangles,
        surface_method='cell-centred phi=0.5 marching tetrahedra; no boundary extrapolation')


def phase_geometry(phase,frame=-1,path=None,inner_tip_m=0.,max_triangles=40000):
    count=len(phase.time_s);frame=int(np.clip(frame if frame>=0 else count-1,0,count-1))
    values=phase.liquid_fraction[frame]
    points,diag=isosurface(phase.grid,values,material=phase.material,max_triangles=max_triangles)
    if path is not None and len(points):
        points=path.map(path.length_m+points[:,2]-inner_tip_m,points[:,0],points[:,1])
    meshes=[]
    if len(points):
        meshes.append(dict(id='native_phase_interface',positions=points.ravel().tolist(),
            indices=list(range(len(points))),color=[.23,.79,.94,1.],opacity=.58,
            doubleSided=True,layer='phase'))
    masks,number=label((values>=.5)&(phase.material!=2))
    droplets=[]
    for identifier in range(1,number+1):
        mask=masks==identifier;idx=np.argwhere(mask)
        volume=float(np.sum(np.clip(values[mask],0,1)*phase.grid.volume[mask]))
        weights=np.clip(values[mask],0,1)*phase.grid.volume[mask]
        centers=np.stack([phase.grid.centers[a][idx[:,a]] for a in range(3)],axis=1)
        centroid=np.sum(centers*weights[:,None],axis=0)/max(weights.sum(),1e-30)
        touches=bool(np.any(idx==0)|np.any(idx==(np.array(values.shape)-1)))
        inlet=bool(np.any(idx[:,2]==0))
        if path is not None:centroid=path.map(path.length_m+centroid[2]-inner_tip_m,centroid[0],centroid[1])
        radius=(3*volume/(4*np.pi))**(1/3)
        occupied=np.nonzero(mask)
        local_h=max(float(np.max(phase.grid.widths[axis][np.unique(occupied[axis])])) for axis in range(3))
        spans=[len(np.unique(occupied[axis])) for axis in range(3)]
        droplets.append(dict(id=identifier,volume_m3=volume,equivalent_radius_m=radius,
            centroid_m=centroid.tolist(),cells=int(mask.sum()),connected_to_upstream_boundary=inlet,
            detached=not inlet,touches_outer_boundary=touches,
            occupied_cells_per_axis=spans,largest_occupied_cell_width_m=local_h,
            resolved=bool(mask.sum()>=27 and min(spans)>=3 and radius>=1.5*local_h and not touches)))
    return dict(meshes=meshes,lines=[],phase_frame=frame,phase_time_s=float(phase.time_s[frame]),
        phase_clock='independent_short_native_transient',droplets=droplets,metadata=diag)
