"""Counter-based Wiener paths for comparisons of adaptive time steps.

A finite Levy-Ciesielski expansion supplies the same free-space noise path
for each event. Wall-correlated fluctuation-dissipation noise is excluded.
"""
import numpy as np


def _hash(x):
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return x^(x>>np.uint64(31))


def _normal(ids,seed,level,index):
    with np.errstate(over='ignore'):
        key=np.asarray(ids,dtype=np.uint64)[:,None]*np.uint64(0x9e3779b97f4a7c15)
        key=key^((np.arange(3,dtype=np.uint64)[None,:]+1)*np.uint64(0xd2b74407b1ce6e93))
        key=key^np.uint64(seed)^(_hash(np.uint64(level+1))*np.uint64(0xa0761d6478bd642f))^(_hash(np.uint64(index+1))*np.uint64(0xe7037ed1a0b428db))
        a=_hash(key);b=_hash(key^np.uint64(0x632be59bd9b4e019))
    u=((a>>np.uint64(11)).astype(float)+.5)/2**53
    v=((b>>np.uint64(11)).astype(float)+.5)/2**53
    return np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)


class ThermalPath:
    def __init__(self,origin,end,seed,levels=24):
        self.origin=float(origin);self.horizon=2.**np.ceil(np.log2(max(end-origin,1.)))
        self.seed=int(seed);self.levels=levels;self.resolution_s=self.horizon/2**levels

    def value(self,ids,time):
        x=float(np.clip((time-self.origin)/self.horizon,0,1));root=np.sqrt(self.horizon)
        value=root*x*_normal(ids,self.seed,0,0)
        for level in range(self.levels):
            q=x*2**level;index=int(np.floor(q));fraction=q-index
            tent=1-abs(2*fraction-1)
            if tent>0:value+=root*2**(-(level+2)/2)*tent*_normal(ids,self.seed,level+1,index)
        return value
