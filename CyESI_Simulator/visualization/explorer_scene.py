"""Pillow scene renderer for the two native visual workspaces.

Each frame is also a standalone exportable PNG. Geometry, concentration,
tracks and electric lines come from ExplorerData. Only the explicitly amber
mechanism view uses an assumed Taylor reference / drop / membrane drawing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..geometry.coaxial_geometry import classify, Domain
from .explorer_data import ExplanationSettings, rayleigh_reference

BG=(7,18,31); PANEL=(13,32,50); BORDER=(32,61,82); WHITE=(227,238,246)
MUTED=(135,165,185); WATER=(63,164,244); METHANOL=(250,166,67)
GAS=(66,208,189); ELECTRIC=(190,153,255); CELL=(167,242,159)
AMBER=(247,187,91)


@lru_cache(maxsize=64)
def font(size=15, bold=False):
    candidates=[Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.exists(): return ImageFont.truetype(str(path), size)
    try: return ImageFont.truetype("DejaVuSans.ttf",size)
    except OSError: return ImageFont.load_default()


def blend(a,b,f):
    return tuple(int(x*(1-f)+y*f) for x,y in zip(a,b))


def solvent_color(C): return blend(WATER,METHANOL,float(np.clip(C,0,1)))


class AxisMap:
    def __init__(self, rect, zmin, zmax, rmax, isotropic=False):
        self.rect=rect;self.zmin=zmin;self.zmax=zmax;self.rmax=rmax
        x0,y0,x1,y1=rect
        self.sx=(x1-x0)/(zmax-zmin);self.sy=(y1-y0)/(2*rmax)
        if isotropic: self.sx=self.sy=min(self.sx,self.sy)
        self.left=x0+((x1-x0)-(zmax-zmin)*self.sx)/2
        self.top=y0+((y1-y0)-2*rmax*self.sy)/2
        self.center=(y0+y1)/2

    def point(self,z,r=0.): return self.left+(z-self.zmin)*self.sx,self.center-r*self.sy
    def inverse(self,x,y): return (x-self.left)/self.sx+self.zmin,(self.center-y)/self.sy
    def pixels(self):
        return (int(self.left),int(self.top),int(self.left+(self.zmax-self.zmin)*self.sx),int(self.top+2*self.rmax*self.sy))


class SceneRenderer:
    def __init__(self,data):
        self.data=data;self.cache={};self.hits=[];self.wrap_cache={}
        self.field_lines=self._field_lines()
        self.streamlines=self._streamlines()

    def text(self,xy,text,size=15,color=WHITE,bold=False,anchor=None):
        self.draw.text(xy,str(text),font=font(size,bold),fill=color,anchor=anchor)

    def wrapped(self,xy,text,max_width,size=12,color=MUTED,line_height=15):
        key=(text,int(max_width),size)
        if key not in self.wrap_cache:
            lines=[];line=""
            for character in text:
                if line and self.draw.textlength(line+character,font=font(size))>max_width:
                    lines.append(line);line=""
                line+=character
            lines.append(line);self.wrap_cache[key]=lines
        for row,line in enumerate(self.wrap_cache[key]):self.text((xy[0],xy[1]+row*line_height),line,size,color)

    def panel(self,rect,radius=14,fill=PANEL):
        self.draw.rounded_rectangle(rect,radius,fill=fill,outline=BORDER,width=1)

    def pill(self,x,y,text,color=GAS):
        width=self.draw.textlength(text,font=font(13))+20
        self.draw.rounded_rectangle((x,y,x+width,y+26),9,fill=blend(BG,color,.13),outline=blend(BG,color,.4))
        self.text((x+10,y+4),text,13,color)
        return width

    def arrow(self,start,end,color,width=2,head=7):
        self.draw.line([start,end],fill=color,width=width)
        a=np.array(end,float);d=a-np.array(start,float);norm=np.linalg.norm(d)
        if norm<1:return
        d/=norm;n=np.array([-d[1],d[0]])
        points=[tuple(a),tuple(a-d*head+n*head*.45),tuple(a-d*head-n*head*.45)]
        self.draw.polygon(points,fill=color)

    def dashed(self,start,end,color,dash=7,width=1):
        a,b=np.array(start,float),np.array(end,float);length=np.linalg.norm(b-a)
        if length<1:return
        for d in np.arange(0,length,dash*2):
            self.draw.line([tuple(a+(b-a)*d/length),tuple(a+(b-a)*min(d+dash,length)/length)],fill=color,width=width)

    def cell(self,x,y,radius=7,C=0.,selected=False,integrity=None,released=0.):
        color=blend(CELL,METHANOL,float(C)*.65)
        if selected:
            self.draw.ellipse((x-radius-6,y-radius-6,x+radius+6,y+radius+6),outline=blend(BG,GAS,.65),width=2)
        self.draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=blend(PANEL,color,.8))
        membrane=WHITE if integrity is None else blend(METHANOL,CELL,float(integrity))
        if integrity is None or integrity>.98:
            self.draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=membrane,width=max(1,int(radius/10)))
        else:
            for a in range(0,360,45):
                self.draw.arc((x-radius,y-radius,x+radius,y+radius),a,a+max(5,43*integrity),fill=membrane,width=2)
        nr=radius*.42
        self.draw.ellipse((x-nr,y-nr,x+nr,y+nr),fill=(45,94,105),outline=blend(WHITE,color,.7))
        if radius>16:
            for j in range(10):
                theta=j*2.4;rr=radius*(.52+.19*math.sin(j))
                xx,yy=x+rr*math.cos(theta),y+rr*math.sin(theta)
                self.draw.ellipse((xx-1.7,yy-1.7,xx+1.7,yy+1.7),fill=METHANOL)
        for j in range(int(released*22)):
            theta=j*2.399;rr=radius*(1.25+.08*j)
            xx,yy=x+rr*math.cos(theta),y+rr*math.sin(theta)
            self.draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill=METHANOL)

    def _field_lines(self):
        data=self.data;g=data.config.geometry;lines=[]
        if data.config.electrical.voltage_kv==0:return lines
        for sign in [-1,1]:
            for seed in np.linspace(.04,.95,7):
                r=sign*g.c*seed;z=12e-6;points=[]
                for _ in range(240):
                    points.append((z,r))
                    er=float(data.sample(r,z,"field_r_v_m"));ez=float(data.sample(r,z,"field_z_v_m"))
                    norm=math.hypot(er,ez)
                    if norm<1e-9:break
                    # Orient lines from the applied electrode towards ground.
                    polarity=1 if data.config.electrical.voltage_kv>0 else -1
                    dz=polarity*ez/norm*12e-6;dr=polarity*er/norm*12e-6
                    z+=dz;r+=dr
                    if z>g.ms_gap_mm*1e-3 or z<0 or abs(r)>g.external_radius_mm*1e-3:break
                if len(points)>4:lines.append(points)
        return lines

    def _streamlines(self):
        data=self.data;g=data.config.geometry;lines=[]
        start=max(g.liquid_start+1e-9,min(-1.8e-3,g.inner_tip-.35e-3))
        radii=np.r_[np.linspace(0,g.a*.82,5),np.linspace(g.b+(g.c-g.b)*.1,g.c-(g.c-g.b)*.1,4)]
        for sign in [-1,1]:
            for initial in radii:
                r=initial;z=start;points=[]
                for _ in range(420):
                    points.append((z,r*sign))
                    ur=float(data.sample(r,z,"velocity_r_m_s"));uz=float(data.sample(r,z,"velocity_z_m_s"))
                    if uz<1e-6:break
                    dz=5e-6;new_r=r+ur/uz*dz
                    z+=dz;r=abs(new_r)
                    if z>=0 or r>=g.c:break
                    if z<g.inner_tip and g.a<r<g.b:break
                if len(points)>5:lines.append(points)
        return lines

    def render(self,width,height,page="device",progress=.78,selected=0,phase=0.,
               show_field=True,show_cells=True,show_streams=True,explanation=False,settings=None):
        width=max(900,int(width));height=max(560,int(height))
        self.image=Image.new("RGB",(width,height),BG);self.draw=ImageDraw.Draw(self.image)
        self.hits=[];self.width=width;self.height=height
        if page=="device":self._device(progress,selected,phase,show_field,show_cells,show_streams)
        else:self._nozzle(progress,selected,phase,show_field,explanation,settings or ExplanationSettings())
        return self.image

    def _title(self,title,subtitle,badge,color=GAS):
        self.text((24,15),title,24,bold=True)
        self.text((25,48),subtitle,13,MUTED)
        bw=self.draw.textlength(badge,font=font(13))+20
        self.pill(self.width-bw-24,22,badge,color)

    def _overview(self,rect,snapshot):
        x0,y0,x1,y1=rect;self.panel(rect)
        w=x1-x0;cy=y0+75
        self.text((x0+18,y0+12)," Sample to MS inlet ",15,bold=True)
        # Sample vial and the long tube are intentionally folded in screen space.
        vx=x0+46
        self.draw.rounded_rectangle((vx-17,cy-25,vx+17,cy+25),7,fill=(20,51,75),outline=WATER,width=2)
        self.draw.rectangle((vx-14,cy-1,vx+14,cy+20),fill=blend(BG,WATER,.65))
        self.cell(vx,cy+7,4)
        tube0=vx+28;tube1=x0+w*.52
        self.draw.line((tube0,cy,tube1,cy),fill=(60,93,115),width=13)
        self.draw.line((tube0,cy,tube1,cy),fill=blend(BG,WATER,.8),width=7)
        for fraction in [.47,.49]:
            x=tube0+(tube1-tube0)*fraction
            self.draw.line((x-3,cy+10,x+3,cy-10),fill=BG,width=5)
        self.text(((tube0+tube1)/2,cy-30),f" Sample capillary  {self.data.config.geometry.sample_length_mm/10:g} cm ·  Compressed axial scale ",13,MUTED,anchor="mm")
        self.text((vx,cy+40)," Cell suspension ",12,WATER,anchor="mm")
        end=x0+w*.73;body=tube1+18
        for offset,col in [(22,GAS),(13,METHANOL),(5,WATER)]:
            self.draw.rounded_rectangle((body,cy-offset,end,cy+offset),max(2,offset//2),fill=blend(BG,col,.34),outline=col)
        for label,color,yy in [("N2  Nebulizing gas ",GAS,cy-32),(" Methanol sheath ",METHANOL,cy+34)]:
            bx=body-101
            self.draw.rounded_rectangle((bx-12,yy-13,bx+9,yy+12),4,fill=blend(PANEL,color,.3),outline=color)
            self.draw.rectangle((bx-7,yy-17,bx+4,yy-12),fill=color)
            self.text((body-12,yy),label,12,color,anchor="rm")
            self.arrow((body-9,yy),(body+25,cy+(-16 if yy<cy else 10)),color,head=5)
        ms=x0+w*.89
        self.draw.rounded_rectangle((ms-22,cy-37,ms+32,cy+38),8,fill=(23,44,65),outline=(81,113,140))
        self.draw.ellipse((ms-26,cy-15,ms-17,cy+15),fill=BG,outline=GAS,width=2)
        self.text((ms+4,cy),"MS",17,WHITE,bold=True,anchor="mm")
        self.arrow((end+9,cy),(ms-32,cy),ELECTRIC,head=8)
        self.text(((end+ms-20)/2,cy+29),f"{self.data.config.geometry.ms_gap_mm:g} mm",12,MUTED,anchor="mm")
        g=self.data.config.geometry
        if snapshot.z_m<g.liquid_start:
            fraction=(snapshot.z_m-(g.inner_tip-g.sample_length_mm*1e-3))/max(g.bulk_length,1e-12)
            x=tube0+(tube1-tube0)*np.clip(fraction,0,1)
        else:
            fraction=(snapshot.z_m-g.liquid_start)/max(-g.liquid_start,1e-12)
            x=body+(end-body)*np.clip(fraction,0,1)
        self.cell(x,cy,7,snapshot.methanol,True)
        self.pill(end-38,y0+10,f" Electrode  {self.data.config.electrical.voltage_kv:+g} kV",ELECTRIC)
        self.text((body+8,cy-43),f" Quartz capillary  ·  Electrode-to-emitter distance  {self.data.config.electrical.contact_distance_mm/10:g} cm",11,ELECTRIC)
        ionic=self.data.summary.get("electric",{}).get("ionic")
        if ionic is not None:
            note=" Open-circuit limit ,  Current unspecified " if self.data.config.electrical.spray_current_na is None else f" Specified  I={abs(ionic['current_a'])*1e9:g} nA"
            self.text((body+8,cy+43),f" Emitter  {ionic['lip_potential_v']/1000:.3f} kV · {note}",11,ELECTRIC)

    def _material_layer(self,mapping):
        key=("material",mapping.rect,mapping.zmin,mapping.zmax,mapping.rmax)
        if key in self.cache:return self.cache[key]
        x0,y0,x1,y1=mapping.pixels();w=max(2,x1-x0);h=max(2,y1-y0)
        xx=np.arange(w)+x0+.5;yy=np.arange(h)+y0+.5
        Z=np.broadcast_to((xx-mapping.left)/mapping.sx+mapping.zmin,(h,w))
        R=np.broadcast_to(abs((mapping.center-yy[:,None])/mapping.sy),(h,w))
        g=self.data.config.geometry;domains=classify(R,Z,g)
        pixels=np.zeros((h,w,4),dtype=np.uint8)
        liquid=(domains==Domain.SAMPLE)|(domains==Domain.SHEATH)
        C=np.clip(self.data.sample(R,Z,"methanol_fraction"),0,1)
        colors=np.array(WATER)[None,None,:]*(1-C[:,:,None])+np.array(METHANOL)[None,None,:]*C[:,:,None]
        shade=.74+.19*np.cos(R/g.c*np.pi/2)
        pixels[:,:,:3]=np.clip(colors*shade[:,:,None],0,255).astype(np.uint8)
        pixels[:,:,3]=np.where(liquid,225,0)
        glass=domains==Domain.QUARTZ
        reflection=.16+.12*np.cos(R*1e6*.15)
        gc=np.clip(np.array([151,188,210])[None,None,:]*reflection[:,:,None]+np.array(PANEL),0,255)
        pixels[glass,:3]=gc[glass].astype(np.uint8);pixels[glass,3]=255
        gas=(R>g.d)&(Z<-g.gas_recess_mm*1e-3)&(R<g.gas_id_um*.5e-6)
        pixels[gas,:3]=blend(PANEL,GAS,.23);pixels[gas,3]=255
        layer=Image.fromarray(pixels,"RGBA")
        self.cache[key]=(layer,(x0,y0));return layer,(x0,y0)

    def _walls(self,mapping):
        g=self.data.config.geometry
        for inner,outer,tip,color in [(g.a,g.b,g.inner_tip,(124,165,192)),
            (g.c,g.d,0.,(124,165,192)),(g.gas_id_um*.5e-6,g.gas_od_um*.5e-6,-g.gas_recess_mm*1e-3,(101,145,171))]:
            for sign in [-1,1]:
                for radius in [inner,outer]:
                    self.draw.line([mapping.point(mapping.zmin,sign*radius),mapping.point(tip,sign*radius)],fill=color,width=1)
                self.draw.line([mapping.point(tip,sign*inner),mapping.point(tip,sign*outer)],fill=color,width=2)

    def _field(self,mapping):
        if not self.field_lines:return
        for line in self.field_lines:
            visible=[mapping.point(z,r) for z,r in line if mapping.zmin<=z<=mapping.zmax and abs(r)<=mapping.rmax]
            if len(visible)<3:continue
            self.draw.line(visible,fill=blend(PANEL,ELECTRIC,.57),width=1)
            for k in [int(len(visible)*.35),int(len(visible)*.75)]:
                if k+3<len(visible):
                    a,b=visible[k],visible[k+3]
                    if self.data.config.electrical.voltage_kv<0:a,b=b,a
                    self.arrow(a,b,ELECTRIC,1,5)

    def _device(self,progress,selected,phase,show_field,show_cells,show_streams):
        W,H=self.width,self.height;data=self.data;g=data.config.geometry
        current=data.snapshot(selected,data.progress_time(selected,progress))
        self._title(" Device and cell transport "," Cutaway view  ·  Blue: aqueous sample  /  Orange: methanol  ·  Select a cell to track "," Continuous feed  ·  Computed fields and trajectories ")
        self._overview((20,76,W-20,213),current)
        main=(20,228,W-286,H-112);side=(W-272,228,W-20,H-112)
        self.panel(main);self.panel(side)
        self.text((38,242)," Emitter section detail ",17,bold=True)
        self.text((38,267),f" Sample capillary  ID/OD {g.inner_id_um:g}/{g.inner_od_um:g} μm  ·   Sheath capillary  {g.sheath_id_um:g}/{g.sheath_od_um:g} μm  ·   Gas capillary  ID {g.gas_id_um:g} μm",12,MUTED)
        self.text((38,287),f" Sample recess  {g.inner_recess_mm:g} mm,  Gas recess  {g.gas_recess_mm:g} mm;  Transverse magnification ,  Independent axis scales .",11,MUTED)
        zmin=min(-1.8e-3,g.inner_tip-.45e-3,-g.gas_recess_mm*1e-3-.35e-3)
        mapping=AxisMap((42,307,W-314,H-175),zmin,g.ms_gap_mm*1e-3+.12e-3,g.gas_od_um*.5e-6*1.08)
        layer,position=self._material_layer(mapping);self.image.paste(layer,position,layer)
        if data.spray is not None:
            frame=round(progress*(len(data.spray["time_s"])-1))
            self._phase_layer(mapping,frame,external_only=True)
        self._walls(mapping)
        self.dashed(mapping.point(zmin,0),mapping.point(g.ms_gap_mm*1e-3,0),blend(PANEL,WHITE,.23))
        if show_streams:
            for k,line in enumerate(self.streamlines):
                visible=[mapping.point(z,r) for z,r in line if z>=zmin]
                if len(visible)>3:
                    self.draw.line(visible,fill=blend(PANEL,WHITE,.27),width=1)
                    at=int((phase*.8+k*.13)%1*(len(visible)-4))
                    self.arrow(visible[at],visible[at+3],blend(PANEL,WHITE,.73),1,4)
        if show_field:self._field(mapping)
        if show_streams and data.spray is not None:self._gas_field(mapping,frame)
        if show_cells:
            records, ledger=data.continuous_cells(phase)
            for record in records:
                index=record["index"];p=record["snapshot"]
                if zmin<=p.z_m<=0:
                    sign=-1 if record["event_id"]%2 else 1;x,y=mapping.point(p.z_m,p.r_m*sign)
                    radius=max(3,min(6,data.config.cell.diameter_um*.5e-6*mapping.sy*3))
                    self.cell(x,y,radius,p.methanol)
                    self.hits.append(dict(kind="cell",index=index,time_s=p.time_s,x=x,y=y,radius=10))
        if current.z_m>=zmin:
            x,y=mapping.point(min(current.z_m,0),current.r_m)
            self.cell(x,y,8,current.methanol,True)
            self.text((x+16,y-20),f"#{data.ids[selected]}",12,CELL)
            self.hits.append(dict(kind="cell",index=selected,time_s=current.time_s,x=x,y=y,radius=14))
        # End positions and explicit physical gap rulers.
        labels=[(g.inner_tip," Sample outlet ",WATER),(-g.gas_recess_mm*1e-3," Gas outlet ",GAS),(0.," Sheath outlet ",METHANOL)]
        for index,(z,label,color) in enumerate(labels):
            x,_=mapping.point(z)
            yy=main[3]-48+(index%2)*18
            self.text((x,yy),label,12,color,anchor="mm")
        self._dimension(mapping,g.inner_tip,0,main[3]-17,f"{g.inner_recess_mm:g} mm")
        self._dimension(mapping,0,g.ms_gap_mm*1e-3,main[3]-17,f" To  MS: {g.ms_gap_mm:g} mm")
        mx,my=mapping.point(g.ms_gap_mm*1e-3,0)
        self.draw.rounded_rectangle((mx-4,my-48,mx+20,my+48),5,fill=(29,51,71),outline=(94,127,150))
        self.draw.ellipse((mx-10,my-16,mx+2,my+16),fill=BG,outline=GAS,width=2)
        self.text((mx-5,my-64),"MS · 0 V",12,WHITE,anchor="mm")
        self.wrapped((mapping.point(.35e-3)[0],mapping.center+20)," Liquid interface evolution ;  Outlet events recorded for downstream continuation ",max(100,mapping.sx*1.25e-3),11,MUTED)
        self._cell_inspector(side,selected,current)
        self.panel((20,H-99,W-20,H-14))
        self._legend(38,H-82)
        self.text((38,H-53)," Aqueous phase  140 mM  Ammonium formate . Colour: computed methanol fraction ;  Cell colour: solvent exposure .",12,MUTED)
        _,ledger=data.continuous_cells(phase)
        self.text((38,H-32),f" Continuous source  t={phase:.2f} s · {ledger['rate_cells_s']:.2f} cells/s ·  Outlet events in window  {ledger['downstream_window']} ·  Green: selected response ",12,MUTED)

    def _dimension(self,mapping,z0,z1,y,label):
        a=mapping.point(z0)[0];b=mapping.point(z1)[0]
        self.draw.line((a,y,b,y),fill=MUTED,width=1)
        for x in [a,b]:self.draw.line((x,y-4,x,y+4),fill=MUTED)
        self.text(((a+b)/2,y-10),label,11,MUTED,anchor="mm")

    def _legend(self,x,y):
        for label,color in [(" Aqueous phase ",WATER),(" Methanol ",METHANOL),("N2  Flow direction ",GAS),(" Electric field ",ELECTRIC),(" Cell ",CELL)]:
            self.draw.ellipse((x,y+3,x+9,y+12),fill=color)
            self.text((x+15,y),label,12,color);x+=self.draw.textlength(label,font=font(12))+40

    def _cell_inspector(self,rect,index,current):
        x0,y0,x1,y1=rect;cx=(x0+x1)/2;h=y1-y0
        self.text((x0+16,y0+14),f" Selected cell  #{self.data.ids[index]}",16,bold=True)
        self.cell(cx,y0+77,26,current.methanol,True)
        self.text((cx,y0+117),f" Diameter  {self.data.config.cell.diameter_um:g} μm ·  Rigid sphere ",12,MUTED,anchor="mm")
        pairs=[(" Elapsed transport ",f"{current.time_s:.3f} s"),(" Local methanol ",f"{current.methanol:.1%}"),
               (" Integrated exposure  ∫Cdt",f"{current.dose_fraction_s*1000:.2f} ms"),
               (" Membrane state "," Not computed ")]
        for k,(label,value) in enumerate(pairs):
            yy=y0+141+k*24
            self.text((x0+16,yy),label,12,MUTED)
            self.text((x1-16,yy),value,13,METHANOL if k==1 else WHITE,anchor="ra")
        track=self.data.track(index)
        if h>300:
            self.text((x0+16,y0+246)," Tip methanol exposure ",12,MUTED)
            self._exposure_chart((x0+20,y0+274,x1-20,y1-22),track,current.time_s)

    def _exposure_chart(self,rect,track,time_s):
        x0,y0,x1,y1=rect
        if y1-y0<18:return
        t=track["t"];mask=t>=track["bulk_time"]
        ts=t[mask];C=track["C"][mask]
        if len(ts)<2:return
        span=max(ts[-1]-ts[0],1e-9)
        self.draw.line((x0,y0,x0,y1,x1,y1),fill=BORDER,width=1)
        points=list(zip(x0+(ts-ts[0])/span*(x1-x0),y1-C*(y1-y0)))
        self.draw.line(points,fill=METHANOL,width=2)
        if time_s>=ts[0]:
            x=x0+np.clip((time_s-ts[0])/span,0,1)*(x1-x0)
            self.draw.line((x,y0,x,y1),fill=CELL,width=1)
        self.text((x0,y0-14),"100%",10,MUTED)
        self.text((x1,y1+3),f"{span*1000:.0f} ms",10,MUTED,anchor="ra")

    def _nozzle(self,progress,selected,phase,show_field,explanation,settings):
        W,H=self.width,self.height;data=self.data;g=data.config.geometry
        badge=" Reference models  ·  Reference model ,  Analytical reference " if explanation else " Computed snapshot  ·  Resolved interface "
        self._title(" Emitter ,  Droplets and membrane response "," Local zoom  ·  Interface forces ,  Detached droplets and solvent exposure ",badge,AMBER if explanation else GAS)
        self.panel((20,78,W-20,123),fill=(43,33,22) if explanation else PANEL)
        if explanation:
            self.wrapped((36,85)," Reference cone ,  Breakup and membrane response models . Spray morphology ,  Specified droplet and lysis parameters .",W-72,13,AMBER)
        else:
            self.wrapped((36,85)," Interface from the two-phase solution ;  If no detached component is resolved ,  Count  0,  Resolved components only .",W-72,13,MUTED)
        main=(20,138,W-286,H-151);side=(W-272,138,W-20,H-151)
        self.panel(main);self.panel(side)
        heading=" Ideal cone reference  49.3° /  Jet  /  Droplets " if explanation and H<680 else "Taylor  Cone  /  Jet  /  Droplets "
        self.text((38,152),heading,17,bold=True)
        self.text((38,179),f"MS  Downstream of the sheath outlet  {g.ms_gap_mm:g} mm;  Emitter detail .",12,MUTED)
        sp=data.spray
        length=min(sp["axial_edges_m"][-1],data.config.spray.length_mm*1e-3) if sp is not None else data.config.spray.length_mm*1e-3
        length=max(length,.45e-3)
        mapping=AxisMap((40,212,W-310,H-202),-.16e-3,length,g.c*1.6,isotropic=True)
        self.dashed(mapping.point(mapping.zmin,0),mapping.point(mapping.zmax,0),blend(PANEL,WHITE,.23))
        if show_field:self._field(mapping)
        if explanation:
            droplets=self._illustrative_spray(mapping,progress,phase,selected,settings)
            frame=None
        elif sp is None:
            self.text(((main[0]+main[2])/2,(main[1]+main[3])/2)," No two-phase solution \n Load a result containing  spray.npz  Data ",16,MUTED,anchor="mm")
            droplets=[];frame=None
        else:
            frame=int(np.clip(round(progress*(len(sp["time_s"])-1)),0,len(sp["time_s"])-1))
            self._phase_layer(mapping,frame)
            droplets=[drop for drop in data.droplet_frame(frame) if drop["resolved"]]
            self._phase_contour(mapping,frame)
            self.pill(main[2]-170,152,f"t = {sp['time_s'][frame]*1000:.4f} ms",GAS)
        self._nozzle_walls(mapping)
        # Arrows explain the force directions; they are not a measured force scale.
        if show_field and data.config.electrical.voltage_kv!=0:
            x0,y0=mapping.point(.035e-3,g.c*.88)
            self.arrow((x0,y0),(x0+28,y0-28),ELECTRIC,2)
            self.text((x0+34,y0-42)," Outward electric stress ",11,ELECTRIC)
        tx,ty=mapping.point(.055e-3,-g.c*.72)
        self.arrow((tx,ty),(tx-17,ty-25),WATER,2)
        self.text((tx+18,ty+10)," Surface-tension recoil ",11,WATER)
        if not explanation and sp is not None:
            self._gas_field(mapping,frame)
        elif data.config.fluid.gas_pressure_kpa>0:
            for sign in [-1,1]:
                a=mapping.point(-.10e-3,sign*g.c*1.42);b=mapping.point(.035e-3,sign*g.c*1.42)
                self.arrow(a,b,GAS,2)
        self.text((42,main[3]-37)," Force directions ;  Schematic vector lengths .",11,MUTED)
        self.text((main[2]-16,main[3]-37)," Emitter detail  ·  Equal axis scales ",11,MUTED,anchor="ra")
        self._spray_inspector(side,droplets,frame,selected,progress,explanation,settings)
        self._cell_stages((20,H-137,W-20,H-14),selected,progress,explanation,settings)

    def _nozzle_walls(self,mapping):
        g=self.data.config.geometry
        x0=mapping.point(mapping.zmin)[0];x1=mapping.point(0)[0]
        for sign in [-1,1]:
            ya=mapping.point(0,sign*g.c)[1]
            yb=mapping.point(0,sign*min(g.d,mapping.rmax))[1]
            ylo,yhi=sorted([ya,yb])
            self.draw.rectangle((x0,ylo,x1,yhi),fill=(43,65,82),outline=(119,154,178),width=1)
            self.draw.line((x1,ylo,x1,yhi),fill=WHITE,width=2)
        xx,yy=mapping.point(0,-g.c)
        self.text((xx-8,yy+14)," Sheath outlet ",12,METHANOL,anchor="ra")

    def _phase_layer(self,mapping,frame,external_only=False):
        key=("phase",frame,mapping.rect,external_only)
        if key not in self.cache:
            from scipy.interpolate import RegularGridInterpolator
            sp=self.data.spray
            r=(sp["radial_edges_m"][:-1]+sp["radial_edges_m"][1:])/2
            z=(sp["axial_edges_m"][:-1]+sp["axial_edges_m"][1:])/2
            x0,y0,x1,y1=mapping.pixels();w=max(2,x1-x0);h=max(2,y1-y0)
            xx=np.arange(w)+x0+.5;yy=np.arange(h)+y0+.5
            Z=np.broadcast_to((xx-mapping.left)/mapping.sx+mapping.zmin,(h,w))
            R=np.broadcast_to(abs((mapping.center-yy[:,None])/mapping.sy),(h,w))
            interpolator=RegularGridInterpolator((r,z),sp["liquid_fraction"][frame],bounds_error=False,fill_value=None)
            phi=np.clip(interpolator(np.c_[np.clip(R.ravel(),r[0],r[-1]),np.clip(Z.ravel(),z[0],z[-1])]).reshape(h,w),0,1)
            phi[(R>r[-1])|(Z>sp["axial_edges_m"][-1])|(Z<sp["axial_edges_m"][0])]=0
            if external_only:phi[Z<0]=0
            rgba=np.zeros((h,w,4),np.uint8)
            shade=.82+.18*np.cos(np.minimum(R/self.data.config.geometry.c,1)*np.pi/2)
            rgba[:,:,:3]=np.clip(np.array([65,179,227])*shade[:,:,None],0,255).astype(np.uint8)
            rgba[:,:,3]=np.where(phi>=.5,255,np.clip(phi*230,0,255)).astype(np.uint8)
            self.cache[key]=(Image.fromarray(rgba,"RGBA"),(x0,y0))
            # Bound memory while playing hundreds of real solver frames.
            phase_keys=[k for k in self.cache if k[0]=="phase"]
            if len(phase_keys)>12:del self.cache[phase_keys[0]]
        layer,position=self.cache[key];self.image.paste(layer,position,layer)

    def _gas_field(self,mapping,frame):
        """Actual transient gas velocity arrows, with a stated speed scale."""
        sp=self.data.spray
        if sp is None:return
        ur_name="velocity_r_history_m_s" if "velocity_r_history_m_s" in sp else "velocity_r_m_s"
        uz_name="velocity_z_history_m_s" if "velocity_z_history_m_s" in sp else "velocity_z_m_s"
        rs=np.linspace(-mapping.rmax*.92,mapping.rmax*.92,9)
        zs=np.linspace(max(mapping.zmin,sp["axial_edges_m"][0]),min(mapping.zmax,sp["axial_edges_m"][-1]),17)
        R,Z=np.meshgrid(rs,zs,indexing="ij")
        ur=self.data.spray_sample(R,Z,ur_name,frame);uz=self.data.spray_sample(R,Z,uz_name,frame)
        phi=self.data.spray_sample(R,Z,"liquid_fraction",frame)
        gas=(phi<.1)&np.isfinite(ur)&np.isfinite(uz)
        if "active" in sp:gas &= self.data.spray_sample(R,Z,"active",frame)>.9
        speed=np.hypot(ur,uz);max_speed=float(np.nanmax(speed[gas])) if np.any(gas) else 0.
        if max_speed<=1e-12:return
        for rr,zz,vr,vz,ok in zip(R.ravel(),Z.ravel(),ur.ravel(),uz.ravel(),gas.ravel()):
            if not ok:continue
            x,y=mapping.point(zz,rr)
            self.arrow((x,y),(x+25*vz/max_speed,y-25*vr/max_speed),GAS,1,4)
        self.text((mapping.left+6,mapping.top+4),f" Gas velocity : 25 px = {max_speed:.2g} m/s",10,GAS)

    def _phase_contour(self,mapping,frame):
        import contourpy
        sp=self.data.spray
        r=(sp["radial_edges_m"][:-1]+sp["radial_edges_m"][1:])/2
        z=(sp["axial_edges_m"][:-1]+sp["axial_edges_m"][1:])/2
        contours=contourpy.contour_generator(x=z,y=r,z=sp["liquid_fraction"][frame]).lines(.5)
        for contour in contours:
            for sign in [-1,1]:
                pts=[mapping.point(zz,rr*sign) for zz,rr in contour if mapping.zmin<=zz<=mapping.zmax and rr<=mapping.rmax]
                if len(pts)>1:self.draw.line(pts,fill=(131,224,255),width=2)

    def _illustrative_spray(self,mapping,progress,phase,selected,settings):
        settings.validate();g=self.data.config.geometry
        reference=rayleigh_reference(settings.jet_diameter_um,self.data.config.fluid.qs+self.data.config.fluid.qh)
        jet=reference["jet_radius_m"]
        development=float(np.clip(progress/.4,0,1));development=development**2*(3-2*development)
        cone_length=max((g.c-jet)/math.tan(math.radians(49.3)),10e-6)
        front=(1-development)*g.c+development*cone_length
        s=np.linspace(0,1,55)
        z=s*front
        radius=(1-development)*g.c*np.sqrt(np.maximum(1-s*s,0))+development*(g.c*(1-s)+jet*s)
        body=[mapping.point(mapping.zmin,g.c)]+[mapping.point(zz,rr) for zz,rr in zip(z,radius)]
        jet_fraction=np.clip((progress-.30)/.28,0,1)
        jet_end=front+jet_fraction*.18e-3
        if jet_fraction>0:
            jz=np.linspace(front,jet_end,150)
            amplitude=np.clip((progress-.4)/.25,0,.95)
            jr=jet*(1+amplitude*np.sin((jz-front)/reference["wavelength_m"]*2*np.pi-phase*8))
            body += [mapping.point(zz,rr) for zz,rr in zip(jz,jr)]
            body += [mapping.point(zz,-rr) for zz,rr in zip(jz[::-1],jr[::-1])]
        body += [mapping.point(zz,-rr) for zz,rr in zip(z[::-1],radius[::-1])]
        body += [mapping.point(mapping.zmin,-g.c)]
        self._liquid_polygon(body,mapping)
        if development>.8 and self.height>=680:
            px,py=mapping.point(cone_length*.5,g.c*.5)
            xx,yy=mapping.point(front,0.);xx+=58;yy-=145
            self.draw.rounded_rectangle((xx-8,yy-6,xx+187,yy+42),7,fill=PANEL)
            self.text((xx,yy)," Ideal  Taylor  Reference cone ",13,(133,224,255))
            self.text((xx,yy+21),"49.3°  Half-angle  ( Ideal limit )",11,MUTED)
            self.draw.line((xx,yy+43,px+7,py-7),fill=(94,171,200),width=1)
        for sign in [-1,1]:
            for fraction in [.25,.55,.8]:
                xx,yy=mapping.point(front*fraction,sign*(g.c*(1-fraction)+jet*fraction))
                self.text((xx,yy-sign*10),"+" if self.data.config.electrical.voltage_kv>=0 else "−",14,ELECTRIC,anchor="mm")
        droplets=[]
        if progress>.57:
            rng=np.random.default_rng(125)
            count=24
            radii=reference["drop_radius_m"]*np.exp(rng.normal(0,settings.log_radius_spread,count))
            offsets=rng.uniform(-1,1,count)
            for j,rr in enumerate(radii):
                age=((j/count+phase*.13)%1)
                zz=jet_end+.024e-3+age*max(.04e-3,mapping.zmax-jet_end-.04e-3)
                radial=offsets[j]*age*.085e-3
                xx,yy=mapping.point(zz,radial)
                rx=max(2,rr*mapping.sx);ry=max(2,rr*mapping.sy)
                self.draw.ellipse((xx-rx,yy-ry,xx+rx,yy+ry),fill=(34,101,145),outline=(122,220,250),width=1)
                self.draw.arc((xx-rx*.8,yy-ry*.8,xx+rx*.8,yy+ry*.8),190,290,fill=(194,240,255),width=1)
                droplets.append({"equivalent_radius_um":rr*1e6,"cells":0,"illustrative":True})
                if j==5:
                    integrity,remaining,released=self.data.explanation_response(selected,progress,settings)
                    self.cell(xx,yy,max(3,min(rx*.55,6)),self.data.track(selected)["C"][-1],True,integrity,released*.25)
                    self.text((xx+12,yy+12)," Highlighted :  Reference cell-bearing droplet ",10,CELL)
        return droplets

    def _liquid_polygon(self,points,mapping):
        mask=Image.new("L",self.image.size);md=ImageDraw.Draw(mask);md.polygon(points,fill=255)
        # A lit cylindrical appearance; shape itself is only the labelled reference.
        h=self.height
        y=np.arange(h)
        light=.55+.45*np.exp(-((y-mapping.center)/max(mapping.sy*self.data.config.geometry.c*.7,1))**2)
        line=np.clip(np.array([53,178,229])[None,:]*light[:,None],0,255).astype(np.uint8)
        strip=Image.fromarray(line[:,None,:],"RGB").resize(self.image.size)
        self.image.paste(strip,(0,0),mask)
        self.draw.line(points+[points[0]],fill=(140,230,255),width=2)

    def _spray_inspector(self,rect,droplets,frame,selected,progress,explanation,settings):
        x0,y0,x1,y1=rect;data=self.data
        self.text((x0+16,y0+14)," Droplets and interface ",16,bold=True)
        title=" Reference droplets " if explanation else " Resolved detached droplets "
        self.text((x0+16,y0+50),title,12,MUTED)
        self.text((x0+16,y0+72),str(len(droplets)),38,AMBER if explanation else GAS,bold=True)
        if explanation:
            ref=rayleigh_reference(settings.jet_diameter_um,data.config.fluid.qs+data.config.fluid.qh)
            self.text((x0+16,y0+126),f" Reference droplet diameter  {ref['drop_radius_m']*2e6:.2f} μm",13,WHITE)
            self.text((x0+16,y0+151),f" Specified jet diameter  {settings.jet_diameter_um:g} μm",11,AMBER)
        elif frame is not None:
            sp=data.spray
            h=min(np.diff(sp["radial_edges_m"]).min(),np.diff(sp["axial_edges_m"]).min())
            self.text((x0+16,y0+126),f" Minimum grid spacing  {h*1e6:.2f} μm",13,WHITE)
            self.text((x0+16,y0+151)," Sub-grid droplets unresolved ",11,MUTED)
        if y1-y0>260:
            self._histogram((x0+20,y0+192,x1-20,y1-40),droplets,explanation)
        if not explanation and len(droplets)==0:
            self.text((x0+16,y1-28)," No detached droplets in this frame ",12,MUTED)

    def _histogram(self,rect,droplets,explanation):
        x0,y0,x1,y1=rect
        if y1-y0<25:return
        self.draw.line((x0,y0,x0,y1,x1,y1),fill=BORDER,width=1)
        self.text((x0,y0-22)," Reference size distribution " if explanation else " Snapshot size distribution ",12,MUTED)
        if not droplets:
            self.text(((x0+x1)/2,(y0+y1)/2)," No data ",14,MUTED,anchor="mm")
            return
        diameters=np.array([d["equivalent_radius_um"]*2 for d in droplets])
        high=max(float(diameters.max())*1.1,1.)
        counts,edges=np.histogram(diameters,bins=np.linspace(0,high,9))
        maxcount=max(1,counts.max())
        for k,count in enumerate(counts):
            left=x0+(x1-x0)*k/8+2;right=x0+(x1-x0)*(k+1)/8-2
            self.draw.rectangle((left,y1-count/maxcount*(y1-y0),right,y1),fill=METHANOL if explanation else WATER)
        self.text((x0,y1+5),"0",10,MUTED)
        self.text((x1,y1+5),f"{high:.1f} μm",10,MUTED,anchor="ra")

    def _cell_stages(self,rect,selected,progress,explanation,settings):
        x0,y0,x1,y1=rect;self.panel(rect)
        self.text((x0+18,y0+11)," Cell state ",15,bold=True)
        track=self.data.track(selected);C=float(track["C"][-1])
        labels=[(" Sample inlet "," Aqueous phase  /  Methanol  0%"),(" Sheath outlet arrival ",f" Computed methanol  {C:.1%}")]
        if explanation:
            integrity,remaining,released=self.data.explanation_response(selected,progress,settings)
            labels.append((" Additional exposure ",f" Membrane integrity  {integrity:.0%} /  Released fraction  {released:.0%}"))
        else:
            integrity=None;remaining=1.;released=0.
            labels.append((" Post-emitter membrane response "," Membrane state not computed  /  Extraction not computed "))
        width=(x1-x0-185)/3
        for j,(title,subtitle) in enumerate(labels):
            left=x0+175+j*width;cx=left+29;cy=y0+65
            self.cell(cx,cy,20,0. if j==0 else C,False,integrity if j==2 and explanation else None,released if j==2 and explanation else 0.)
            self.text((cx+37,cy-20),title,13,WHITE)
            lines=subtitle.split(" / ") if j==2 else [subtitle]
            for row,line in enumerate(lines):self.text((cx+37,cy+row*16),line,11,AMBER if j==2 and explanation else MUTED)
        footer=(f" Specified constitutive model : k Lysis ={settings.lysis_rate_s:g}/s, k Extraction ={settings.extraction_rate_s:g}/s;  Post-outlet exposure  {settings.extra_exposure_ms:g} ms ( Reference morphology )." if explanation else " Rigid-sphere transport model ;  Colour: solvent exposure ,  See membrane model in Methods .")
        self.text((x0+18,y1-23),footer,11,AMBER if explanation else MUTED)
