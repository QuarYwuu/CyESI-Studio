"""Live drawings consume the shared physical clock and saved fields."""
import tkinter as tk
import numpy as np
from PIL import Image,ImageTk
from ..gui.trajectory_stream import group_screen_points
from ..visualization.process_data import CELL_COLORS,STATUS_COLORS,STATUS_LABELS
from .widgets import WHITE,INK,MUTED,TEAL,BLUE,PURPLE,PALE,LINE,FONT


def txt(c,x,y,text,size=10,color=INK,anchor='w',**kw):
    return c.create_text(x,y,text=text,font=(FONT,size),fill=color,anchor=anchor,**kw)


def stream_problem(session):
    if not getattr(session,'views_ready',True):return '连续流与采集统计在后台更新; 完成后自动显示.'
    return getattr(session,'process_error',None)


class DeviceCanvas(tk.Canvas):
    def __init__(self,parent,session):
        super().__init__(parent,bg=WHITE,highlightthickness=0,height=160)
        self.session=session;self.bind('<Configure>',lambda e:self.draw())
    def draw(self):
        self.delete('all');s=self.session;data=s.data;c=data.config if data else s.draft
        w=max(500,self.winfo_width());cy=75
        x=np.linspace(65,w-70,4)
        names=[' Cell suspension ','长内管运输','同轴聚焦 /  Emitter ','Q Exactive  Inlet ']
        colors=[BLUE,BLUE,PURPLE,TEAL]
        for a,b in zip(x[:-1],x[1:]):self.create_line(a+32,cy,b-33,cy,fill='#c4d3e1',width=3,arrow='last')
        for i,(xx,name,color) in enumerate(zip(x,names,colors)):
            self.create_oval(xx-29,cy-29,xx+29,cy+29,fill=PALE,outline=color,width=2)
            txt(self,xx,cy,f'0{i+1}',size=18,color=color,anchor='center')
            txt(self,xx,cy-50,name,size=10,anchor='center')
        txt(self,x[0],cy+49,f'{c.cell.concentration_per_ml:g} cells/mL',size=9,color=MUTED,anchor='center')
        txt(self,x[1],cy+49,f'{c.geometry.sample_length_mm/10:g} cm · {c.supply.orientation}',size=9,color=MUTED,anchor='center')
        txt(self,x[2],cy+49,f' Sheath  {c.fluid.sheath_flow_ul_min:g} μL/min · {c.electrical.voltage_kv:g} kV',size=9,color=MUTED,anchor='center')
        txt(self,x[3],cy+49,f'距喷口 {c.geometry.ms_gap_mm:g} mm',size=9,color=MUTED,anchor='center')
        txt(self,18,148,'装置流程总览; 各区域按长度压缩展示.入口条件与膜状态在后续页面查看.',size=9,color=MUTED)


class StreamCanvas(tk.Canvas):
    def __init__(self,parent,session,on_pick):
        super().__init__(parent,bg=WHITE,highlightthickness=0,height=440)
        self.session=session;self.on_pick=on_pick;self.hits=[];self.images={};self.photos={}
        self.bind('<Configure>',lambda e:self.draw());self.bind('<Button-1>',self.pick)
    def pick(self,e):
        x=self.canvasx(e.x);y=self.canvasy(e.y)
        if self.hits:
            hit=min(self.hits,key=lambda p:(p[0]-x)**2+(p[1]-y)**2)
            if (hit[0]-x)**2+(hit[1]-y)**2<120:self.on_pick(hit[2])
    def bodies(self,points,transform):
        drawn=points[::max(1,int(np.ceil(len(points)/400)))]
        groups=group_screen_points(drawn,transform,tolerance=6)
        for group in groups:
            selected=next((item for item in group if item[0]['event_id']==self.session.selected_event),None)
            p,x,y=selected or group[0];radius=6 if selected else 3.8
            status=p.get('interaction_status');bound=status in ('cell_bound','wall_bound','outlet_bound','bulk_wall_bound')
            color='#dc4c91' if status in ('wall_bound','outlet_bound','bulk_wall_bound') else '#df9a31' if status=='cell_bound' else CELL_COLORS[p['event_id']%len(CELL_COLORS)]
            self.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,
                outline=TEAL if selected else WHITE,width=2 if selected else 1)
            if selected or bound:txt(self,x+9,y-12,f'#{p["event_id"]}'+('  Binding ' if bound else ''),size=9,color=TEAL if selected else '#c43d84')
            elif len(group)>=12:txt(self,x+5,y-12,f'×{len(group)}',size=8,color=MUTED)
            self.hits.extend((px,py,item['event_id']) for item,px,py in group)
        return len(drawn),len(groups)
    def background(self,data,zmin,rect,key):
        left,top,right,bottom=map(int,rect);width=max(1,right-left);height=max(1,bottom-top)
        token=(id(data.fields),zmin,width,height)
        if key not in self.images or self.images[key][0]!=token:
            z=np.linspace(zmin,0,width);r=np.linspace(data.config.geometry.c,0,height)
            R,Z=np.meshgrid(r,z,indexing='ij');C=np.clip(data.sample(R,Z,'methanol_fraction'),0,1)
            a=np.array([224,239,252]);b=np.array([240,227,253])
            colors=np.uint8(a[None,None,:]*(1-C[:,:,None])+b[None,None,:]*C[:,:,None])
            self.images[key]=(token,Image.fromarray(colors))
            self.photos[key]=ImageTk.PhotoImage(self.images[key][1])
        self.create_image(left,top,image=self.photos[key],anchor='nw')
    def draw(self):
        self.delete('all');self.hits=[];s=self.session
        if s.data is None:txt(self,30,60,'先载入已有结果, 或在实验设置后运行一次计算.',color=MUTED);return
        problem=stream_problem(s)
        if problem:txt(self,25,60,problem,color=MUTED,width=600);return
        try:m=s.process();positions=s.positions()
        except (ValueError,KeyError,TypeError) as e:txt(self,25,60,str(e),color=MUTED,width=600);return
        g=s.data.config.geometry;w=max(650,self.winfo_width());h=max(390,self.winfo_height())
        self.configure(scrollregion=(0,0,w,h))
        bulk=[p for p in positions if p['z_m']<g.liquid_start]
        tip=[p for p in positions if g.liquid_start<=p['z_m']<0]
        zfocus=-max(1.5*g.inner_recess_mm,.5)*1e-3;zoom=[p for p in tip if p['z_m']>=zfocus]
        left=55;right=w-30;top=60;bottom=135;mid=(top+bottom)/2
        txt(self,20,23,f'长管前段 {g.bulk_length*1e3:g} mm',size=11,color=INK)
        txt(self,right,23,f' Current  {len(bulk)}  Cells ',size=10,color=TEAL,anchor='e')
        self.create_rectangle(left,top-5,right,bottom+5,fill='#dbe5ef',outline='')
        self.create_rectangle(left,top,right,bottom,fill='#edf6fd',outline='')
        transform=lambda p:(left+(p['z_m']-m.start_z)/max(g.bulk_length,1e-12)*(right-left),mid-p['y_m']/g.a*(bottom-top)/2)
        n,marks=self.bodies(bulk,transform)
        txt(self,left,155,' Inlet :  Continuous injection ',size=9,color=MUTED)
        txt(self,right,155,f'末端入口 →    {n} 个绘制细胞 / {marks} 个显示位置',size=9,color=MUTED,anchor='e')
        y0=210;y1=h-85
        for slot,zmin,name,points in [(0,g.liquid_start,' Tip overview ',tip),(1,zfocus,' Sample outlet  /  Focusing detail ',zoom)]:
            l=left+slot*(w-left-20)/2;r=l+(w-left-20)/2-35
            xx=lambda z:l+(z-zmin)/(0-zmin)*(r-l)
            yy=lambda radial:y1-radial/g.c*(y1-y0)
            txt(self,l,y0-25,f'{name} · {len(points)}  Count ',size=10,color=TEAL)
            self.background(s.data,zmin,(l,y0,r,y1),slot)
            self.create_rectangle(l,y0,r,y1,outline=LINE)
            if zmin<g.inner_tip:
                self.create_rectangle(l,yy(g.b),xx(g.inner_tip),yy(g.a),fill='#d2dce5',outline='')
                txt(self,(l+xx(g.inner_tip))/2,yy((g.a+g.b)/2),' Quartz ',size=9,color=MUTED,anchor='center')
            self.create_line(xx(g.inner_tip),y0,xx(g.inner_tip),y1,fill='#95aabc',dash=(3,3))
            self.create_line(r,y0,r,y1,fill=TEAL,width=2)
            self.bodies(points,lambda p:(xx(p['z_m']),yy(p['r_m'])))
            txt(self,l,y1+18,f'z: {zmin*1e3:.2f} → 0 mm',size=9,color=MUTED)
            txt(self,l-28,y0,f'{g.c*1e6:g}',size=8,color=MUTED)
            txt(self,l-28,y1,'0',size=8,color=MUTED)
        txt(self,20,h-38,'背景: 已求解外液甲醇分数,  Blue: aqueous sample  → 淡紫甲醇; 圆点放大, 仅合并像素内重合.',size=9,color=MUTED,width=w-40)
        if not m.has_bulk:txt(self,20,h-15,'此旧结果缺少长管空间历史: 长管段为端点插值, 需重算.',size=9,color='#b8771c')


class GateCanvas(tk.Canvas):
    def __init__(self,parent,session):
        super().__init__(parent,bg=WHITE,highlightthickness=0,height=160)
        self.session=session;self.cached_model=None;self.scan_t=None;self.bind('<Configure>',lambda e:self.draw())
    def draw(self):
        self.delete('all');s=self.session
        if s.data is None:txt(self,20,70,'载入结果后显示同一事件流的采集门.',color=MUTED);return
        problem=stream_problem(s)
        if problem:txt(self,20,65,problem,color=MUTED,width=600);return
        try:m=s.process()
        except (ValueError,KeyError,TypeError) as error:txt(self,20,65,str(error),color=MUTED);return
        if self.cached_model is not m:
            self.cached_model=m;self.scan_t=np.array([row['start_s'] for row in m.scans])
        w=max(620,self.winfo_width());h=max(155,self.winfo_height());left=90;right=w-28
        span=max(2.,m.row['record_cycle_ms']*.006);a=max(0.,s.now-span*.35);b=a+span
        sx=lambda value:left+(value-a)/span*(right-left)
        txt(self,15,54,'IT 门',size=9,color=MUTED);txt(self,15,116,'离子包',size=9,color=MUTED)
        self.create_rectangle(left,38,right,75,fill=PALE,outline='')
        self.create_rectangle(left,90,right,132,fill=PALE,outline='')
        lo=max(0,np.searchsorted(self.scan_t,a)-1);hi=np.searchsorted(self.scan_t,b,side='right')
        for scan in m.scans[lo:hi]:
            color='#7ac8bd' if len(scan['event_ids'])==1 else '#e1b569' if len(scan['event_ids'])>1 else '#dae3ed'
            for start,end in scan['gates']:
                if end>=a and start<=b:self.create_rectangle(sx(max(a,start)),42,sx(min(b,end)),71,fill=color,outline='')
        width=s.acquisition.signal_width_ms*.001
        for row in m.ledger:
            start=row['packet_onset_s'];end=start+width
            if end<a or start>b:continue
            y=104 if row['event_id']==s.selected_event else 123
            self.create_line(sx(max(start,a)),y,sx(min(end,b)),y,fill=STATUS_COLORS[row['status']],width=7)
            self.create_oval(sx(start)-3,y-3,sx(start)+3,y+3,fill=STATUS_COLORS[row['status']],outline='')
            txt(self,sx(start)+5,y-14,f'#{row["event_id"]}',size=8,color=MUTED)
        if a<=s.now<=b:self.create_line(sx(s.now),28,sx(s.now),136,fill=TEAL,width=2)
        for t in np.linspace(a,b,7):txt(self,sx(t),h-10,f'{t:.2f} s',size=8,color=MUTED,anchor='center')
        txt(self,left,16,'绿: 单来源门   黄: 混合门   灰: 空门; 与运输页面使用同一编号和时钟.',size=9,color=MUTED)
