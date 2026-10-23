"""Continuous injection at the FULL capillary entrance, on one physical clock.

No extra cell source is introduced at the resolved-tip entrance. Positions
follow injection_time + the saved long-tube/tip response, keeping event IDs.
"""
from copy import deepcopy
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
from ..visualization.process_data import ProcessModel,ProcessParameters,CELL_COLORS
from ..acquisition.q_exactive import AcquisitionParameters


def window_residence(track,left,right):
    """Time inside an axial interval along the stored piecewise-linear path."""
    total=0.
    for a,b,x,y in zip(track['t'][:-1],track['t'][1:],track['z'][:-1],track['z'][1:]):
        if b<=a:continue
        if abs(y-x)<1e-30:
            if left<=x<=right:total+=b-a
        else:
            low,high=sorted(((left-x)/(y-x),(right-x)/(y-x)))
            total+=(b-a)*max(0.,min(1.,high)-max(0.,low))
    return float(total)


def group_screen_points(points,transform,tolerance=7.):
    """Aggregate only render collisions, preserving each underlying identity."""
    buckets={}
    for point in points:
        x,y=transform(point);key=(round(x/tolerance),round(y/tolerance))
        buckets.setdefault(key,[]).append((point,x,y))
    return list(buckets.values())


def next_stage_event(model,key,after):
    """Next actual saved-path crossing on the shared clock, including preroll."""
    r=model.records;k=r['template_id']
    ages=np.array([tr[key] for tr in model.tracks])
    stage={'bulk_time':model.data.config.geometry.liquid_start,
           'inner_time':model.data.config.geometry.inner_tip}[key]
    reached=np.array([np.any(tr['z']>=stage-1e-12) for tr in model.tracks])
    times=r['injection_s']+ages[k]
    eligible=np.flatnonzero(reached[k]&np.isfinite(times)&(times>after+1e-8))
    if not len(eligible):return None
    i=eligible[np.argmin(times[eligible])]
    return int(r['event_id'][i]),float(times[i])


class TrajectoryStreamPanel(ttk.Frame):
    def __init__(self,parent,data=None):
        super().__init__(parent,padding=5)
        self.data=None;self.model=None;self.closed=False;self.job=None;self.playing=False
        self.now=0.;self.horizon=600.;self.last_tick=time.perf_counter();self.selected_event=None
        self.pending_config=None;self.hits=[]
        self.startup=tk.StringVar(value='稳态连续进样');self.speed=tk.StringVar(value='1×')
        self.header=tk.StringVar();self.pending=tk.StringVar();self.status=tk.StringVar();self.clock=tk.StringVar()
        ttk.Label(self,text='真实连续进样: 内管入口 →  Long capillary  → 聚焦区',font=('Microsoft YaHei UI',12,'bold')).pack(fill='x',padx=6,pady=(3,2))
        self.wrapped=[]
        for variable,color in ((self.header,'#304b61'),(self.pending,'#a65b21')):
            label=ttk.Label(self,textvariable=variable,wraplength=900,foreground=color)
            label.pack(fill='x',padx=6);self.wrapped.append(label)
        bar=ttk.Frame(self);bar.pack(fill='x',padx=6,pady=4)
        self.play_button=ttk.Button(bar,text='▶  Play ',command=self.toggle);self.play_button.pack(side='left')
        ttk.Button(bar,text=' Start time ',command=lambda:self.seek(0.)).pack(side='left',padx=3)
        mode=ttk.Combobox(bar,textvariable=self.startup,values=['从无细胞开始','稳态连续进样'],state='readonly',width=16);mode.pack(side='left',padx=5)
        mode.bind('<<ComboboxSelected>>',lambda e:self.rebuild())
        ttk.Label(bar,text='物理时间速度').pack(side='left',padx=(5,3))
        ttk.Combobox(bar,textvariable=self.speed,values=['0.05×','0.1×','1×','10×','50×','100×'],state='readonly',width=7).pack(side='left')
        next_bar=ttk.Frame(self);next_bar.pack(fill='x',padx=6)
        ttk.Button(next_bar,text='下一颗进入末端',command=lambda:self.jump('bulk_time')).pack(side='left')
        ttk.Button(next_bar,text='下一颗聚焦 / 慢放',command=self.focus_slow).pack(side='left',padx=5)
        ttk.Label(next_bar,text='同一细胞编号贯穿长管与出口; 点击圆点查看到达时刻.').pack(side='left',padx=4)
        label=ttk.Label(self,textvariable=self.clock,foreground='#24516d',wraplength=900)
        label.pack(fill='x',padx=6,pady=2);self.wrapped.append(label)
        self.slider=ttk.Scale(self,from_=0,to=self.horizon,command=self.slider_changed);self.slider.pack(fill='x',padx=6)
        drawing=ttk.Frame(self);drawing.pack(fill='both',expand=True,pady=4)
        self.canvas=tk.Canvas(drawing,bg='#f6f9fc',highlightthickness=0,height=440)
        scroll=ttk.Scrollbar(drawing,orient='vertical',command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.bind('<MouseWheel>',lambda e:self.canvas.yview_scroll(-int(e.delta/120),'units'))
        self.canvas.bind('<Configure>',lambda e:self.draw());self.canvas.bind('<Button-1>',self.pick)
        label=ttk.Label(self,textvariable=self.status,wraplength=900)
        label.pack(fill='x',padx=6,pady=2);self.wrapped.append(label)
        self.bind('<Configure>',self.resize,add='+')
        self.bind('<Destroy>',lambda e:self.close() if e.widget is self else None,add='+')
        self.set_data(data);self.job=self.after(80,self.tick)

    def set_data(self,data):
        if self.closed:return
        same=(self.data is not None and data is not None and self.data.source==data.source
              and self.data.config.to_dict()==data.config.to_dict() and self.data.rate_config.to_dict()==data.rate_config.to_dict()
              and self.data.fields.get('trajectory_time_s') is data.fields.get('trajectory_time_s'))
        self.data=data
        if not same:self.rebuild()
        if self.pending_config is not None:self.set_pending(self.pending_config)

    def rebuild(self):
        self.playing=False;self.play_button.configure(text='▶  Play ');self.now=0.;self.selected_event=None
        self.model=None;self.status.set('');self.canvas.yview_moveto(0.)
        if self.data is None:
            self.header.set('运行或加载物理结果后显示连续源.');self.draw();return
        if self.data.rate_preview:
            self.header.set('样品流速已修改: 请先完成物理求解, 再更新全程位置与到达时间.');self.draw();return
        try:
            options=ProcessParameters(duration_s=self.horizon,startup='empty' if self.startup.get()=='从无细胞开始' else 'steady')
            self.model=ProcessModel(self.data,AcquisitionParameters(signal_width_ms=0.),options)
        except (ValueError,TypeError,KeyError) as error:
            self.header.set(str(error));self.draw();return
        m=self.model;g=self.data.config.geometry
        self.header.set(f'入口浓度 {self.data.config.cell.concentration_per_ml:g}/mL ×  Actual  Q {self.data.config.fluid.sample_flow_ul_min:.5g} μL/min → {m.source_rate:.3f}  Count /s.'
            f'每次进样带独立编号, 复用 {len(m.tracks)} 条保存的完整响应; 响应样本不是同时注入的细胞.')
        if self.startup.get()=='稳态连续进样':self.now=1.
        self.focus_zmin=-max(1.5*g.inner_recess_mm,.5)*1e-3
        self.mean_tip=sum(window_residence(tr,g.liquid_start,0) for tr in m.tracks)/len(m.tracks)
        self.mean_zoom=sum(window_residence(tr,self.focus_zmin,0) for tr in m.tracks)/len(m.tracks)
        self.status.set(('稳态连续运行: t=0 为观察起点; 负的注入时刻属于上游预运行, 管内已有连续细胞流.'
            if self.startup.get()=='稳态连续进样' else ' Empty capillary : t=0 开始连续注入, 细胞需经过完整运输时间才到达出口.')
            +'  Magnified cell markers , 位置取自保存的运输轨迹; 不移动重合位置.')
        if not m.has_bulk:
            self.status.set(self.status.get()+' 此旧结果缺少长管空间历史, 长管段仅为端点插值, 需重新求解后定量解释.')
        self.playing=True;self.play_button.configure(text='Ⅱ  Pause ');self.last_tick=time.perf_counter()
        self.slider.set(self.now);self.draw()

    def set_pending(self,config):
        self.pending_config=deepcopy(config)
        if self.data is not None:
            from .source_state import pending_message
            self.pending.set(pending_message(self.data.config,config))

    def resize(self,event):
        if event.widget is self:
            for label in self.wrapped:label.configure(wraplength=max(300,event.width-24))

    def slider_changed(self,value):
        self.now=float(value);self.draw()

    def seek(self,value):
        self.playing=False;self.play_button.configure(text='▶  Play ')
        self.now=float(np.clip(value,0,self.horizon));self.slider.set(self.now);self.draw()

    def toggle(self):
        if self.model is None:return
        self.playing=not self.playing;self.last_tick=time.perf_counter()
        self.play_button.configure(text='Ⅱ  Pause ' if self.playing else '▶  Play ')

    def next_time(self,key):
        if self.model is None:return None
        result=next_stage_event(self.model,key,self.now)
        if result is None or result[1]>self.horizon:return None
        self.selected_event,t=result
        return t

    def jump(self,key):
        t=self.next_time(key)
        if t is not None:self.seek(t)

    def focus_slow(self):
        t=self.next_time('inner_time')
        if t is not None:
            self.speed.set('0.05×');self.seek(t);self.playing=True
            self.play_button.configure(text='Ⅱ  Pause ');self.last_tick=time.perf_counter()

    def tick(self):
        self.job=None
        if self.closed:return
        clock=time.perf_counter();elapsed=min(.25,clock-self.last_tick);self.last_tick=clock
        # Hidden tabs do not advance or redraw in the background.
        if self.playing and self.model is not None and self.winfo_ismapped():
            self.now=min(self.horizon,self.now+elapsed*float(self.speed.get().replace('×','')))
            self.slider.set(self.now);self.draw()
            if self.now>=self.horizon:self.playing=False;self.play_button.configure(text='▶  Play ')
        self.job=self.after(80,self.tick)

    def pick(self,event):
        click_x=self.canvas.canvasx(event.x);click_y=self.canvas.canvasy(event.y)
        for x,y,eid in reversed(self.hits):
            if (click_x-x)**2+(click_y-y)**2<100:self.selected_event=eid;self.draw();return

    def text(self,x,y,value,**kwargs):
        return self.canvas.create_text(x,y,text=value,anchor='w',fill=kwargs.pop('fill','#304b61'),
            font=kwargs.pop('font',('Microsoft YaHei UI',9)),**kwargs)

    def _bodies(self,points,transform):
        groups=group_screen_points(points,transform);c=self.canvas
        for group in groups:
            selected=next((item for item in group if item[0]['event_id']==self.selected_event),None)
            point,x,y=selected or group[0];eid=point['event_id'];highlight=selected is not None
            r=6 if highlight else 4
            c.create_oval(x-r,y-r,x+r,y+r,fill=CELL_COLORS[eid%len(CELL_COLORS)],outline='#243b54',width=2 if highlight else 1)
            if len(group)>1 and (len(group)>=10 or highlight):self.text(x+8,y-10,f'×{len(group)}')
            elif highlight:self.text(x+8,y-10,f'#{eid}')
            self.hits.extend((xx,yy,p['event_id']) for p,xx,yy in group)
        return len(groups)

    def draw(self):
        if self.closed:return
        c=self.canvas;c.delete('all');self.hits=[];m=self.model
        if m is None:
            self.text(18,32,'暂无可回放的全程运输结果.载入已有结果或完成求解后, 连续细胞从内管入口进入.')
            self.clock.set('');return
        g=self.data.config.geometry;w=max(600,c.winfo_width());h=max(340,c.winfo_height())
        c.configure(scrollregion=(0,0,w,h))
        positions=m.positions(self.now)
        r=m.records;created=(r['injection_s']>=0)&(r['injection_s']<=self.now)
        crossed=(r['nozzle_s']>=0)&(r['nozzle_s']<=self.now)
        bulk=[p for p in positions if p['z_m']<g.liquid_start]
        tip=[p for p in positions if g.liquid_start<=p['z_m']<0]
        zoom=[p for p in tip if p['z_m']>=self.focus_zmin]
        self.clock.set(f'观察时钟 t={self.now:.5f} s | 当前长管 {len(bulk)}  Count  + 末端 {len(tip)}  Count  (其中局部放大 {len(zoom)}  Count )'
            f' | 本观察段注入 {int(created.sum())}  Count  / 出中管口 {int(crossed.sum())}  Count ')
        finished=r['injection_s']+m.delays[r['template_id']]
        stopped=(finished>=0)&(finished<=self.now)&~m.exited[r['template_id']]
        if np.any(stopped):self.clock.set(self.clock.get()+f'; 运输停止 /  Pending continuation  {int(stopped.sum())}  Count ')
        left=65;right=w-30;top=56;bottom=min(155,max(110,h*.30));mid=(top+bottom)/2
        c.create_rectangle(left,top-6,right,bottom+6,fill='#d5dfe7',outline='')
        c.create_rectangle(left,top,right,bottom,fill='#e6f2fb',outline='#7997ae')
        sx=lambda z:left+(z-m.start_z)/(g.bulk_length or 1e-12)*(right-left)
        sy=lambda y:mid-y/g.a*(bottom-top)*.5
        self.text(14,16,f'长管前段 {g.bulk_length*1e3:g} mm; 下方放大末端内管 {g.sample_length_mm-g.bulk_length*1e3:g} mm + 管后 {g.inner_recess_mm:g} mm')
        draw_bulk=bulk[::max(1,int(np.ceil(len(bulk)/300)))]
        bulk_marks=self._bodies(draw_bulk,lambda p:(sx(p['z_m']),sy(p['y_m'])))
        self.text(left,bottom+20,'内管前端');self.text(right-92,bottom+20,'进入末端数值区')
        self.text(4,mid,' Projection ',font=('Microsoft YaHei UI',8))
        note=f'前段 {len(bulk)}  Cells ; 绘制 {len(draw_bulk)}  Count , 合并为 {bulk_marks} 个显示位置'
        self.text(left,top-18,note)
        low_top=bottom+62;low_bottom=h-115
        for slot,zmin,title,visible in [(0,g.liquid_start,'末端全部区域',tip),(1,self.focus_zmin,'内管出口及聚焦区',zoom)]:
            axleft=left+slot*(w-left-25)/2;axright=axleft+(w-left-25)/2-24
            xx=lambda z:axleft+(z-zmin)/(0-zmin)*(axright-axleft)
            yy=lambda r:low_bottom-r/g.c*(low_bottom-low_top)
            c.create_rectangle(axleft,low_top,axright,low_bottom,fill='#e8f2fa',outline='#9bb2c3')
            if zmin<g.inner_tip:
                c.create_rectangle(axleft,yy(g.b),xx(g.inner_tip),yy(g.a),fill='#cbd2d8',outline='')
                self.text((axleft+xx(g.inner_tip))/2-13,yy((g.a+g.b)/2),' Quartz ',fill='#61717f')
            c.create_line(xx(g.inner_tip),low_top,xx(g.inner_tip),low_bottom,fill='#8e9da9',dash=(3,3))
            c.create_line(axright,low_top,axright,low_bottom,fill='#31826f',width=2)
            self.text(axleft,low_top-19,f'{title}:  Current  {len(visible)}  Count ')
            drawn=visible[::max(1,int(np.ceil(len(visible)/300)))]
            markers=self._bodies(drawn,lambda p:(xx(p['z_m']),yy(p['r_m'])))
            self.text(axleft,low_bottom+21,f'{zmin*1e3:.2f} → 0 mm; {len(drawn)}  Cells  / {markers} 个位置',font=('Microsoft YaHei UI',8))
            self.text(axleft-43,low_top-12,'r μm');self.text(axleft-28,low_top+8,f'{g.c*1e6:g}');self.text(axleft-28,low_bottom,'0')
            if not visible:self.text(axleft+12,(low_top+low_bottom)/2,'此刻没有细胞; 没有在出口补放粒子.',width=max(200,axright-axleft-24))
        interval='无进样' if m.source_rate==0 else f'{1/m.source_rate:.2f} s'
        self.text(14,h-63,f'平均进样间隔 {interval}; 稳态末端平均 {m.source_rate*self.mean_tip:.2f}  Count , 局部放大平均 {m.source_rate*self.mean_zoom:.2f}  Count .',width=w-28)
        if self.selected_event is not None:
            i=self.selected_event;track=m.event_track(i);a=r['injection_s'][i]
            self.text(14,h-25,f'#{i}:  Injection  {a:.4f} s → 进入末端 {a+track["bulk_time"]:.4f} s →  Sample outlet  {a+track["inner_time"]:.4f} s →  Sheath outlet  {r["nozzle_s"][i]:.4f} s; 样本 {self.data.ids[r["template_id"][i]]}',width=w-28,font=('Microsoft YaHei UI',8))

    def close(self):
        if self.closed:return
        self.closed=True;self.playing=False
        if self.job is not None:
            try:self.after_cancel(self.job)
            except tk.TclError:pass
        self.job=None
