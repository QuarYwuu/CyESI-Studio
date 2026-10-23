"""Linked numerical transport, focusing, conditional inlet and MS-gate movie."""
from copy import deepcopy,copy
from datetime import datetime
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk,messagebox
import numpy as np
from ..visualization.process_data import (ProcessModel,ProcessParameters,selected_flight,
    STATUS_LABELS,STATUS_COLORS,CELL_COLORS)
from ..acquisition.q_exactive import AcquisitionParameters


class ProcessWindow:
    def __init__(self,parent,data,acquisition=None,resolution=17500,actual_it=30.,on_run=None,state_parameters=None,process_parameters=None):
        self.window=tk.Toplevel(parent);self.window.title('CyESI · 连续细胞:  Long capillary  →  Focusing  → 入孔 →  Acquisition ')
        self.window.geometry('1480x960');self.window.minsize(1080,740)
        self.data=data;self.acquisition=deepcopy(acquisition or AcquisitionParameters(signal_width_ms=20.))
        options=process_parameters or ProcessParameters(resolution=resolution,actual_it_ms=actual_it)
        self.state_parameters=deepcopy(state_parameters);self.on_run=on_run
        self.closed=False;self.job=None;self.tick_job=None;self.poll_job=None;self.messages=queue.Queue()
        self.model=None;self.output=None;self.playing=False;self.busy=False;self.children=[];self.flight_revision=0
        self.header=tk.StringVar();self.pending=tk.StringVar();self.status=tk.StringVar()
        ttk.Label(self.window,text='连续细胞过程与采集回收',font=('Microsoft YaHei UI',17,'bold')).pack(anchor='w',padx=16,pady=(10,4))
        self.header_label=ttk.Label(self.window,textvariable=self.header,wraplength=1420);self.header_label.pack(fill='x',padx=16)
        notice=ttk.Frame(self.window);notice.pack(fill='x',padx=16,pady=3)
        self.pending_label=ttk.Label(notice,textvariable=self.pending,foreground='#ad541a',wraplength=1180);self.pending_label.pack(side='left',fill='x',expand=True)
        if on_run:ttk.Button(notice,text='应用参数并重算流场',command=on_run).pack(side='right')
        form=ttk.Frame(self.window);form.pack(fill='x',padx=16,pady=6)
        self.entries={}
        inputs=[('duration',' Inspect  s',f'{options.duration_s:g}'),('resolution',' Resolution ',str(options.resolution)),('it',' Actual  IT ms',f'{options.actual_it_ms:g}'),
            ('width','信号峰宽 ms',f'{self.acquisition.signal_width_ms:g}'),('delivery',' Exited →离子包',f'{self.acquisition.signal_delivery_fraction:g}'),
            ('delay',' Exited →包时延 ms',f'{options.packet_delay_ms:g}'),('startup','启动','稳态' if options.startup=='steady' else ' Empty capillary ')]
        for i,(key,label,value) in enumerate(inputs):
            cell=ttk.Frame(form);cell.pack(side='left',padx=(0,12));ttk.Label(cell,text=label).pack(anchor='w')
            var=tk.StringVar(value=value);self.entries[key]=var
            options={'duration':['60','600'],'resolution':['17500','35000','70000','140000'],'startup':['稳态',' Empty capillary ']}
            if key in options:ttk.Combobox(cell,textvariable=var,values=options[key],width=11,state='readonly' if key=='startup' else 'normal').pack()
            else:ttk.Entry(cell,textvariable=var,width=11).pack()
        ttk.Button(form,text='更新过程',command=self.rebuild).pack(side='left',anchor='s',padx=3)
        self.export_button=ttk.Button(form,text='导出四张图 / GIF /  Report ',command=self.export);self.export_button.pack(side='right',anchor='s')
        self.assumptions_label=ttk.Label(self.window,text='峰宽, 输送比例与时延为可修改假设; 非零峰宽用于条件情景. Actual  IT 可能小于 Maximum IT.完整离子信号 ≠ 生物完整细胞; 0 ms 时完整度未定义.',
            foreground='#586b7c',wraplength=1420);self.assumptions_label.pack(fill='x',padx=16,pady=(0,5))
        bar=ttk.Frame(self.window);bar.pack(fill='x',padx=16,pady=4)
        self.play_button=ttk.Button(bar,text='▶  Play ',command=self.toggle);self.play_button.pack(side='left')
        ttk.Label(bar,text='追踪细胞').pack(side='left',padx=(12,3))
        self.tag=tk.StringVar();self.tag_selector=ttk.Combobox(bar,textvariable=self.tag,values=[],state='readonly',width=12);self.tag_selector.pack(side='left')
        self.tag_selector.bind('<<ComboboxSelected>>',lambda e:self.select_tag())
        for label,value in [('进入长管',0.),('长管末段',.55),(' Sample outlet ',.72),('中管出口',.86),('采集去向',.97)]:
            ttk.Button(bar,text=label,command=lambda v=value:self.seek(v)).pack(side='left',padx=3)
        self.progress=tk.DoubleVar(value=.72)
        ttk.Scale(self.window,from_=0,to=1,variable=self.progress,command=lambda v:self.draw()).pack(fill='x',padx=16)
        self.clock=tk.StringVar();ttk.Label(self.window,textvariable=self.clock,foreground='#22466b').pack(fill='x',padx=16,pady=(0,5))
        self.notebook=ttk.Notebook(self.window);self.notebook.pack(fill='both',expand=True,padx=12)
        self.notebook.bind('<<NotebookTabChanged>>',lambda e:self.draw())
        self.canvases={}
        for key,title in [('tube','①  Long capillary : 间距与沉降'),('focus','②  Outlet : 聚焦与溶剂'),('gates','③  Acquisition :  Complete  / 丢失 / 回收'),('inlet','④  Inlet : 液滴与细胞状态')]:
            pane=ttk.Frame(self.notebook);self.notebook.add(pane,text=title)
            if key=='gates':
                self.recovery=tk.StringVar();self.recovery_label=ttk.Label(pane,textvariable=self.recovery,wraplength=1400,justify='left',font=('Microsoft YaHei UI',10,'bold'));self.recovery_label.pack(fill='x',padx=10,pady=6)
            if key=='inlet':self._inlet_controls(pane)
            canvas=tk.Canvas(pane,bg='#f6f9fc',highlightthickness=0,height=220 if key=='gates' else 420)
            canvas.pack(fill='both',expand=True);canvas.bind('<Configure>',lambda e:self.draw())
            self.canvases[key]=canvas
            if key=='gates':self._event_table(pane)
        ttk.Label(self.window,textvariable=self.status,wraplength=1430).pack(fill='x',padx=16,pady=7)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.window.bind('<Destroy>',lambda e:self.cleanup() if e.widget is self.window else None,add='+')
        self.window.bind('<Configure>',self.resize,add='+')
        for var in self.entries.values():var.trace_add('write',self.schedule)
        self.rebuild();self.poll_job=self.window.after(100,self.poll)

    def resize(self,event):
        if event.widget is not self.window:return
        width=max(600,event.width-50)
        for label in (self.header_label,self.assumptions_label,self.recovery_label):label.configure(wraplength=width)
        self.pending_label.configure(wraplength=max(500,width-200))

    def _inlet_controls(self,parent):
        from ..cell_state.parameters import StateParameters
        p=self.state_parameters or StateParameters()
        form=ttk.Frame(parent,padding=8);form.pack(fill='x');self.flight_entries={}
        specs=[('drop_radii_um','载体半径 μm',p.drop_radii_um[len(p.drop_radii_um)//2]),
            ('charge_fractions','q / Rayleigh',p.charge_fractions[len(p.charge_fractions)//2]),
            ('methanol_permeabilities_m_s',' Membrane permeability  m/s',p.methanol_permeabilities_m_s[len(p.methanol_permeabilities_m_s)//2]),
            ('dissolution_rates_s',' Dissolution  k /s',p.dissolution_rates_s[len(p.dissolution_rates_s)//2]),
            ('flow_factors','吸气倍率',p.flow_factors[len(p.flow_factors)//2]),
            ('alignment_offset_um','横向偏移 μm',p.alignment_offset_um)]
        for key,label,value in specs:
            cell=ttk.Frame(form);cell.pack(side='left',padx=(0,9));ttk.Label(cell,text=label).pack(anchor='w')
            v=tk.StringVar(value=f'{value:g}');self.flight_entries[key]=v;ttk.Entry(cell,textvariable=v,width=12).pack()
            v.trace_add('write',self.invalidate_flight)
        self.flight_button=ttk.Button(form,text='计算选中细胞入孔情景',command=self.compute_flight);self.flight_button.pack(side='left',anchor='s',padx=8)
        ttk.Label(parent,text=f' Inlet ID  {p.inlet_diameter_mm:g} mm;  Wall temperature  {p.inlet_wall_temperature_c:g}°C (不是细胞温度); 默认吸气为工程估算.条件情景不构成总体入口回收率.',wraplength=1400).pack(fill='x',padx=10)

    def _event_table(self,parent):
        frame=ttk.Frame(parent);frame.pack(fill='both',expand=True,padx=10,pady=5)
        columns=['id','source','lip','pulse','status','coverage','pure','spectra']
        self.table=ttk.Treeview(frame,columns=columns,show='headings',height=8,selectmode='browse')
        for key,title,width in zip(columns,[' Cell  ID','轨迹样本 ID',' Exited  s','离子包 s','去重去向','信号收集 %','纯信号 %','参与谱数'],[85,100,100,100,210,105,105,90]):
            self.table.heading(key,text=title);self.table.column(key,width=width,anchor='center')
        scrollbar=ttk.Scrollbar(frame,orient='vertical',command=self.table.yview);self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side='left',fill='both',expand=True);scrollbar.pack(side='right',fill='y')
        self.table.bind('<<TreeviewSelect>>',self.table_select)

    def schedule(self,*args):
        if self.closed:return
        if self.job is not None:self.window.after_cancel(self.job)
        self.job=self.window.after(400,self.rebuild)

    def collect(self):
        p=deepcopy(self.acquisition);p.signal_width_ms=float(self.entries['width'].get())
        p.signal_delivery_fraction=float(self.entries['delivery'].get());p.validate()
        options=ProcessParameters(duration_s=float(self.entries['duration'].get()),resolution=int(self.entries['resolution'].get()),
            actual_it_ms=float(self.entries['it'].get()),packet_delay_ms=float(self.entries['delay'].get()),
            startup='steady' if self.entries['startup'].get()=='稳态' else 'empty')
        return p,options.validate(p)

    def rebuild(self):
        if self.closed:return
        if self.job is not None:self.window.after_cancel(self.job);self.job=None
        try:
            p,options=self.collect();model=ProcessModel(self.data,p,options)
        except (ValueError,TypeError,KeyError) as error:
            self.model=None;self.playing=False;self.play_button.configure(text='▶  Play ')
            self.table.delete(*self.table.get_children());self.recovery.set('当前过程尚未计算: '+str(error))
            self.export_button.configure(state='disabled');self.flight_button.configure(state='disabled')
            self.status.set(str(error));self.draw();return
        previous=self.selected_id
        self.model=model;self.acquisition=p;self.output=None
        if not self.busy:self.export_button.configure(state='normal');self.flight_button.configure(state='normal')
        ordered=sorted(model.ledger,key=lambda r:r['nozzle_s']);ids=[r['event_id'] for r in ordered]
        if not ids:ids=model.records['event_id'][:20].tolist()
        self.tag_selector.configure(values=[f'#{i}' for i in ids]);self.tag.set(f'#{previous if previous in ids else model.default_event}' if ids else '')
        self.header.set(f"计算源: {Path(self.data.source).name}  |   Long capillary  {self.data.config.geometry.sample_length_mm:g} mm  |  Q={self.data.config.fluid.sample_flow_ul_min:.4g} μL/min  |   Concentration  {self.data.config.cell.concentration_per_ml:g}/mL  |   Injection  {model.source_rate:.3f}/s  |  保存 {len(model.tracks)} / 全部 {self.data.config.cell.count}  Responses   |  Full MS: {p.microscans} 微扫, 附加 {p.overhead_ms:g} ms ( Model settings )")
        self.status.set('同一编号跟随真实轨迹; 连续流在有限观察窗内采样. Playback speed varies by transport stage , 位置不按播放速度改变.' if model.has_bulk else '旧结果缺少长管历史; 前段位置是插值, 请重新运行以获得真实沉降轨迹.')
        self._populate();self.draw()

    @property
    def selected_id(self):
        try:return int(self.tag.get().lstrip('#'))
        except ValueError:return None

    def _populate(self):
        m=self.model;summary=m.summary();n=len(m.ledger);pure=summary['pure_unique_count']
        complete=' Unknown  (0 ms 脉冲)' if not m.acquisition.signal_width_ms else f"{summary['counts']['pure_complete']} / {n}"
        row=next((r for r in m.theory['rows'] if r['resolution_at_mz200']==m.parameters.resolution and r['actual_injection_ms']==m.parameters.actual_it_ms),None)
        theoretical='' if row is None else f" 全样本稳态理论:  Unique single-source events  {row['unique_pure_single_cell_events_s']:.3f}/s."
        self.recovery.set('本窗尚无出管事件; 细胞可能仍在长管运输.' if n==0 else f"本窗出管 {n} 个来源 → 至少一张纯谱 {pure} ({pure/n:.1%}); 纯信号完整收集 {complete}.")
        self.recovery.set(self.recovery.get()+theoretical+' 分母为出管队列; 输送, 采集与生物完整性分别统计.')
        self.table.delete(*self.table.get_children())
        for r in sorted(m.ledger,key=lambda x:x['nozzle_s']):
            fmt=lambda v:'未定义' if v is None else f'{v*100:.1f}'
            self.table.insert('','end',iid=str(r['event_id']),values=(f"#{r['event_id']}",r['transport_sample_id'],f"{r['nozzle_s']:.5f}",
                f"{r['packet_onset_s']:.5f}",STATUS_LABELS[r['status']],fmt(r['captured_fraction']),fmt(r['pure_captured_fraction']),r['scan_count']))

    def table_select(self,event=None):
        selection=self.table.selection()
        if selection:self.tag.set('#'+selection[0]);self.select_tag()

    def select_tag(self):
        if self.model:self.model.flight=None
        self.progress.set(.86);self.draw()

    def seek(self,value):self.progress.set(value);self.draw()

    def toggle(self):
        if self.model is None or self.selected_id is None:return
        self.playing=not self.playing;self.play_button.configure(text='Ⅱ  Pause ' if self.playing else '▶  Play ')
        if self.playing:
            if self.progress.get()>=1:self.progress.set(0)
            self.tick()

    def tick(self):
        self.tick_job=None
        if self.closed or not self.playing:return
        self.progress.set(min(1,self.progress.get()+.006));self.draw()
        if self.progress.get()>=1:self.playing=False;self.play_button.configure(text='▶  Play ')
        else:self.tick_job=self.window.after(80,self.tick)

    def set_pending(self,config):
        from .source_state import pending_message
        self.pending.set(pending_message(self.data.config,config))

    def set_data(self,data):
        if self.closed:return
        same=(data.source==self.data.source and data.config.to_dict()==self.data.config.to_dict() and data.rate_config.to_dict()==self.data.rate_config.to_dict())
        self.data=data
        if not same:self.pending.set('');self.rebuild()

    def draw(self):
        if self.closed:return
        for canvas in self.canvases.values():canvas.delete('all')
        m=self.model;tag=self.selected_id
        if m is None or tag is None:
            text=('当前没有有效过程结果.修改流速后需先完成物理求解; 输入未完成时请继续输入.' if m is None else
                  '连续源注入率为 0, 当前没有细胞事件.' if m.source_rate==0 else
                  '本观察窗暂无可回放事件.可延长观察时长; 空管启动需经过真实长管运输延迟.')
            for canvas in self.canvases.values():canvas.create_text(30,35,text=text,anchor='w',fill='#566c80')
            return
        t=m.global_time(tag,self.progress.get());age=t-m.records['injection_s'][tag]
        counts=m.conservation(t);positions=m.positions(t)
        self.clock.set(f" Track  #{tag} · 样本 {self.data.ids[m.records['template_id'][tag]]} · 入管后 {age:.5f} s · 全局 {t:.5f} s | 此刻在管 {counts['in_transit']}; 累计过中管口 {counts['crossed_lip']}; 运输待解析 {counts['transport_unresolved']} (含稳态预充液)")
        # Draw only the visible tab; this keeps a long-pipe movie inexpensive.
        selected=self.notebook.index(self.notebook.select())
        if selected==0:self.draw_tube(self.canvases['tube'],positions,tag,t)
        elif selected==1:self.draw_focus(self.canvases['focus'],positions,tag,t)
        elif selected==2:self.draw_gates(self.canvases['gates'],tag,t)
        else:self.draw_inlet(self.canvases['inlet'],tag,t)

    def _text(self,c,x,y,text,**kw):return c.create_text(x,y,text=text,anchor='w',font=('Microsoft YaHei UI',10),fill=kw.pop('fill','#30465b'),**kw)

    def draw_tube(self,c,positions,tag,t):
        m=self.model;g=self.data.config.geometry;w=max(800,c.winfo_width());h=max(330,c.winfo_height())
        left=78;right=w-45;top=75;bottom=min(h*.44,230);mid=(top+bottom)/2
        sx=lambda z:left+(z-m.start_z)/(g.sample_length_mm*1e-3)*(right-left)
        sy=lambda y:mid-y/(g.inner_id_um*.5e-6)*(bottom-top)*.5
        self._text(c,18,22,'长管内的真实轨迹与连续细胞列车: 轴向按管长显示, 径向放大显示.圆点尺寸为可见性放大.')
        c.create_rectangle(left,top-9,right,bottom+9,fill='#d7dfe7',outline='')
        c.create_rectangle(left,top,right,bottom,fill='#e7f3fc',outline='#7e9ab2',width=2)
        for k in np.unique(np.linspace(0,len(m.tracks)-1,min(18,len(m.tracks)),dtype=int)):
            tr=m.tracks[k];keep=tr['z']<=g.inner_tip
            points=[v for z,y in zip(tr['z'][keep],m.local_y[k][keep]) for v in (sx(z),sy(y))]
            if len(points)>=4:c.create_line(*points,fill='#bbcfdf',width=1)
        live=[p for p in positions if p['z_m']<=g.inner_tip]
        # Include the selected event in the decimated display.
        shown=live[::max(1,len(live)//250)];selected=[p for p in live if p['event_id']==tag]
        for p in shown+selected:
            x=sx(p['z_m']);y=sy(p['y_m']);r=7 if p['event_id']==tag else 3.5
            c.create_oval(x-r,y-r,x+r,y+r,fill=CELL_COLORS[p['event_id']%len(CELL_COLORS)],outline='#102f4c' if p['event_id']==tag else '')
            if p['event_id']==tag:self._text(c,x+9,y-13,f'#{tag}  r={p["r_m"]*1e6:.1f} μm')
        for fraction in [0,.25,.5,.75,1]:
            x=left+(right-left)*fraction;c.create_line(x,bottom,x,bottom+5,fill='#3d5568')
            self._text(c,x-10,bottom+23,f'{g.sample_length_mm*fraction:g} mm')
        self._text(c,18,mid,'重力投影');self._text(c,18,bottom+58,f'管路摆放: {self.data.config.supply.orientation}; 沿管路展开显示.径向采用固定重力平面投影, 水平管内向下为沉降方向.')
        y0=bottom+92;self._text(c,18,y0,'各截面的到达时刻 (同一编号连线): 斜率对应运输延迟, 线条交叉表示出管顺序变化.')
        from ..visualization.process_report import _neighbors
        neighbors=_neighbors(m,tag,9);origin=m.records['injection_s'][tag]
        series=[]
        for eid in neighbors:
            a=m.records['injection_s'][eid]+m.crossing[m.records['template_id'][eid]]-origin;series.append((eid,a))
        values=np.concatenate([v[np.isfinite(v)] for _,v in series]);minimum=float(values.min());maximum=max(float(values.max()),minimum+1e-6)
        ystart=y0+38;yend=max(ystart+80,h-52)
        for eid,a in series:
            xy=[v for j,value in enumerate(a) if np.isfinite(value) for v in (left+j*(right-left)/5,ystart+(value-minimum)/(maximum-minimum)*(yend-ystart))]
            if len(xy)>=4:c.create_line(*xy,fill=CELL_COLORS[eid%len(CELL_COLORS)],width=3 if eid==tag else 1)
            if len(xy)>=2:self._text(c,xy[-2]+6,xy[-1],f'#{eid}',fill=CELL_COLORS[eid%len(CELL_COLORS)])
        for j,name in enumerate(m.stage_names):self._text(c,left+j*(right-left)/5-20,ystart-16,name)
        partner=next((eid for eid in neighbors if eid==tag+1),None)
        if partner is not None:
            gap_in=m.records['injection_s'][partner]-m.records['injection_s'][tag]
            gap_out=m.records['nozzle_s'][partner]-m.records['nozzle_s'][tag]
            self._text(c,18,h-18,f'同一对 #{tag} → #{partner}:  Inlet  Δt={gap_in:.3f} s;  Exited  Δt={gap_out:.3f} s (负值表示次序交换).径向聚焦不保证时间隔离.')

    def draw_focus(self,c,positions,tag,t):
        m=self.model;g=self.data.config.geometry;w=max(800,c.winfo_width());h=max(330,c.winfo_height())
        left=80;right=w-55;top=70;bottom=h-160;zmin=-max(1.5*g.inner_recess_mm,.5)*1e-3
        sx=lambda z:left+(z-zmin)/(-zmin)*(right-left)
        sy=lambda r:bottom-r/g.c*(bottom-top)
        self._text(c,18,24,'出口混合区放大: 蓝色越深, 局部甲醇体积分数越高;  Grey: quartz .只使用保存的数值场和球心轨迹.')
        f=self.data.fields;ze=f['liquid_z_edges_m'];re=f['liquid_r_edges_m'];active=f['liquid_active'];field=f['methanol_fraction']
        # Sample rectangles directly from the solved finite-volume grid.
        c.create_rectangle(left,top,right,bottom,fill='#d9dfe5',outline='')
        for j in range(len(ze)-1):
            if ze[j+1]<zmin or ze[j]>0:continue
            for i in range(len(re)-1):
                if not active[i,j]:continue
                value=float(np.clip(field[i,j],0,1));rgb=[int(234-174*value),int(245-106*value),int(252-52*value)]
                color='#%02x%02x%02x'%tuple(rgb)
                c.create_rectangle(sx(max(ze[j],zmin)),sy(re[i+1]),sx(min(ze[j+1],0)),sy(re[i]),fill=color,outline='')
        for index in np.unique(np.linspace(0,len(m.tracks)-1,min(30,len(m.tracks)),dtype=int)):
            tr=m.tracks[index];keep=tr['z']>=zmin
            xy=[v for z,r in zip(tr['z'][keep],tr['r'][keep]) for v in (sx(z),sy(r))]
            if len(xy)>=4:c.create_line(*xy,fill='#9bb6c6',width=1)
        k=m.records['template_id'][tag];tr=m.tracks[k];keep=tr['z']>=zmin
        xy=[v for z,r in zip(tr['z'][keep],tr['r'][keep]) for v in (sx(z),sy(r))]
        if len(xy)>=4:c.create_line(*xy,fill=CELL_COLORS[tag%len(CELL_COLORS)],width=3)
        for p in positions:
            if p['z_m']<zmin or p['z_m']>0:continue
            x=sx(p['z_m']);y=sy(p['r_m']);r=8 if p['event_id']==tag else 5
            c.create_oval(x-r,y-r,x+r,y+r,fill=CELL_COLORS[p['event_id']%len(CELL_COLORS)],outline='#163f60')
            self._text(c,x+10,y-12,f'#{p["event_id"]}')
        x=sx(g.inner_tip);c.create_line(x,top,x,bottom,fill='#6f7c88',dash=(4,4));self._text(c,x+5,top-18,' Sample outlet ')
        self._text(c,right-55,top-18,'中管出口');self._text(c,left,bottom+24,f'{zmin*1e3:.2f} mm');self._text(c,right-25,bottom+24,'0 mm')
        self._text(c,15,top,'r (μm)');self._text(c,24,top+24,f'{g.c*1e6:g}');self._text(c,24,bottom,'0')
        age=t-m.records['injection_s'][tag];snapshot=self.data.snapshot(k,age)
        self._text(c,18,bottom+60,f'#{tag}:  Radial  {snapshot.r_m*1e6:.2f} μm;  External methanol  {snapshot.methanol:.1%}; 进入内管至今 {age:.5f} s.')
        self._text(c,18,bottom+88,f' Inlet  r={tr["r"][0]*1e6:.2f} μm →  Exited  r={tr["r"][-1]*1e6:.2f} μm; 出口混合区历时 {(tr["end_time"]-tr["inner_time"])*1000:.2f} ms.')
        self._text(c,18,bottom+118,'出管后仍保留事件编号: 采集去向见第③页; 可解析的条件飞行见第④页.')

    def draw_gates(self,c,tag,t):
        m=self.model;w=max(800,c.winfo_width());h=max(200,c.winfo_height());left=115;right=w-35
        center=m.by_id[tag]['packet_onset_s'] if tag in m.by_id else t
        span=max(2.,m.row['record_cycle_ms']*.008,m.acquisition.signal_width_ms*.004)
        a=center-span*.25;b=a+span;sx=lambda x:left+(x-a)/span*(right-left)
        self._text(c,15,22,'同一事件列车与收集门: 横轴为全局物理时间; 绿色单来源, 橙色混合, 灰色为空门.')
        self._text(c,15,77,' Actual  IT 门');self._text(c,15,145,'离子信号包')
        for scan in m.scans:
            if scan['gates'][-1][1]<a or scan['gates'][0][0]>b:continue
            ids=scan['event_ids'];color='#49ae98' if len(ids)==1 else '#e3a258' if ids else '#b2c2d0'
            for start,end in scan['gates']:
                c.create_rectangle(sx(max(a,start)),53,sx(min(b,end)),100,fill=color,outline='')
            if ids:self._text(c,sx(scan['start_s']),40,f'{len(ids)}  Source ')
        width=m.acquisition.signal_width_ms*.001
        for row in m.ledger:
            start=row['packet_onset_s'];end=start+width
            if end<a or start>b:continue
            y=132 if row['event_id']==tag else 157;x=sx(start)
            color=STATUS_COLORS[row['status']]
            c.create_line(x,y,sx(end) if width else x,y+0.1,fill=color,width=8)
            c.create_oval(x-4,y-4,x+4,y+4,fill=color,outline='')
            self._text(c,x+4,y+19,f'#{row["event_id"]} {STATUS_LABELS[row["status"]]}')
        for q in np.linspace(a,b,9):
            x=sx(q);c.create_line(x,h-27,x,h-20,fill='#61788c');self._text(c,x-13,h-10,f'{q:.2f}')
        if a<=t<=b:c.create_line(sx(t),30,sx(t),h-30,fill='#bd3556',width=2)
        else:self._text(c,right-290,21,'时标正在回放上游; 此处固定显示该事件入包附近.')

    def invalidate_flight(self,*args):
        self.flight_revision+=1
        if getattr(self,'model',None) is not None:self.model.flight=None;self.output=None;self.draw()

    def draw_inlet(self,c,tag,t):
        w=max(800,c.winfo_width());h=max(330,c.winfo_height());m=self.model;g=self.data.config.geometry
        left=80;right=w*.72;top=90;bottom=h-110;mid=(top+bottom)/2
        sx=lambda z:left+z/(g.ms_gap_mm*1e-3)*(right-left)
        sy=lambda r:mid-r/(.35e-3)*(bottom-top)/2
        self._text(c,18,26,'出口到入口的飞行与细胞状态: 选中细胞的条件微滴模型.未解析端点保留, 不把粒子直接移动到质谱内.')
        c.create_rectangle(left-20,mid-30,left,mid+30,fill='#bcc9d3',outline='')
        a=(self.state_parameters.inlet_diameter_mm if self.state_parameters else .6)*.5e-3
        c.create_rectangle(right,top-20,right+25,sy(a),fill='#aab9c6',outline='')
        c.create_rectangle(right,sy(-a),right+25,bottom+20,fill='#aab9c6',outline='')
        c.create_line(left,mid,right,mid,fill='#bdd1df',dash=(5,5))
        self._text(c,left-10,bottom+42,' Sheath outlet  z=0');self._text(c,right-80,bottom+42,f'MS 孔面 z={g.ms_gap_mm:g} mm')
        flight=m.flight
        if flight is None or flight['event_id']!=tag:
            self._text(c,left+30,mid-55,'请计算选中细胞入孔情景, 得到路径, 液滴尺寸和膜状态.')
            self._text(c,left+30,mid-25,'几何入孔后到 C-trap 离子包的传输, 展宽与检出需要校准.')
            return
        result=flight['result'];history=flight['history']
        if history:
            xy=[v for p in history for v in (sx(p['z_m']),sy(p['r_m']))]
            if len(xy)>3:c.create_line(*xy,fill='#538bc0',width=2)
            age=t-m.records['nozzle_s'][tag];times=[p['time_after_lip_s'] for p in history]
            if age<0:
                self._text(c,left+30,mid-25,'选中细胞仍在上游; 蓝线为该情景求得的后续飞行路径.')
                return
            position=min(np.searchsorted(times,age),len(history)-1);p=history[position]
            x=sx(p['z_m']);y=sy(p['r_m']);radius=14
            c.create_oval(x-radius,y-radius,x+radius,y+radius,fill='#9bd1e3',outline='#2379aa',width=2)
            c.create_oval(x-6,y-6,x+6,y+6,fill=CELL_COLORS[tag%len(CELL_COLORS)],outline='#414c69')
            self._text(c,x+16,y-15,f'#{tag}  {p["time_after_lip_s"]*1e3:.4f} ms')
            details=[f'液滴半径 {p["drop_radius_m"]*1e6:.2f} μm',f' Temperature  {p["temperature_k"]-273.15:.1f} °C',
                f'内部甲醇 {p["intracellular_methanol_fraction"]:.1%}',f' Remaining membrane lipid  {p["membrane_lipid_remaining_fraction"]:.1%}',
                f' Intracellular tracer  {p["intracellular_tracer_fraction"]:.1%}']
            for i,label in enumerate(details):self._text(c,w*.78,top+30+i*32,label)
            if age>=times[-1]:self._text(c,left+15,bottom+75,'路径终点保留为计算终止状态; 未解析部分不外推.',fill='#a35a26')
        else:self._text(c,left+30,mid-25,'本情景没有满足条件的可解析飞行段.')
        self._text(c,18,56,f' Status : {result["status"]}; 本页不产生总体入孔效率或细胞存活率.',fill='#a35a26')

    def compute_flight(self):
        if self.busy or self.model is None or self.selected_id is None:return
        try:
            from ..cell_state.parameters import StateParameters
            p=deepcopy(self.state_parameters or StateParameters())
            for key,var in self.flight_entries.items():setattr(p,key,float(var.get()) if key=='alignment_offset_um' else [float(var.get())])
            p.validate()
        except ValueError as error:messagebox.showerror('入孔参数错误',str(error),parent=self.window);return
        model=self.model;tag=self.selected_id;messages=self.messages;revision=self.flight_revision
        self.busy=True;self.flight_button.configure(state='disabled');self.export_button.configure(state='disabled');self.status.set('正在计算同一轨迹的膜扩散, 电作用, 蒸发与条件飞行…')
        def work():
            try:messages.put(('flight',model,(revision,selected_flight(model,tag,p))))
            except Exception:messages.put(('error',model,traceback.format_exc()))
        threading.Thread(target=work,daemon=True).start()

    def export(self,directory=None,animation=True):
        if self.busy or self.model is None:return
        model=self.model;snapshot=copy(model);snapshot.default_event=self.selected_id;snapshot.flight=deepcopy(model.flight)
        output=Path(directory) if directory else Path(model.data.source)/'process_views'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        messages=self.messages;self.busy=True;self.flight_button.configure(state='disabled');self.export_button.configure(state='disabled')
        self.status.set('正在导出当前参数和事件队列的图, 动画与原始账本…')
        def work():
            try:
                from ..visualization.process_report import export
                result=export(snapshot,output,animation,progress=lambda s:messages.put(('progress',model,s)))
                messages.put(('export',model,result))
            except Exception:messages.put(('error',model,traceback.format_exc()))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,model,payload=self.messages.get_nowait()
                if kind=='progress':self.status.set(payload);continue
                self.busy=False;state='normal' if self.model is not None else 'disabled'
                self.flight_button.configure(state=state);self.export_button.configure(state=state)
                if kind=='error':self.status.set('计算或导出失败; 可以修改参数后重试.');messagebox.showerror('过程视图',payload,parent=self.window)
                elif model is not self.model:self.status.set('已完成旧参数快照; 当前参数已更新, 请重新计算或导出.')
                elif kind=='flight':
                    revision,flight=payload
                    if revision==self.flight_revision and flight['event_id']==self.selected_id:self.model.flight=flight;self.status.set('选中细胞的条件飞行已计算;  Final state : '+flight['result']['status']);self.draw()
                    else:self.status.set('入孔结果对应旧选择或旧情景参数; 请重新计算.')
                else:
                    self.output=payload;self.status.set('已导出: '+str(payload/'process_report.html'))
                    import webbrowser
                    webbrowser.open((payload/'process_report.html').as_uri())
        except queue.Empty:pass
        self.poll_job=self.window.after(100,self.poll)

    def cleanup(self):
        if self.closed:return
        self.closed=True;self.playing=False
        for job in (self.job,self.tick_job,self.poll_job):
            if job is not None:
                try:self.window.after_cancel(job)
                except tk.TclError:pass
        self.job=self.tick_job=self.poll_job=None

    def close(self):
        if self.closed:return
        self.cleanup();self.window.destroy()


def launch(source=None,acquisition=None,parameters=None,state_parameters=None):
    from ..visualization.explorer_data import ExplorerData,find_results
    root=tk.Tk();root.withdraw();path=source or find_results()
    if path is None:root.destroy();raise ValueError('请先运行一次物理模拟')
    window=ProcessWindow(root,ExplorerData.load(path),acquisition,state_parameters=state_parameters,process_parameters=parameters)
    window.window.protocol('WM_DELETE_WINDOW',lambda:(window.close(),root.destroy()))
    root.mainloop()
