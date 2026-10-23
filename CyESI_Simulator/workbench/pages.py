from copy import deepcopy
from pathlib import Path
import json
import math
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np
from .widgets import *
from .drawing import DeviceCanvas,StreamCanvas,GateCanvas,txt,stream_problem
from .session import catalog
from ..gui.trajectory_stream import next_stage_event
from ..visualization.process_data import STATUS_LABELS
from ..acquisition.q_exactive import TRANSIENT_MS,timing,occupancy


def metric_string(value,unit=''):
    return ' Unknown ' if value is None else f'{value:.4g}{unit}'


def field_array(data,kind):
    f=data.fields
    if kind==' Electric field ':
        value=np.hypot(f['field_r_v_m'],f['field_z_v_m'])/1e6
        return f['electric_r_edges_m'],f['electric_z_edges_m'],value,None,'|E| (MV/m)','magma'
    if kind=='流速':value=np.hypot(f['velocity_r_m_s'],f['velocity_z_m_s'])*1000;unit='Speed (mm/s)';cmap='viridis'
    elif kind==' Pressure ':value=f['pressure_pa'];unit='Pressure (Pa)';cmap='viridis'
    elif kind==' Methanol ':value=f['methanol_fraction'];unit='Methanol fraction';cmap='BuPu'
    else:value=f.get('equivalent_stress_pa');unit='Equivalent stress (Pa)';cmap='magma'
    if value is None:raise ValueError('此旧结果尚未保存剪切应力场; 请查看已有压力 / 流速或重新计算.')
    return f['liquid_r_edges_m'],f['liquid_z_edges_m'],value,f['liquid_active'],unit,cmap


def draw_fields(plot,data,kind,baseline=None):
    if data is None:return
    datasets=[data]+([baseline] if baseline is not None else [])
    arrays=[field_array(d,kind) for d in datasets]
    lows=[];highs=[]
    for re,ze,v,active,_,_ in arrays:
        finite=v[np.isfinite(v)&(active if active is not None else True)]
        if len(finite):lows.append(finite.min());highs.append(finite.max())
    lo=min(lows) if lows else 0.;hi=max(highs) if highs else 1.
    if kind==' Methanol ':lo,hi=0.,1.
    if hi<=lo:hi=lo+1.
    axes=plot.axes(len(arrays))
    for ax,d,(re,ze,value,active,unit,cmap),name in zip(axes,datasets,arrays,['Current','Baseline']):
        masked=np.ma.masked_where(~active,value) if active is not None else value
        im=ax.pcolormesh(ze*1e3,re*1e6,masked,shading='flat',cmap=cmap,vmin=lo,vmax=hi)
        ax.set(xlabel='z (mm)',ylabel='r (um)',title=name)
        ax.axvline(d.config.geometry.inner_tip*1e3,color='#8395a8',ls='--',lw=.7)
        ax.axvline(0,color=TEAL,lw=.7)
        plot.figure.colorbar(im,ax=ax,label=unit,pad=.02,shrink=.88)
    plot.show()


def draw_resolution(plot,session,baseline=None):
    if session.data is None:return
    rows=[r for r in session.acquisition_result()['rows'] if r['actual_injection_ms']==session.actual_it_ms]
    values=[r['unique_pure_single_cell_events_s']*60 for r in rows];ax=plot.axes()[0];x=np.arange(4)
    if baseline:
        old=[r for r in session.acquisition_result(baseline)['rows'] if r['actual_injection_ms']==session.actual_it_ms]
        ax.bar(x-.18,[r['unique_pure_single_cell_events_s']*60 for r in old],width=.35,color='#c8d5e4',label='Baseline')
        ax.bar(x+.18,values,width=.35,color=TEAL,label='Current');ax.legend(frameon=False,fontsize=9)
    else:ax.bar(x,values,width=.55,color=[TEAL if r['resolution_at_mz200']==session.resolution else '#bfdedb' for r in rows])
    ax.set_xticks(x,[f'{r["resolution_at_mz200"]/1000:g}k' for r in rows]);ax.set_ylabel('Unique pure opportunities / min')
    for i,value in enumerate(values):ax.text(i,value,f'{value:.3g}',ha='center',va='bottom',fontsize=9,color=INK)
    ax.set_ylim(0,max(values+[.01])*1.3);plot.show()


class OverviewPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session
        card=Card(self,'从实验条件到单细胞信号','同一份数值结果贯穿所有页面.主流程默认观察稳态连续运行.');card.pack(fill='x')
        self.device=DeviceCanvas(card,self.session);self.device.pack(fill='x',padx=12,pady=(0,5))
        row=tk.Frame(self,bg=BG);row.pack(fill='both',expand=True,pady=(12,0));row.columnconfigure(0,weight=1,uniform='overview');row.columnconfigure(1,weight=1,uniform='overview');row.rowconfigure(0,weight=1)
        a=Card(row,'当前稳态分布');a.grid(row=0,column=0,sticky='nsew',padx=(0,6))
        self.population=tk.Canvas(a,bg=WHITE,highlightthickness=0,height=140);self.population.pack(fill='both',expand=True,padx=10)
        self.population.bind('<Configure>',lambda e:self.draw())
        self.note=tk.StringVar();self.note_label=label(a,textvariable=self.note,size=9,color=MUTED,wraplength=380,justify='left');self.note_label.pack(fill='x',padx=16,pady=8)
        a.bind('<Configure>',lambda e:self.note_label.configure(wraplength=max(200,e.width-32)))
        button(a,'查看连续运输与聚焦 →',lambda:app.navigate('transport'),primary=True).pack(side='bottom',anchor='w',padx=16,pady=(0,14),before=self.population)
        self.note_label.pack(side='bottom',fill='x',padx=16,pady=8,before=self.population)
        b=Card(row,' Resolution settings  · 理论单细胞机会','Full MS; 采集时序与信号输送为可修改情景.');b.grid(row=0,column=1,sticky='nsew',padx=(6,0))
        self.plot=Plot(b);self.plot.pack(fill='both',expand=True,padx=8,pady=8)
        button(b,'查看采集门与去重回收 →',lambda:app.navigate('acquisition')).pack(side='bottom',anchor='e',padx=14,pady=(0,10),before=self.plot)
        self.key=None
    def refresh(self):
        self.device.draw();s=self.session
        key=(id(s.data),s.revision)
        if key!=self.key:
            self.key=key
            if s.data:draw_resolution(self.plot,s,s.current_pin.data if s.current_pin else None)
        self.draw()
    def draw(self):
        c=self.population;c.delete('all');s=self.session
        if s.data is None:txt(c,20,65,'载入已有结果后即可开始观察.',color=MUTED);return
        problem=stream_problem(s)
        if problem:txt(c,20,35,problem,color=MUTED,width=max(260,c.winfo_width()-40));self.note.set('理论总体速率仍可查看; 动画容量不限制实际连续供液.');return
        try:
            points=s.positions();g=s.data.config.geometry
            counts=[sum(p['z_m']<g.liquid_start for p in points),sum(g.liquid_start<=p['z_m']<0 for p in points)]
            w=max(330,c.winfo_width());h=max(130,c.winfo_height());maximum=max(counts+[1])
            for i,(title,value,color) in enumerate(zip(['长管前段','末端数值区'],counts,[BLUE,PURPLE])):
                y=30+i*55;txt(c,12,y,title,size=10,color=MUTED)
                c.create_rectangle(115,y-10,w-55,y+10,fill=PALE,outline='')
                c.create_rectangle(115,y-10,115+(w-170)*value/maximum,y+10,fill=color,outline='')
                txt(c,w-40,y,str(value),size=11,color=color,anchor='center')
            self.note.set(f' Inspect  t={s.now:.2f} s; 长管与末端共 {len(points)} 个在途事件.\n稳态是统计分布稳定, 出口瞬时数量仍会波动.\n Measured  MS 检出率和生物完整率尚未校准.')
        except (ValueError,KeyError,TypeError) as error:self.note.set(str(error))


class TransportPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.key=None;self.exposure_key=None
        top=Card(self);top.pack(fill='x',pady=(0,10));bar=tk.Frame(top,bg=WHITE);bar.pack(fill='x',padx=12,pady=10)
        self.play=button(bar,' Pause ',app.toggle_play,primary=True);self.play.pack(side='left')
        button(bar,'下一颗聚焦 / 慢放',self.next_focus).pack(side='left',padx=6)
        label(bar,' Playback speed ',size=9,color=MUTED).pack(side='left',padx=(5,4))
        self.speed=tk.StringVar(value='1×');combo=ttk.Combobox(bar,textvariable=self.speed,values=['0.05×','0.1×','1×','10×','50×'],
            width=7,state='readonly',style='WB.TCombobox');combo.pack(side='left');combo.bind('<<ComboboxSelected>>',self.speed_changed)
        self.mode=tk.StringVar(value='稳态连续运行');combo=ttk.Combobox(bar,textvariable=self.mode,values=['稳态连续运行',' Empty capillary '],
            width=15,state='readonly',style='WB.TCombobox');combo.pack(side='right');combo.bind('<<ComboboxSelected>>',self.mode_changed)
        duration=tk.Frame(bar,bg=WHITE);duration.pack(side='right',padx=10)
        label(duration,' Inspect  s',size=9,color=MUTED).pack(side='left',padx=(0,4))
        self.duration=tk.StringVar(value=f'{self.session.duration_s:g}')
        combo=ttk.Combobox(duration,textvariable=self.duration,values=['60','600'],width=5,state='readonly',style='WB.TCombobox')
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',self.duration_changed)
        self.clock=tk.StringVar();label(top,textvariable=self.clock,size=9,color=MUTED).pack(fill='x',padx=15)
        self.slider=ttk.Scale(top,from_=0,to=self.session.duration_s,command=self.scrub);self.slider.pack(fill='x',padx=12,pady=(3,10));self._setting=False
        self.tabs=ttk.Notebook(self,style='WB.TNotebook');self.tabs.pack(fill='both',expand=True)
        live=Card(self.tabs);self.tabs.add(live,text=' Continuous cell stream ')
        self.inspector=tk.StringVar();label(live,textvariable=self.inspector,size=9,color=MUTED,wraplength=850,justify='left').pack(side='bottom',fill='x',padx=16,pady=8)
        drawing=tk.Frame(live,bg=WHITE);drawing.pack(fill='both',expand=True)
        self.stream=StreamCanvas(drawing,self.session,self.pick)
        scrollbar=ttk.Scrollbar(drawing,orient='vertical',command=self.stream.yview);scrollbar.pack(side='right',fill='y')
        self.stream.configure(yscrollcommand=scrollbar.set);self.stream.pack(side='left',fill='both',expand=True)
        field=Card(self.tabs);self.tabs.add(field,text=' Flow field  /  Focusing ')
        line=tk.Frame(field,bg=WHITE);line.pack(fill='x',padx=12,pady=10)
        label(line,'查看数值场',size=9,color=MUTED).pack(side='left',padx=(0,8))
        self.field=tk.StringVar(value='流速');combo=ttk.Combobox(line,textvariable=self.field,values=['流速',' Pressure ',' Methanol ',' Electric field ',' Shear '],
            width=10,style='WB.TCombobox',state='readonly');combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        label(line,'设定对比基准后, 以相同色标并排比较.',size=9,color=MUTED).pack(side='left',padx=14)
        self.field_plot=Plot(field);self.field_plot.pack(fill='both',expand=True,padx=8,pady=8)
        exposure=Card(self.tabs);self.tabs.add(exposure,text='暴露 /  Shear  / 时间')
        self.exposure_plot=Plot(exposure);self.exposure_plot.pack(fill='both',expand=True,padx=8,pady=8)
        self.exposure_note=tk.StringVar();label(exposure,textvariable=self.exposure_note,size=9,color=MUTED,wraplength=850,
            justify='left').pack(fill='x',padx=15,pady=8)
        original=Card(self.tabs);self.tabs.add(original,text='原始图 /  Animation ')
        line=tk.Frame(original,bg=WHITE);line.pack(fill='x',padx=12,pady=9)
        self.original=tk.StringVar(value='有限球动画');combo=ttk.Combobox(line,textvariable=self.original,
            values=['有限球动画','轨迹与出口',' Residence time ','聚焦与剪切',' Methanol exposure ','电压对照'],width=15,state='readonly',style='WB.TCombobox')
        combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        button(line,'从已有轨迹恢复原动画',app.restore_animation).pack(side='right')
        self.picture=Picture(original);self.picture.pack(fill='both',expand=True)
        label(original,'原动画按进入末端后的同龄响应对齐; 连续流页显示同一时刻的真实到达差异.',size=9,color=MUTED,
            wraplength=850).pack(fill='x',padx=15,pady=8)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self.refresh())
    def speed_changed(self,e=None):self.session.speed=float(self.speed.get().replace('×',''))
    def mode_changed(self,e=None):
        self.session.startup='steady' if self.mode.get()=='稳态连续运行' else 'empty'
        self.session.now=10. if self.session.startup=='steady' else 0.;self.session.selected_event=None;self.app.refresh()
    def duration_changed(self,e=None):
        s=self.session;s.duration_s=float(self.duration.get());s.now=min(s.now,s.duration_s);s.selected_event=None;s.revision+=1
        self.slider.configure(to=s.duration_s);self.app.refresh()
    def scrub(self,value):
        if not self._setting:self.session.now=float(value);self.draw()
    def pick(self,event_id):self.session.selected_event=event_id;self.draw()
    def next_focus(self):
        if self.session.data is None:return
        if stream_problem(self.session):self.app.message(stream_problem(self.session));return
        try:result=next_stage_event(self.session.process(),'inner_time',self.session.now)
        except ValueError as e:self.app.message(str(e));return
        if result:
            self.session.selected_event,self.session.now=result;self.session.speed=.05;self.speed.set('0.05×');self.session.playing=True;self.draw()
    def refresh(self):
        s=self.session;page=self.tabs.index(self.tabs.select())
        if s.data:
            if page==1:
                baseline=s.current_pin.data if s.current_pin else None;key=(id(s.data.fields),id(baseline.fields) if baseline else None,self.field.get())
                if key!=self.key:
                    try:draw_fields(self.field_plot,s.data,self.field.get(),baseline);self.key=key
                    except ValueError as e:self.app.message(str(e))
            if page==2:self.exposure()
            if page==3:
                names={'有限球动画':'cell_trajectories.gif','轨迹与出口':'figure_04_trajectories.png',' Residence time ':'figure_05_residence_events.png',
                    '聚焦与剪切':'figure_03_focusing_shear.png',' Methanol exposure ':'figure_06_methanol.png','电压对照':'figure_07_electric.png'}
                path=self.app.restored_animation if self.original.get()=='有限球动画' and self.app.restored_animation else Path(s.data.source)/names[self.original.get()]
                self.picture.load(path)
        self.draw()
    def exposure(self):
        s=self.session;data=s.data;key=(id(data.fields),s.selected_event,s.revision)
        if key==self.exposure_key:return
        if not getattr(s,'views_ready',True):return
        self.exposure_key=key
        m=None if stream_problem(s) else s.process()
        index=int(m.records['template_id'][s.selected_event]) if m is not None and s.selected_event is not None else data.default_cell
        tr=data.track(index);ax,other=self.exposure_plot.axes(2)
        t=(tr['t']-tr['inner_time'])*1000;mask=t>=-100
        ax.plot(t[mask],tr['C'][mask],color=PURPLE,lw=2);ax.set(xlabel='Time from inner lip (ms)',ylabel='Extracellular methanol',ylim=(0,1.05))
        ax.axvline(0,color=MUTED,lw=.7,ls='--')
        values=data.fields.get('response_residence_time_s',np.array([data.track(i)['end_time'] for i in range(len(data.ids))]))
        exited=data.fields.get('response_exited',np.ones(len(values),bool));other.hist(values[exited],bins=25,color='#accbdf',edgecolor=WHITE)
        other.set(xlabel='Full tube residence (s)',ylabel='Exited response count')
        self.exposure_plot.show();p=data.summary.get('particles',{})
        self.exposure_note.set(f'样本 {data.ids[index]}; 剪切剂量均值 {metric_string(p.get("mean_shear_dose_pa_s")," Pa·s")}; '
            f'平均近壁时间 {metric_string(p.get("mean_near_wall_time_s")," s")}.外液暴露与条件剪切剂量不直接等于生物损伤率.')
    def draw(self):
        s=self.session;self.play.configure(text=' Pause ' if s.playing else ' Play ')
        self.clock.set(f'共用观察时钟 t={s.now:.4f} s    ·    '+('观察起点前已持续预运行' if s.startup=='steady' else '从无细胞开始注入'))
        self._setting=True;self.slider.set(s.now);self._setting=False
        page=self.tabs.index(self.tabs.select())
        if page==0:self.stream.draw()
        elif page==3 and s.playing:self.picture.render(int(s.now*15))
        if s.data and s.selected_event is not None and not stream_problem(s):
            m=s.process();i=s.selected_event
            if i<len(m.records['event_id']):
                tr=m.event_track(i);a=m.records['injection_s'][i]
                self.inspector.set(f'#{i}   Injection  {a:.4f} s → 末端 {a+tr["bulk_time"]:.4f} s →  Sample outlet  {a+tr["inner_time"]:.4f} s →  Sheath outlet  {m.records["nozzle_s"][i]:.4f} s')
                if s.has_interactions() and i in s.interaction_result.event_ids:
                    h=s.interaction_result.event_history(i);j=int(np.searchsorted(h['time_s'],s.now,side='right')-1)
                    status=h['statuses'][max(0,j)] if h['time_s'] else 'not_entered'
                    self.inspector.set(self.inspector.get()+f'  [上述为基线]; 局部状态 {status}, 局部实际出管 '+
                        ('尚未解析' if h['exit_s'] is None else f'{h["exit_s"]:.6f} s'))
            else:s.selected_event=None;self.inspector.set('')
        else:self.inspector.set('点击细胞查看同一编号的全程时间链; 局部区域可能暂时没有细胞.')


class AcquisitionPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.vars={};self.muted=False;self.key=None;self.measured=None
        form=Card(self,' Acquisition method  · 即时更新','信号峰宽与输送比例为情景输入; 完整信号指离子包覆盖, 不指生物完整细胞.');form.pack(fill='x')
        grid=tk.Frame(form,bg=WHITE);grid.pack(fill='x',padx=12,pady=(0,10))
        p=self.session.acquisition
        specs=[('resolution',' Resolution ',self.session.resolution),('actual_it',' Actual  IT · ms',self.session.actual_it_ms),
            ('max_injection_time_ms','Maximum IT · ms',p.max_injection_time_ms),('microscans','Microscans',p.microscans),
            ('signal_width_ms','信号峰宽 · ms',p.signal_width_ms),('signal_delivery_fraction',' Exited → Packet delivery fraction ',p.signal_delivery_fraction),
            ('overhead_ms','时序开销 · ms',p.overhead_ms),('single_event_identification_efficiency',' Identification efficiency  · 未知留空',p.single_event_identification_efficiency)]
        for i,(key,title,value) in enumerate(specs):
            box=tk.Frame(grid,bg=WHITE);box.grid(row=i//4,column=i%4,padx=6,pady=5,sticky='ew');grid.columnconfigure(i%4,weight=1)
            label(box,title,size=9,color=MUTED).pack(anchor='w',pady=(0,3));v=tk.StringVar(value='' if value is None else str(value));self.vars[key]=v
            w=(ttk.Combobox(box,textvariable=v,values=list(TRANSIENT_MS),state='readonly',style='WB.TCombobox',width=12)
                if key=='resolution' else ttk.Entry(box,textvariable=v,style='WB.TEntry',width=12));w.pack(fill='x')
            v.trace_add('write',lambda *a:app.schedule_method())
        self.tabs=ttk.Notebook(self,style='WB.TNotebook');self.tabs.pack(fill='both',expand=True,pady=(10,0))
        gates=Card(self.tabs);self.tabs.add(gates,text='连续采集门')
        line=tk.Frame(gates,bg=WHITE);line.pack(fill='x',padx=12,pady=5)
        button(line,' Play  /  Pause ',app.toggle_play).pack(side='left')
        button(line,'下一次出管',self.next_outlet,primary=True).pack(side='left',padx=6)
        self.recovery=tk.StringVar();label(line,textvariable=self.recovery,size=9,color=MUTED,wraplength=570,justify='left').pack(side='left',padx=12)
        self.timeline=GateCanvas(gates,self.session);self.timeline.pack(fill='both',expand=True,padx=10,pady=5)
        button(gates,'查看逐细胞去向与覆盖 →',lambda:self.tabs.select(1)).pack(side='bottom',anchor='e',padx=14,pady=6,before=self.timeline)
        ledger=Card(self.tabs,' Per-cell acquisition ledger ','显示当前时钟前后 3 s 的模拟出管事件; 同一细胞跨谱去重.');self.tabs.add(ledger,text='逐细胞回收')
        row=tk.Frame(ledger,bg=WHITE);row.pack(fill='x',padx=12,pady=4)
        button(row,' Play  /  Pause ',app.toggle_play).pack(side='left')
        button(row,'下一次出管',self.next_outlet).pack(side='left',padx=5)
        box,self.table=table(ledger,['id','nozzle','status','coverage','spectra'],['细胞编号',' Exited  s','去重去向','纯信号覆盖','参与谱数'],
            [90,100,225,130,100],height=5);box.pack(fill='both',expand=True,padx=12,pady=(0,10))
        self.table.bind('<<TreeviewSelect>>',self.select_event)
        chart=Card(self.tabs);self.tabs.add(chart,text=' Resolution  / IT  Comparison ');self.plot=Plot(chart);self.plot.pack(fill='both',expand=True,padx=10,pady=10)
        actual=Card(self.tabs,'导入实际积累门','CSV 起点应为真实 accumulation_start_s; RAW 扫描时间不能直接替代.');self.tabs.add(actual,text='实测采集门')
        button(actual,' Import acquisition gates  CSV',self.import_gates,primary=True).pack(anchor='w',padx=16,pady=8)
        self.measured_text=tk.StringVar(value='尚未导入; 默认页使用周期 Full MS  Model settings .')
        label(actual,textvariable=self.measured_text,wraplength=800,justify='left').pack(fill='x',padx=16,pady=8)
        box,self.measured_table=table(actual,['start','it','pure','mixed'],['积累起点 s',' Actual  IT ms','纯来源概率','混合概率'])
        box.pack(fill='both',expand=True,padx=12,pady=10)
        button(actual,'导出采集门报告',self.export_gates).pack(anchor='w',padx=16,pady=8)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self.refresh());self.table_key=None
    def collect(self):
        p=deepcopy(self.session.acquisition)
        for key,var in self.vars.items():
            if key in ('resolution','actual_it'):continue
            value=var.get().strip()
            value=None if key=='single_event_identification_efficiency' and not value else int(value) if key=='microscans' else float(value)
            setattr(p,key,value)
        p.actual_injection_times_ms=[min(x,p.max_injection_time_ms) for x in p.actual_injection_times_ms]
        return p.validate(),int(self.vars['resolution'].get()),float(self.vars['actual_it'].get())
    def refresh(self):
        s=self.session
        if not s.data:return
        key=(id(s.data),s.revision)
        if key!=self.key:
            self.key=key;draw_resolution(self.plot,s,s.current_pin.data if s.current_pin else None)
            if stream_problem(s):self.recovery.set(stream_problem(s))
            else:
                m=s.process();counts=m.summary()['counts'];total=len(m.ledger)
                self.recovery.set(f'{s.duration_s:g}s 出管来源 {total}  Count : 完整纯 {counts["pure_complete"]}; 部分纯 {counts["pure_partial"]}; '
                    f' Mixing  {counts["mixed_only"]}; 错过门 {counts["missed"]}'+(f'; 瞬时纯 {counts["pure_impulse"]}' if counts['pure_impulse'] else '')
                    +(f'; 未送达 {counts["not_delivered"]}' if counts['not_delivered'] else ''))
            self.table_key=None
            if self.measured:self.update_measured()
        self.draw()
    def draw(self):
        if not self.session.data:return
        page=self.tabs.index(self.tabs.select())
        if page==0:self.timeline.draw();return
        if page!=1:return
        if stream_problem(self.session):self.table.delete(*self.table.get_children());return
        m=self.session.process();bucket=int(self.session.now*2)
        key=(id(m),bucket)
        if key==self.table_key:return
        self.table_key=key;self.table.delete(*self.table.get_children())
        rows=[r for r in m.ledger if self.session.now-3<=r['nozzle_s']<=self.session.now+3]
        for r in sorted(rows,key=lambda x:x['nozzle_s'])[:100]:
            self.table.insert('','end',iid=str(r['event_id']),values=[r['event_id'],f'{r["nozzle_s"]:.4f}',STATUS_LABELS[r['status']],
                '未定义' if r['pure_captured_fraction'] is None else f'{r["pure_captured_fraction"]:.1%}',r['scan_count']])
    def select_event(self,e=None):
        chosen=self.table.selection()
        if chosen:self.session.selected_event=int(chosen[0])
    def next_outlet(self):
        if not self.session.data:return
        if stream_problem(self.session):self.app.message(stream_problem(self.session));return
        m=self.session.process();r=m.records;eligible=np.flatnonzero(np.isfinite(r['nozzle_s'])&(r['nozzle_s']>self.session.now+1e-8))
        if len(eligible):
            i=eligible[np.argmin(r['nozzle_s'][eligible])];self.session.selected_event=int(i);self.session.now=float(r['nozzle_s'][i]);self.draw()
    def import_gates(self,path=None):
        from ..acquisition.measured_timing import load_gates
        path=path or filedialog.askopenfilename(parent=self.app.root,filetypes=[('CSV','*.csv')])
        if path:
            try:self.measured=(str(path),load_gates(path));self.update_measured()
            except (ValueError,OSError) as e:self.app.message(str(e))
    def update_measured(self):
        from ..acquisition.measured_timing import analyze_gates
        if not self.session.data or not self.measured:return
        p=self.session.acquisition
        if p.microscans!=1:
            self.measured_text.set('当前导入模型要求 Microscans=1.');self.measured_table.delete(*self.measured_table.get_children())
            if hasattr(self,'measured_result'):del self.measured_result
            return
        self.measured_result=analyze_gates(self.measured[1],self.session.metrics()['nozzle'],p.signal_width_ms,p.signal_delivery_fraction,
            p.single_event_identification_efficiency,source=self.measured[0])
        r=self.measured_result
        self.measured_text.set(f'已导入 {r["gate_count"]} 个门;  Inspect  {r["observation_duration_s"]:.4g} s.\n'
            f'理论去重纯来源 {r["unique_pure_opportunities_s"]:.4g}  Count /s; 非空谱纯度 {metric_string(r["aggregate_nonempty_single_cell_purity"])}.真实检出率仍未知.')
        self.measured_table.delete(*self.measured_table.get_children())
        for row in r['per_scan'][:1000]:self.measured_table.insert('','end',values=[row['accumulation_start_s'],row['actual_it_ms'],f'{row["pure_probability"]:.3%}',f'{row["mixed_probability"]:.3%}'])
    def export_gates(self):
        if hasattr(self,'measured_result'):
            from ..acquisition.measured_timing import export
            directory=self.app.new_directory('exports');export(self.measured_result,directory);self.app.message('采集门报告已导出: '+str(directory))
