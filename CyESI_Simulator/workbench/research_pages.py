from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np
from .widgets import *
from .pages import draw_fields,metric_string
from .session import catalog
from ..acquisition.q_exactive import timing,occupancy,TRANSIENT_MS


class SprayPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.phase_play=False;self.phase_value=1.;self.phase_key=None
        self.flight_result=None;self.flight_source=None;self.flight_vars={}
        self.tabs=ttk.Notebook(self,style='WB.TNotebook');self.tabs.pack(fill='both',expand=True)
        phase=Card(self.tabs,'出口两相 / 气场','回放已保存的短时两相解; 尚未验证稳定喷雾.此页使用喷口自身的 ms 时间.');self.tabs.add(phase,text=' Emitter  / 气场 /  Droplets ')
        bar=tk.Frame(phase,bg=WHITE);bar.pack(fill='x',padx=12,pady=8)
        button(bar,' Play  / 暂停两相回放',self.toggle_phase).pack(side='left')
        self.phase_field=tk.StringVar(value=' Gas velocity ')
        selector=ttk.Combobox(bar,textvariable=self.phase_field,values=[' Gas velocity ',' Electric field magnitude ','自由电荷',' Potential ',' Methanol ',' Salt '],width=10,state='readonly')
        selector.pack(side='left',padx=8);selector.bind('<<ComboboxSelected>>',lambda e:self.draw_phase())
        button(bar,'完整装置 / 喷口可视化',lambda:app.advanced_tool('explorer')).pack(side='right')
        self.phase_clock=tk.StringVar();label(bar,textvariable=self.phase_clock,size=9,color=MUTED).pack(side='left',padx=15)
        self.phase_slider=ttk.Scale(phase,from_=0,to=1,command=self.phase_scrub);self.phase_slider.pack(fill='x',padx=12)
        self.phase_plot=Plot(phase);self.phase_plot.pack(fill='both',expand=True,padx=8,pady=8)
        self.phase_note=tk.StringVar();label(phase,textvariable=self.phase_note,size=9,color=MUTED,wraplength=850,
            justify='left').pack(fill='x',padx=15,pady=10)
        self.phase_slider.set(1.)
        inlet=Card(self.tabs,'脂质膜细胞与载体微滴','所选保存响应 + 条件膜 /  Carrier  / 吸气参数; 单个案例不构成总体入口回收率.');self.tabs.add(inlet,text=' Inlet  /  Cell state ')
        p=self.session.state_parameters;grid=tk.Frame(inlet,bg=WHITE);grid.pack(fill='x',padx=10,pady=(0,8))
        specs=[('inlet_diameter_mm','入口孔径 mm',p.inlet_diameter_mm),('inlet_wall_temperature_c',' Wall temperature  °C',p.inlet_wall_temperature_c),
            ('drop_radii_um','载体半径 μm',20.),('charge_fractions','q / Rayleigh',.15),
            ('methanol_permeabilities_m_s',' Membrane permeability  m/s',1e-6),('dissolution_rates_s','膜溶解 k /s',20.),
            ('flow_factors','吸气倍率',1.),('alignment_offset_um','横向偏移 μm',0.)]
        for i,(key,title,value) in enumerate(specs):
            cell=tk.Frame(grid,bg=WHITE);cell.grid(row=i//4,column=i%4,sticky='ew',padx=6,pady=5);grid.columnconfigure(i%4,weight=1)
            label(cell,title,size=9,color=MUTED).pack(anchor='w',pady=(0,3));var=tk.StringVar(value=str(value));self.flight_vars[key]=var
            ttk.Entry(cell,textvariable=var,width=12,style='WB.TEntry').pack(fill='x')
            var.trace_add('write',lambda *a:self.invalidate_flight())
        bar=tk.Frame(inlet,bg=WHITE);bar.pack(fill='x',padx=14,pady=5)
        self.flight_button=button(bar,'计算选中细胞入孔情景',self.compute_flight,primary=True);self.flight_button.pack(side='left')
        button(bar,'完整膜 / 入口情景研究',lambda:app.advanced_tool('state')).pack(side='right')
        self.flight_text=tk.StringVar(value='在运输页面选中细胞后, 可计算其条件入口状态.暂无选择时采用保存的代表响应.')
        label(inlet,textvariable=self.flight_text,size=9,color=MUTED,wraplength=850,justify='left').pack(fill='x',padx=16,pady=8)
        self.flight_plot=Plot(inlet);self.flight_plot.pack(fill='both',expand=True,padx=8,pady=8)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self.refresh())
    def toggle_phase(self):self.phase_play=not self.phase_play
    def phase_scrub(self,value):self.phase_value=float(value);self.draw_phase()
    def collect(self):
        p=deepcopy(self.session.state_parameters)
        for key,var in self.flight_vars.items():
            value=float(var.get());setattr(p,key,[value] if isinstance(getattr(p,key),list) else value)
        return p.validate()
    def invalidate_flight(self):
        self.flight_result=None;self.flight_text.set('入口情景参数已修改; 应用后复用同条件的计算结果.')
        if hasattr(self,'flight_plot'):self.flight_plot.figure.clear();self.flight_plot.show()
    def refresh(self):
        if self.flight_source is not None and self.session.data is not None and self.flight_source!=id(self.session.data.fields):self.invalidate_flight()
        if self.tabs.index(self.tabs.select())==0:self.draw_phase()
    def draw(self):
        if self.tabs.index(self.tabs.select())==0:self.draw_phase()
    def draw_phase(self):
        data=self.session.data
        if data is None or data.spray is None:
            self.phase_clock.set('此结果没有喷口两相记录')
            self.phase_note.set('可先观察载液与电场; 启用两相后计算新的出口记录.已有内容与模型限制保留.')
            if self.phase_key is not None:
                self.phase_key=None;self.phase_plot.figure.clear();self.phase_plot.show()
            return
        sp=data.spray;n=len(sp['time_s']);frame=int(np.clip(round(self.phase_value*(n-1)),0,n-1));key=(id(sp),frame,self.phase_field.get())
        if key==self.phase_key:return
        self.phase_key=key;phi=sp['liquid_fraction'][frame];active=sp.get('active',np.ones_like(phi,bool))
        a,b=self.phase_plot.axes(2);re=sp['radial_edges_m'];ze=sp['axial_edges_m']
        im=a.pcolormesh(ze*1e3,re*1e6,np.ma.masked_where(~active,phi),vmin=0,vmax=1,cmap='Blues',shading='flat')
        a.set(xlabel='z (mm)',ylabel='r (um)',title='Saved gas-liquid interface');self.phase_plot.figure.colorbar(im,ax=a,label='Liquid fraction',shrink=.9)
        ur=sp.get('velocity_r_history_m_s');uz=sp.get('velocity_z_history_m_s')
        ur=ur[frame] if ur is not None else sp.get('velocity_r_m_s',np.zeros_like(phi))
        uz=uz[frame] if uz is not None else sp.get('velocity_z_m_s',np.zeros_like(phi))
        field=self.phase_field.get();value=None;unit='';title=''
        if field==' Gas velocity ':value=np.ma.masked_where((phi>=.5)|~active,np.hypot(ur,uz));unit='Gas speed (m/s)';title='Saved gas velocity'
        elif field==' Electric field magnitude ' and 'field_r_v_m_history' in sp:
            value=np.hypot(sp['field_r_v_m_history'][frame],sp['field_z_v_m_history'][frame]);unit='E (V/m)';title='Coupled moving-interface electric field'
        elif field=='自由电荷' and 'charge_density_c_m3_history' in sp:
            value=np.ma.masked_where(~active,sp['charge_density_c_m3_history'][frame]);unit='Free charge (C/m3)';title='Conserved free charge'
        elif field==' Potential ' and 'potential_v_history' in sp:value=sp['potential_v_history'][frame];unit='Potential (V)';title='Coupled electric potential'
        elif field==' Methanol ' and 'methanol_fraction_history' in sp:
            value=np.ma.masked_where((phi<1e-8)|~active,sp['methanol_fraction_history'][frame]);unit='Methanol fraction';title='Liquid methanol inventory transport'
        elif field==' Salt ' and 'salt_mol_m3_history' in sp:
            value=np.ma.masked_where((phi<1e-8)|~active,sp['salt_mol_m3_history'][frame]);unit='Salt (mM)';title='Liquid salt and conductivity coupling'
        if value is None:b.text(.5,.5,'Archived result has no coupled charge history;\ncompute a new charged two-phase result.',ha='center',transform=b.transAxes)
        else:
            limits={}
            if field=='自由电荷':
                span=max(float(np.max(np.abs(value))),1e-30);limits={'vmin':-span,'vmax':span}
            im=b.pcolormesh(ze*1e3,re*1e6,value,cmap='coolwarm' if field=='自由电荷' else 'viridis',shading='flat',**limits)
            self.phase_plot.figure.colorbar(im,ax=b,label=unit,shrink=.9)
        b.set(xlabel='z (mm)',ylabel='r (um)',title=title);self.phase_plot.show()
        t=float(sp['time_s'][frame]);droplets=data.droplet_frame(t)
        self.phase_clock.set(f'喷口时间 {t*1000:.5g} ms · 帧 {frame+1}/{n}')
        resolved=[d for d in droplets if d.get('resolved')]
        self.phase_note.set(f'当前脱落连通体 {len(droplets)}  Count , 其中达到空间解析条件 {len(resolved)}  Count .'
            '锥形, 射流与滴分裂只显示求解结果; 短时记录和少量连通体不能解释为稳定 Taylor 锥或实测滴径分布.')
        current=sp.get('supply_current_a_history');node=sp.get('inlet_voltage_v_history')
        if current is not None:
            text='初始供电电流未定义' if not np.isfinite(current[frame]) else f'总供电电流 {current[frame]*1e6:.5g} µA ( Includes displacement current )'
            self.phase_note.set(self.phase_note.get()+f' {text}; 入口电位 {node[frame]:.5g} V.支持电导率为未校准假设.')
    def compute_flight(self):
        s=self.session
        if s.data is None:self.app.message('请先载入运输结果.');return
        if not getattr(s,'views_ready',True):self.app.message('连续源正在更新, 完成后可选择出管细胞.');return
        if getattr(s,'process_error',None):self.app.message(s.process_error);return
        try:
            p=self.collect();model=s.process();event=s.selected_event if s.selected_event is not None else model.default_event
            if event is None:raise ValueError('零浓度或没有可用事件; 请选择已出管响应.')
            source=id(model.data.fields)
        except (ValueError,KeyError,TypeError) as error:self.app.message(str(error));return
        def complete(result):
            if s.data is None or id(s.data.fields)!=source:return
            self.flight_source=source;self.flight_result=result;self.show_flight()
        self.app.run_task('计算所选响应的膜状态与条件飞行',lambda notify:s.flight(event,p,model),complete)
    def show_flight(self):
        f=self.flight_result;r=f['result'];history=f.get('history',[])
        self.flight_text.set(f'计算时事件 #{f["event_id"]} / 保存响应 {f["transport_sample_id"]}; 终态 {r["status"]}.'
            f'到入口平面: {r.get("arrived",False)};  Transfer  {metric_string(r.get("flight_time_ms")," ms")}; '
            f'终点温度 {metric_string(r.get("final_temperature_c")," °C")}.壁温不等于细胞温度; 真实完整率 / 检出率未知.')
        if not history:
            self.flight_plot.figure.clear();ax=self.flight_plot.figure.add_subplot();ax.axis('off')
            ax.text(.5,.5,'No resolved flight history\n'+str(r['status']),ha='center',va='center',color=MUTED)
            self.flight_plot.show();return
        axes=self.flight_plot.axes(3);t=np.array([h['time_after_lip_s'] for h in history])*1000
        values=[('z_m',1000,'z (mm)'),('drop_radius_m',1e6,'Carrier radius (um)'),('membrane_lipid_remaining_fraction',1.,'Membrane lipid fraction')]
        for ax,(key,factor,title) in zip(axes,values):
            ax.plot(t,[h.get(key,np.nan)*factor for h in history],color=TEAL,lw=2)
            ax.set(xlabel='Flight time (ms)',ylabel=title)
        self.flight_plot.show()


class ComparePage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.key=None;self.archive=None
        bar=Card(self);bar.pack(fill='x',pady=(0,10));line=tk.Frame(bar,bg=WHITE);line.pack(fill='x',padx=12,pady=10)
        button(line,'将当前结果保存为基准',app.pin,primary=True).pack(side='left')
        self.baseline=tk.StringVar(value='尚未设置');self.baseline_selector=ttk.Combobox(line,textvariable=self.baseline,
            values=[],state='readonly',style='WB.TCombobox',width=12);self.baseline_selector.pack(side='left',padx=15)
        self.baseline_selector.bind('<<ComboboxSelected>>',self.select_baseline)
        button(line,'清除基准',self.clear_pin).pack(side='right')
        self.tabs=ttk.Notebook(self,style='WB.TNotebook');self.tabs.pack(fill='both',expand=True)
        metrics=Card(self.tabs,' Current  vs  Reference ','下方使用同一套当前采集方法比较; 源物理参数分别保留.');self.tabs.add(metrics,text='指标与变化')
        box,self.table=table(metrics,['metric','baseline','current','delta'],[' Metrics ',' Reference ',' Current ','变化'],[170,140,140,160],height=8);box.pack(fill='both',expand=True,padx=12,pady=8)
        fields_card=Card(self.tabs,'共同色标的场对比','只比较已计算的数值场; 浓度变化不改变稀相载液场.');self.tabs.add(fields_card,text='数值场对比')
        row=tk.Frame(fields_card,bg=WHITE);row.pack(fill='x',padx=12,pady=5)
        self.field=tk.StringVar(value=' Methanol ');combo=ttk.Combobox(row,textvariable=self.field,values=['流速',' Pressure ',' Methanol ',' Electric field ',' Shear '],
            width=12,state='readonly',style='WB.TCombobox');combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        label(row,'共同色标; 变更物理条件需先完成计算.',size=9,color=MUTED).pack(side='left',padx=10)
        self.fields=Plot(fields_card);self.fields.pack(fill='both',expand=True,padx=8,pady=8)
        curves=Card(self.tabs,' Concentration  ×  Resolution  · 即时理论曲线','固定已解 Q 和运输比例; 仅改变 Poisson 输入强度与采集设置, 不假造新流场.');self.tabs.add(curves,text='实时统计敏感性')
        self.curves=Plot(curves);self.curves.pack(fill='both',expand=True,padx=10,pady=10)
        archives=Card(self.tabs,'已有专题研究','直接查看已完成的尺寸与长管研究; 另建研究时保留其原功能和参数.');self.tabs.add(archives,text=' Size  / 长管研究')
        row=tk.Frame(archives,bg=WHITE);row.pack(fill='x',padx=12,pady=8)
        button(row,'查看已有四尺寸结果',self.load_sizes,primary=True).pack(side='left')
        button(row,'新建尺寸研究',lambda:app.advanced_tool('size')).pack(side='left',padx=6)
        button(row,' Length  /  Orientation  / 剪切研究',lambda:app.advanced_tool('length')).pack(side='right')
        self.archive_text=tk.StringVar();label(archives,textvariable=self.archive_text,size=9,color=MUTED,wraplength=850,justify='left').pack(fill='x',padx=15,pady=5)
        row=tk.Frame(archives,bg=WHITE);row.pack(fill='x',padx=12,pady=5)
        self.archive_names=dict(zip(['总览','长管轨迹','出口聚焦','时间与暴露','采集率','条件入口状态','布朗运动对照'],
            ['size_01_overview','size_02_long_tube','size_03_focusing','size_04_transit_exposure','size_05_acquisition','size_06_conditional_inlet','size_07_brownian_control']))
        self.archive_figure=tk.StringVar(value='总览');combo=ttk.Combobox(row,textvariable=self.archive_figure,
            values=list(self.archive_names),
            state='readonly',width=26,style='WB.TCombobox');combo.pack(side='left');combo.bind('<<ComboboxSelected>>',lambda e:self.show_archive())
        self.picture=Picture(archives);self.picture.pack(fill='both',expand=True)
        self.tabs.bind('<<NotebookTabChanged>>',lambda e:self.refresh())
    def clear_pin(self):self.session.pins=[];self.session.revision+=1;self.app.refresh()
    def select_baseline(self,e=None):
        s=self.session;pin=next((p for p in s.pins if p.name==self.baseline.get()),None)
        if pin is not None:s.pins.remove(pin);s.pins.append(pin);s.revision+=1;self.app.refresh()
    def refresh(self):
        s=self.session;pin=s.current_pin;self.baseline.set(pin.name if pin else '尚未设置');self.baseline_selector.configure(values=[p.name for p in s.pins])
        if not s.data:return
        key=(id(s.data),id(pin.data) if pin else None,s.revision,self.field.get())
        if key==self.key:return
        self.key=key;current=s.metrics();old=s.metrics(pin.data) if pin else None;self.table.delete(*self.table.get_children())
        titles={'injection':' Input rate  ·  Count /s','nozzle':' Predicted outlet rate  ·  Count /s','transit':'已出管平均运输 · s','r95':'出口球心 r95 · μm',
            'pure':'理论去重纯来源 ·  Count /s','purity':'非空谱纯度','transport':'模型运输比例'}
        for metric,title in titles.items():
            a=None if old is None else old[metric];b=current[metric]
            delta='—' if a is None or b is None else f'{b-a:+.4g}'
            self.table.insert('','end',values=[title,metric_string(a),metric_string(b),delta])
        try:draw_fields(self.fields,s.data,self.field.get(),pin.data if pin else None)
        except ValueError as error:self.app.message(str(error))
        self.draw_curves()
    def draw_curves(self):
        s=self.session;data=s.data;p=s.acquisition;c=data.config;fraction=s.metrics()['transport']
        if fraction is None:return
        concentrations=np.geomspace(1e3,1e7,100);rates=concentrations*c.fluid.sample_flow_ul_min/60000*fraction
        ax=self.curves.axes()[0]
        for resolution,color in zip(TRANSIENT_MS,[TEAL,BLUE,PURPLE,'#c09646']):
            values=[occupancy(rate*p.signal_delivery_fraction,timing(resolution,s.actual_it_ms,p))['unique_pure_single_cell_events_s']*60 for rate in rates]
            ax.plot(concentrations,values,label=f'R={resolution:,}',color=color,lw=2)
        ax.axvline(c.cell.concentration_per_ml,color=MUTED,ls='--',lw=1)
        ax.set(xscale='log',xlabel='Concentration (cells/mL), fixed solved flow',ylabel='Unique pure opportunities / min')
        ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);self.curves.show()
    def load_sizes(self,directory=None):
        if directory is None:
            files=sorted(Path('outputs/size_studies').glob('*/size_summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
            directory=files[0].parent if files else filedialog.askdirectory(parent=self.app.root,title='选择尺寸研究结果')
        if not directory:return
        path=Path(directory)
        try:
            summary=json.loads((path/'size_summary.json').read_text(encoding='utf-8'));self.archive=path;c=summary['config']
            self.archive_text.set(f'保存研究: {path.name}; {c["geometry"]["sample_length_mm"]/10:g} cm, '
                f'Q={c["fluid"]["sample_flow_ul_min"]:.5g} μL/min, C={c["cell"]["concentration_per_ml"]:g}/mL.与当前配置分别保留.')
            self.show_archive()
        except (ValueError,OSError,KeyError) as e:self.app.message(str(e))
    def show_archive(self):
        if self.archive:self.picture.load(self.archive/(self.archive_names[self.archive_figure.get()]+'.png'))


class LibraryPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.key=None
        card=Card(self,'结果资料库','只读取结果元信息; 选择后载入数值数组.查看和对比已有结果不启动物理求解.');card.pack(fill='both',expand=True)
        row=tk.Frame(card,bg=WHITE);row.pack(fill='x',padx=12,pady=8)
        button(row,'刷新列表',self.scan).pack(side='left')
        button(row,'载入选中结果',self.load_selected,primary=True).pack(side='left',padx=6)
        button(row,'导出工作台报告',app.export_report).pack(side='right')
        box,self.table=table(card,['name','length','size','flow','concentration','voltage','spray'],
            ['结果目录',' Length  cm','粒径 μm',' Actual  Q μL/min',' Concentration  /mL',' Voltage  kV',' Two-phase '],[235,90,90,115,125,85,75],height=10)
        box.pack(fill='both',expand=True,padx=12,pady=8);self.table.bind('<Double-1>',lambda e:self.load_selected())
        row=tk.Frame(card,bg=WHITE);row.pack(fill='x',padx=12,pady=8)
        button(row,'打开当前源报告',app.open_source_report).pack(side='left')
        button(row,'保留的旧版完整界面',app.open_legacy).pack(side='left',padx=6)
        button(row,' Long capillary  / 全过程研究',lambda:app.advanced_tool('process')).pack(side='right')
        self.notes=tk.StringVar();label(card,textvariable=self.notes,size=9,color=MUTED,wraplength=850,justify='left').pack(fill='x',padx=16,pady=12)
        self.scan()
    def scan(self):
        self.rows=catalog();self.table.delete(*self.table.get_children())
        for i,r in enumerate(self.rows):self.table.insert('','end',iid=str(i),values=[r['name'],r['length_cm'],r['diameter_um'],
            f'{r["flow_ul_min"]:.5g}',f'{r["concentration"]:g}',r['voltage_kv'],'有' if r['spray'] else ' None '])
    def load_selected(self):
        selected=self.table.selection()
        if selected:self.app.load_result(self.rows[int(selected[0])]['path'])
    def refresh(self):
        s=self.app.session;cache=s.cache_summary()
        self.notes.set(f' Current source : {s.data.source if s.data else "尚未载入"}\n'
            f'会话复用: 文件命中 {cache["files"].get("hit",0)}; 过程模型命中 {cache["process"].get("hit",0)}; '
            f'采集分析命中 {cache["acquisition"].get("hit",0)}.\n'
            'See README.md for installation, usage and acquisition-model scope.')
