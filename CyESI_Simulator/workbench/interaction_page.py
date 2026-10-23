"""Two-dimensional workbench view of the shared continuous particle physics."""
from copy import deepcopy
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np

from .widgets import BG,WHITE,INK,MUTED,TEAL,PURPLE,AMBER,Card,ScrollFrame,Plot,label,button
from ..particle.interactions import InteractionParameters,STATE_LABELS
from ..scene3d.data import _clean


class InteractionPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.key=None;self.seeded=None
        self.columnconfigure(1,weight=1);self.rowconfigure(0,weight=1)
        settings=ScrollFrame(self);settings.grid(row=0,column=0,sticky='nsew',padx=(0,10));settings.configure(width=315)
        settings.canvas.configure(width=300)
        settings.body.columnconfigure(1,weight=1)
        label(settings.body,' Axisymmetric fields  · 共用细胞物理',size=12,bold=True).grid(row=0,column=0,columnspan=2,sticky='w',pady=8)
        label(settings.body,'背景为真实二维已解场; 细胞按实际三维位置相互作用, 保留原连续编号.',
            size=9,color=MUTED,wraplength=290,justify='left').grid(row=1,column=0,columnspan=2,sticky='w',pady=5)
        defaults=InteractionParameters();self.vars={}
        inputs=[('start_s',' Start time  (s)'),('duration_s','时长 (s)'),('dt_s','步长上限 (s)'),
            ('contact_gap_um',' Contact gap  (µm)'),('cell_adhesion_on_rate_s',' Cell binding  k_on (1/s)'),
            ('wall_adhesion_on_rate_s',' Wall binding  k_on (1/s)'),('outlet_adhesion_on_rate_s',' Outlet binding  k_on (1/s)'),
            ('adhesion_off_rate_s',' Unbinding  k_off (1/s)'),('adhesion_force_scale_pn','力敏感尺度 (pN)'),
            ('neighbor_cutoff_radii','扰动范围 / 球半径')]
        for row,(key,title) in enumerate(inputs,2):
            label(settings.body,title,size=9,color=MUTED).grid(row=row,column=0,sticky='w',pady=4,padx=(0,7))
            var=tk.StringVar(value=f'{getattr(defaults,key):g}');self.vars[key]=var
            ttk.Entry(settings.body,textvariable=var,width=13).grid(row=row,column=1,sticky='ew',pady=4)
        row=len(inputs)+2;self.hydro=tk.BooleanVar(value=True)
        ttk.Checkbutton(settings.body,text=' Hydrodynamic coupling ',variable=self.hydro).grid(row=row,column=0,columnspan=2,sticky='w',pady=6)
        button(settings.body,' Use current time ',lambda:self.vars['start_s'].set(f'{self.session.now:.6f}')).grid(row=row+1,column=0,columnspan=2,sticky='ew')
        self.run=button(settings.body,'计算二维连续流相互作用',self.calculate,primary=True);self.run.grid(row=row+2,column=0,columnspan=2,sticky='ew',pady=6)
        button(settings.body,'播放已计算窗口',self.play).grid(row=row+3,column=0,columnspan=2,sticky='ew')
        button(settings.body,' Independent transport ',self.clear).grid(row=row+4,column=0,columnspan=2,sticky='ew',pady=6)
        line=tk.Frame(settings.body,bg=WHITE);line.grid(row=row+5,column=0,columnspan=2,sticky='ew')
        button(line,'保存局部研究',self.save).pack(side='left');button(line,'载入局部研究',self.load).pack(side='left',padx=5)
        label(settings.body,'黏附率默认 0; 非零值是未校准情景.近邻模型限于末端解析区, 长管仍用分段迁移与近壁捕获; 背景场为单向耦合.',
            size=9,color=AMBER,wraplength=290,justify='left').grid(row=row+6,column=0,columnspan=2,sticky='w',pady=10)
        self.info=tk.StringVar();label(settings.body,textvariable=self.info,size=9,color=MUTED,wraplength=290,
            justify='left').grid(row=row+7,column=0,columnspan=2,sticky='w')
        tabs=ttk.Notebook(self,style='WB.TNotebook');tabs.grid(row=0,column=1,sticky='nsew')
        plots=Card(tabs);tabs.add(plots,text='局部轨迹与计数');self.plot=Plot(plots);self.plot.pack(fill='both',expand=True)
        self.snapshot=tk.StringVar();label(plots,textvariable=self.snapshot,size=9,color=MUTED,wraplength=650,
            justify='left').pack(fill='x',padx=10,pady=8)
        signals=Card(tabs);tabs.add(signals,text='出管与采集影响')
        self.signals=tk.StringVar();label(signals,textvariable=self.signals,size=10,color=INK,wraplength=660,
            justify='left').pack(fill='x',padx=15,pady=15)
        self.table=ttk.Treeview(signals,columns=('id','time','baseline','current','capture'),show='headings',height=12)
        for name,title in zip(('id','time','baseline','current','capture'),('细胞编号','局部信号时刻 / s','基线分类','局部分类','纯信号比例')):
            self.table.heading(name,text=title);self.table.column(name,width=95,stretch=True)
        self.table.pack(fill='both',expand=True,padx=10,pady=10)
        charging=Card(tabs);tabs.add(charging,text='细胞带电 /  Current ')
        self.charging=tk.StringVar();label(charging,textvariable=self.charging,size=10,color=INK,wraplength=660,
            justify='left').pack(fill='x',padx=16,pady=15)
        charge_inputs=tk.Frame(charging,bg=WHITE);charge_inputs.pack(fill='x',padx=16)
        self.capacitance=tk.StringVar(value='.01');self.cytosol=tk.StringVar(value='.5')
        for text,var in [('膜电容 (F/m²)',self.capacitance),(' Intracellular conductivity  (S/m)',self.cytosol)]:
            label(charge_inputs,text,size=9,color=MUTED).pack(side='left',padx=(0,6))
            ttk.Entry(charge_inputs,textvariable=var,width=10).pack(side='left',padx=(0,14))
        button(charging,'计算选中细胞的带电与极化',self.charge).pack(anchor='w',padx=16,pady=10)
        self.charge_info=tk.StringVar(value='在运输页选中细胞后计算.ζ 等效电荷, 膜极化电荷与真实结构净电荷分别解释.')
        label(charging,textvariable=self.charge_info,size=10,color=MUTED,wraplength=660,justify='left').pack(fill='x',padx=16,pady=12)
        self.tabs=tabs
        for box in (plots,signals,charging):
            box.bind('<Configure>',lambda e,b=box:[w.configure(wraplength=max(200,e.width-32))
                for w in b.winfo_children() if isinstance(w,tk.Label)])

    def calculate(self):
        try:p=InteractionParameters(**{k:float(v.get()) for k,v in self.vars.items()},hydrodynamics=self.hydro.get()).validate()
        except (ValueError,TypeError) as e:self.app.message(str(e));return
        frozen=self.app.frozen_session();key=frozen.interaction_source_key()
        path=self.app.new_directory('interactions')
        def work(notify):
            scene,result=frozen.compute_interactions(p,notify)
            frozen.accept_interactions(scene,result,key)
            from .interaction_storage import save_study
            notify('保存独立二维相互作用记录');save_study(frozen,path)
            return scene,result
        def done(payload):
            scene,result=payload
            if not self.session.accept_interactions(scene,result,key):self.app.message('来源已改变, 过期相互作用结果未应用');return
            self.key=None;self.app.refresh();self.app.message('二维相互作用已完成, 流场已复用; 保存至 '+str(path))
        self.app.run_task('共用物理模块 · 二维连续流相互作用',work,done)

    def clear(self):self.session.clear_interactions();self.key=None;self.app.refresh()
    def play(self):
        s=self.session
        if not s.has_interactions():self.app.message('先计算或载入相互作用研究');return
        s.now=float(s.interaction_result.time_s[0]);s.speed=.1;s.playing=True
        self.app.navigate('transport')
        self.app.pages['transport'].speed.set('0.1×')

    def refresh(self):
        s=self.session;self.run.configure(state='disabled' if self.app.busy or s.data is None or s.pending else 'normal')
        if s.data is None:return
        if self.seeded!=s.interaction_source_key() and s.peek_process() is not None:
            self.seeded=s.interaction_source_key();m=s.process();i=m.default_event
            if i is not None and np.isfinite(m.records['nozzle_s'][i]):self.vars['start_s'].set(f'{max(0.,m.records["nozzle_s"][i]-.1):.6f}')
        from ..scene3d.current import estimate_current
        current=estimate_current(s.data.config)
        self.charging.set(f'与三维共用 I=α√(γKQ) 条件估算: {current["predicted_current_ua"]:.4g} µA.\n'
            '给定电流范围 0.2–1 µA, 当前估算'+('在范围内.\n' if current['within_user_range'] else '不在范围内.\n')+
            '仅适用于已建立锥射流的情景;  Conductivity , ζ 电势和膜参数需校准.\n电泳已进入运输力学, 电荷诊断不会重复添加 qE 力.')
        result=s.interaction_result if s.has_interactions() else None
        token=(id(result),s._method_key,s.resolution,s.actual_it_ms)
        if token!=self.key:
            self.key=token;self.table.delete(*self.table.get_children())
            ax,count=self.plot.axes(2)
            if result is None:
                ax.text(.5,.5,'Compute or load a local continuous study',ha='center',transform=ax.transAxes)
                self.info.set('二维与三维共用近邻扰动,  Exclusion , 黏连及脱附方程.切换页面与播放不重算物理.')
                self.signals.set('等待局部研究.全程指标仍显示独立运输基线, 局部事件的实际出管和采集影响在此对照.')
            else:
                for j,eid in enumerate(result.event_ids):
                    valid=result.states[:,j]!=0;xyz=result.positions_m[valid,j]
                    ax.plot(xyz[:,2]*1e3,np.hypot(xyz[:,0],xyz[:,1])*1e6,lw=1,label=f'#{eid}')
                    bound=np.isin(result.states[valid,j],[2,3,4])
                    if bound.any():ax.scatter(xyz[bound,2]*1e3,np.hypot(xyz[bound,0],xyz[bound,1])*1e6,s=14,color='#d44a91')
                ax.set(xlabel='z (mm)',ylabel='r (um)',title='Actual local finite-sphere paths')
                times=result.time_s-result.time_s[0]
                for code,name in STATE_LABELS.items():
                    if code and np.any(result.states==code):count.step(times,(result.states==code).sum(axis=1),where='post',label=name)
                count.set(xlabel='Local elapsed time (s)',ylabel='Cells');count.legend(fontsize=7)
                d=result.diagnostics
                self.info.set(f' Cell  {d["event_count"]};  Exited  {d["exited_count"]}; 仍在域 {d["retained_at_end"]}\n'
                    f'细胞对接触 {d["pair_contacts"]};  Wall  / 出口接触 {d["wall_contacts"]} / {d["outlet_contacts"]}\n'
                    f'形成黏连 /  Wall  / 出口键 {d["pair_bonds_formed"]} / {d["wall_bonds_formed"]} / {d["outlet_bonds_formed"]};  Unbinding  {d["bonds_released"]}\n'
                    f'最大扰动 {d["max_pair_disturbance_m_s"]*1000:.5g} mm/s; 守恒残差 {d["mass_balance_residual"]}')
                analysis=s.interaction_acquisition();c=analysis['counts']
                self.signals.set(f'实际完成窗口 {analysis["duration_s"]:.5g} s:  Exited  {analysis["local_exit_count"]}  Count , '
                    f'完整纯来源包 {c["pure_complete"]}, 部分纯包 {c["pure_partial"]},  Mixing  {c["mixed_only"]}, 错过采集 {c["missed"]}, 边界待解析 {c["boundary_unresolved"]}.\n'
                    '按实际相互作用出管时刻重新匹配采集门; 未出管, 未完成与黏附分别统计.初始居民参与计数, 因此不把短窗出管数 / 窗长称为整体回收率.信号包与输送仍为条件假设, 顶部总体理论率保留基线.')
                for r in analysis['ledger']:
                    self.table.insert('', 'end',values=(r['event_id'],f'{r["packet_onset_s"]:.6f}',r['baseline_status'],r['status'],
                        '—' if r['pure_captured_fraction'] is None else f'{r["pure_captured_fraction"]:.1%}'))
            self.plot.show()
        self.draw()

    def draw(self):
        s=self.session
        if s.active_interactions():
            from ..particle.interactions import STATE_LABELS
            r=s.interaction_result;k=int(np.clip(np.searchsorted(r.time_s,s.now,side='right')-1,0,len(r.time_s)-1))
            values={name:int(np.sum(r.states[k]==code)) for code,name in STATE_LABELS.items()}
            self.snapshot.set(f't={s.now:.6f} s · 当前共同时间窗: '+str(values))
        else:self.snapshot.set('当前不在已计算的局部相互作用窗口内.运输页在窗口内显示修正后的细胞位置.')

    def charge(self):
        s=self.session
        if s.data is None or s.pending:return
        from ..scene3d.cell_charge import ChargeParameters
        try:parameters=ChargeParameters(membrane_capacitance_f_m2=float(self.capacitance.get()),cytosol_conductivity_s_m=float(self.cytosol.get())).validate()
        except (ValueError,TypeError) as e:self.app.message(str(e));return
        frozen=self.app.frozen_session();source_key=frozen.interaction_source_key()
        model=frozen.process();eid=s.selected_event if s.selected_event is not None else model.default_event
        if eid is None:self.charge_info.set('当前浓度下没有细胞事件');return
        def work(notify):
            from ..particle.axisymmetric_interactions import AxisymmetricScene
            from ..scene3d.cell_charge import charge_at_track
            scene=frozen.interaction_scene if frozen.has_interactions() else AxisymmetricScene(frozen.data,model)
            h=charge_at_track(scene,int(model.records['template_id'][eid]),parameters);return h,scene.field_provider.salt_scope
        def done(payload):
            if s.interaction_source_key()!=source_key:self.app.message('来源已改变, 旧带电诊断未覆盖当前结果');return
            h,scope=payload;j=-1
            self.charge_info.set(f' Cell  #{eid} 独立运输基线最后状态: \nζ  Surface-equivalent charge  {h["q_zeta_c"][j]*1e12:.5g} pC; '
                f' Electrophoretic charge  {h["q_electrophoretic_effective_c"][j]*1e15:.5g} fC\n'
                f'诱导正 / 负极电荷 ±{h["positive_pole_charge_c"][j]*1e15:.5g} fC, 诱导净和 0\n'
                f'膜极点电压 {h["membrane_pole_voltage_v"][j]*1e3:.5g} mV\n盐场来源: {scope}\n'
                '真实结构净电荷未知; 相互作用改轨后的充电不直接复用该基线历史.')
        self.app.run_task('复用二维场 · 共用带电模型',work,done)

    def save(self):
        if not self.session.has_interactions():return
        from .interaction_storage import save_study
        path=self.app.new_directory('interactions');frozen=self.app.frozen_session()
        self.app.run_task('保存二维局部研究',lambda notify:save_study(frozen,path),lambda path:self.app.message('已保存 '+str(path)))

    def load(self):
        path=filedialog.askdirectory(parent=self.app.root,title='选择二维相互作用研究文件夹')
        if not path:return
        from .interaction_storage import load_study
        frozen=self.app.frozen_session();key=frozen.interaction_source_key()
        def work(notify):
            load_study(frozen,Path(path));return frozen.interaction_scene,frozen.interaction_result
        def done(payload):
            if self.session.accept_interactions(*payload,key):self.key=None;self.app.refresh()
            else:self.app.message('来源已改变, 已载入研究未覆盖当前结果')
        self.app.run_task('读取二维局部研究',work,done)
