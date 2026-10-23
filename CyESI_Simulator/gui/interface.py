"""Native Tk GUI. Worker computes; all widgets are updated on the Tk thread."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageSequence
from ..config import Config


PANELS = ["流速 /  Pressure ", " Focusing  /  Shear ", "细胞轨迹", " Methanol exposure ", " Electric field ", "喷口两相", " Residence time ", "几何", " Parameter sweep "," Gas velocity  /  Pressure "]
FILE_MAP = {0: 1, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7, 6: 4, 7: 0, 8: 8,9:9}
INPUTS = {
    "Geometry / 几何": [("geometry", "inner_id_um", " Sample capillary  ID (μm)"), ("geometry", "inner_od_um", " Sample capillary  OD (μm)"),
        ("geometry", "sheath_id_um", " Sheath capillary  ID (μm)"), ("geometry", "sheath_od_um", " Sheath capillary  OD (μm)"),
        ("geometry", "gas_id_um", " Gas capillary  ID (μm)"), ("geometry", "sample_length_mm", "内管流经长度 (mm)"),
        ("geometry", "inner_recess_mm", " Sample recess  (mm)"), ("geometry", "gas_recess_mm", " Gas recess  (mm)"),
        ("geometry", "ms_gap_mm", "中管口至 MS (mm)")],
    "Supply / 供液与摆放": [("supply", "sample_control", "样品供液模式"),
        ("supply", "sample_pressure_mbar", "样品表压 (mbar)"),
        ("supply", "orientation", "内管摆放方向"),
        ("supply", "reservoir_head_mm", "液面高于入口 (mm)")],
    "Flow / 流动": [("fluid", "sample_flow_ul_min", " Sample flow  (μL/min)"),
        ("fluid", "sheath_flow_ul_min", " Sheath flow  (μL/min)"), ("fluid", "gas_pressure_kpa", "N₂ 压差 (kPa)"),
        ("chemistry", "sheath_methanol_fraction", " Methanol volume fraction  (0–1)")],
    "Electrical /  Electric field ": [("electrical", "voltage_kv", "上游电极电压 (kV)"),
        ("electrical", "contact_distance_mm", "加电点至中管口 (mm)"),
        ("electrical", "spray_current_na", " Spray current  (nA, 空= Unknown )"),
        ("electrical", "sheath_conductivity_s_m", "鞘液电导率 (S/m)"),
        ("electrical", "quartz_zeta_mv", "石英壁面 Zeta (mV)"), ("cell", "zeta_mv", " Cell  Zeta (mV)")],
    "Cell /  Cell ": [("cell", "diameter_um", " Cell diameter  (μm)"), ("cell", "density", "细胞密度 (kg/m³)"),
        ("cell", "concentration_per_ml", " Concentration  (cells/mL)"), ("cell", "count", "轨迹采样量 ( Continuous source )")],
    "Numerics /  Value ": [("spray", "duration_ms", "喷雾计算时长 (ms)"),
        ("numerics", "radial_cells", "内部径向网格数"), ("numerics", "mixing_cells", "混合段轴向网格数"),
        ("numerics", "workers", "并行数量 (0=自动)"), ("numerics", "particle_backend", "粒子计算后端")]
}


class Viewer(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.label = ttk.Label(self, text="运行后显示计算结果", anchor="center")
        self.label.pack(expand=True, fill="both")
        self.source = None; self.photo = None; self.animation = []; self.frame = 0; self.job = None
        self.path = None
        self.closed = False
        self.bind("<Configure>", lambda e: self.render())
        self.bind("<Destroy>", lambda e: self.close() if e.widget is self else None)

    def close(self):
        self.closed = True
        if self.job:
            try:self.after_cancel(self.job)
            except tk.TclError:pass
            self.job = None
        self.photo = None; self.animation = []; self.source = None; self.path = None

    def clear(self,text='运行后显示计算结果'):
        if self.closed:return
        if self.job is not None:self.after_cancel(self.job);self.job=None
        self.source=None;self.photo=None;self.animation=[];self.frame=0;self.path=None
        self.label.configure(image='',text=text)

    def load(self, path):
        if self.job: self.after_cancel(self.job); self.job = None
        with Image.open(path) as picture: self.source = picture.convert("RGB").copy()
        self.path=Path(path)
        self.animation = []; self.render()

    def play(self, path):
        if self.job: self.after_cancel(self.job); self.job = None
        with Image.open(path) as picture:
            self.animation = [x.convert("RGB").copy() for x in ImageSequence.Iterator(picture)]
        self.path=Path(path)
        self.frame = 0; self._next()

    def _next(self):
        if self.closed or not self.animation: return
        self.source = self.animation[self.frame % len(self.animation)]
        self.frame += 1; self.render()
        self.job = self.after(80, self._next)

    def render(self):
        if self.closed or self.source is None: return
        picture = self.source.copy()
        picture.thumbnail((max(300, self.winfo_width()-8), max(200, self.winfo_height()-8)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(picture)
        self.label.configure(image=self.photo, text="")


class Application:
    def __init__(self, root, config, on_result=None, on_data=None, on_pending=None):
        self.root = root; self.config = config; self.messages = queue.Queue(); self.busy = False; self.output = None
        self.latest_result = None; self.result_data = None; self.running_output = None
        self.explorer_windows = []; self.state_windows = []; self.on_result = on_result; self.on_data = on_data
        self.on_pending=on_pending
        self.trajectory_windows=[]; self.study_windows=[];self.process_windows=[];self.size_windows=[];self.interaction_apps=[]
        self.closed = False; self.poll_job = None; self.rate_job = None
        self.worker_thread=None;self.started_at=None;self.current_stage='';self.busy_caption=None
        root.title("CyESI Multiphysics Simulator")
        root.geometry("1500x960"); root.minsize(1080, 700)
        style = ttk.Style(); style.theme_use("clam")
        style.configure("TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", padding=7)
        outer = ttk.Panedwindow(root, orient="horizontal"); outer.pack(fill="both", expand=True, padx=10, pady=8)
        left = ttk.Frame(outer, width=335); right = ttk.Frame(outer)
        outer.add(left, weight=0); outer.add(right, weight=1)
        ttk.Label(left, text="CyESI  Parameters ", font=("Microsoft YaHei UI", 16, "bold")).pack(anchor="w", padx=8, pady=8)
        canvas = tk.Canvas(left, width=335, highlightthickness=0)
        scroll = ttk.Scrollbar(left, command=canvas.yview); canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y"); canvas.pack(fill="both", expand=True)
        form = ttk.Frame(canvas); canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.entries = {}
        self.entry_widgets = {}
        for group, inputs in INPUTS.items():
            box = ttk.LabelFrame(form, text=group, padding=7); box.pack(fill="x", padx=6, pady=5)
            for row, (section, key, label) in enumerate(inputs):
                ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
                value=getattr(getattr(config, section),key)
                var = tk.StringVar(value="" if value is None else str(value))
                self.entries[(section, key)] = var
                choices = {("supply", "sample_control"): ("flow", "pressure"),
                    ("numerics", "particle_backend"): ("auto", "numba", "numpy"),
                    ("supply", "orientation"): ("horizontal", "down", "up", "horizontal_down", "down_horizontal", "custom", "legacy")}
                widget = (ttk.Combobox(box, textvariable=var, values=choices[(section,key)], state="readonly", width=16)
                    if (section,key) in choices else ttk.Entry(box, textvariable=var, width=12))
                widget.grid(row=row, column=1, sticky="e"); self.entry_widgets[(section,key)] = widget
            if group.startswith('Supply'):
                converter=ttk.Frame(box);converter.grid(row=len(inputs),column=0,columnspan=2,sticky='ew',pady=5)
                ttk.Label(converter,text='pressure=恒压; flow=恒流\n方向: horizontal  Horizontal  / down 向下 / up  Vertical rise \n组合按入口→出口排列; custom 在高级 JSON 编辑.',wraplength=310).pack(anchor='w')
                line=ttk.Frame(converter);line.pack(fill='x',pady=3)
                self.pressure_value=tk.StringVar(value='40');self.pressure_unit=tk.StringVar(value='kPa')
                ttk.Entry(line,textvariable=self.pressure_value,width=7).pack(side='left')
                ttk.Combobox(line,textvariable=self.pressure_unit,values=('mbar','kPa','bar','psi','Pa'),state='readonly',width=5).pack(side='left')
                ttk.Button(line,text='换算并写入',command=self.apply_pressure_conversion).pack(side='left')
                self.pressure_equivalence=tk.StringVar();ttk.Label(converter,textvariable=self.pressure_equivalence,wraplength=305).pack(anchor='w')
                target=ttk.Frame(converter);target.pack(fill='x',pady=3)
                self.target_flow=tk.StringVar(value=str(config.fluid.sample_flow_ul_min))
                ttk.Entry(target,textvariable=self.target_flow,width=7).pack(side='left')
                ttk.Label(target,text='μL/min').pack(side='left')
                ttk.Button(target,text='反推压力',command=self.infer_sample_pressure).pack(side='left')
                self.supply_note=tk.StringVar();ttk.Label(converter,textvariable=self.supply_note,wraplength=305).pack(anchor='w')
        self.spray_enabled = tk.BooleanVar(value=config.spray.enabled)
        ttk.Checkbutton(form, text="计算喷口两相流 (实验性)", variable=self.spray_enabled).pack(anchor="w", padx=12, pady=4)
        self.free_charge_enabled=tk.BooleanVar(value=config.spray.free_charge_enabled)
        ttk.Checkbutton(form,text='两相自由电荷 / RC供电 / Maxwell Feedback ',variable=self.free_charge_enabled).pack(anchor='w',padx=12,pady=3)
        ttk.Label(form, text=" Default : 140 mM  Ammonium formate  / 纯甲醇\n所有材料与数值参数可通过高级 JSON 修改.", wraplength=300).pack(padx=12, pady=5)
        buttons = ttk.Frame(left); buttons.pack(fill="x", padx=8, pady=8)
        visual = ttk.Frame(buttons); visual.pack(fill="x", pady=(0,5))
        ttk.Button(visual, text="全装置可视化", command=lambda:self.open_explorer("device")).pack(side="left", expand=True, fill="x")
        ttk.Button(visual, text=" Emitter  / 液滴可视化", command=lambda:self.open_explorer("nozzle")).pack(side="left", expand=True, fill="x")
        ttk.Button(buttons,text="脂质膜细胞状态 /  Single-source rate ",command=self.open_cell_state).pack(fill="x",pady=3)
        ttk.Button(buttons,text=" MS acquisition  / 理论单细胞率",command=lambda:self.open_cell_state(True)).pack(fill="x",pady=3)
        ttk.Button(buttons,text="连续细胞:  Long capillary  →  Focusing  → 采集回收",command=self.open_process).pack(fill='x',pady=3)
        ttk.Button(buttons,text='二维细胞扰动 /  Binding  /  Charging ',command=self.open_interactions).pack(fill='x',pady=3)
        ttk.Button(buttons,text="有限球轨迹原图 /  Animation ",command=self.open_trajectories).pack(fill='x',pady=3)
        ttk.Button(buttons,text=' Sample capillary length  /  Orientation  /  Shear  /  Single-source rate ',command=self.open_length_study).pack(fill='x',pady=3)
        ttk.Button(buttons,text=' Cell size  0.5 / 2 / 8 / 20 μm  Comparison ',command=self.open_size_study).pack(fill='x',pady=3)
        self.run_button = ttk.Button(buttons, text="Run simulation", command=self.start); self.run_button.pack(fill="x")
        self.scan_button = ttk.Button(buttons, text=" Run  + 19 点参数扫描", command=lambda: self.start(True)); self.scan_button.pack(fill="x", pady=4)
        self.run_progress=ttk.Progressbar(buttons,mode='indeterminate');self.run_progress.pack(fill='x',pady=3)
        self.run_note=tk.StringVar(value='并行 0=按 CPU / 可用内存自动分配, 1=串行; auto 自动使用编译粒子计算.首次编译会稍慢.')
        ttk.Label(buttons,textvariable=self.run_note,wraplength=315).pack(fill='x',pady=(0,5))
        row = ttk.Frame(buttons); row.pack(fill="x")
        ttk.Button(row, text="加载", command=self.load).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="保存", command=self.save).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="高级 JSON", command=self.advanced).pack(side="left", expand=True, fill="x")
        ttk.Button(buttons,text='加载已有结果 (含原图 /  Animation )',command=self.load_results).pack(fill='x',pady=3)
        ttk.Button(buttons, text="打开结果报告", command=self.open_report).pack(fill="x", pady=4)
        self.rate_text = tk.StringVar()
        ttk.Label(right,textvariable=self.rate_text,wraplength=1000,justify="left").pack(fill="x",pady=(4,8))
        self.tabs = ttk.Notebook(right); self.tabs.pack(fill="both", expand=True)
        self.viewers = []
        for name in PANELS:
            if name=='细胞轨迹':
                self.trajectory_tabs=ttk.Notebook(self.tabs);self.tabs.add(self.trajectory_tabs,text=name)
                from .trajectory_stream import TrajectoryStreamPanel
                self.trajectory_stream=TrajectoryStreamPanel(self.trajectory_tabs)
                self.trajectory_tabs.add(self.trajectory_stream,text=' Continuous feed  / 全程物理时钟')
                self.trajectory_animation=Viewer(self.trajectory_tabs)
                self.trajectory_tabs.add(self.trajectory_animation,text='有限球轨迹动画 (原图)')
                ttk.Label(self.trajectory_animation,text='原图按进入末端后的同龄响应对齐, 重合图元可代表多条轨迹; 真实到达时差见“ Continuous feed ”.',wraplength=1000).pack(side='bottom',fill='x',padx=6,pady=4)
                v=Viewer(self.trajectory_tabs);self.trajectory_tabs.add(v,text='轨迹 / 出口分布')
                self.viewers.append(v)
            else:
                v = Viewer(self.tabs); self.viewers.append(v); self.tabs.add(v, text=name)
        self.status = tk.StringVar(value=" Ready .坐标原点位于中管出口.")
        ttk.Label(right, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=6)
        self.log = tk.Text(right, height=6, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="x")
        self.poll_job = self.root.after(100, self.poll)
        for var in self.entries.values():var.trace_add("write",self.inputs_changed)
        self.spray_enabled.trace_add("write",self.inputs_changed)
        self.free_charge_enabled.trace_add('write',self.inputs_changed)
        self.update_rates()
        self.root.bind("<Destroy>", lambda e:self.close() if e.widget is self.root else None, add="+")

    def close(self):
        if self.closed:return
        self.closed = True
        try:self.run_progress.stop()
        except tk.TclError:pass
        if self.poll_job is not None:
            try:self.root.after_cancel(self.poll_job)
            except tk.TclError:pass
            self.poll_job = None
        if self.rate_job is not None:
            try:self.root.after_cancel(self.rate_job)
            except tk.TclError:pass
            self.rate_job = None
        for viewer in self.viewers:viewer.close()
        self.trajectory_animation.close()
        self.trajectory_stream.close()
        for viewer in self.explorer_windows:
            if not viewer.closed:viewer.close()
        self.explorer_windows.clear()
        for window in self.state_windows:
            if not window.closed:window.close()
        self.state_windows.clear()
        for window in self.trajectory_windows:
            if not window.closed:window.close()
        self.trajectory_windows.clear()
        for window in self.study_windows:
            if not window.closed:window.close()
        self.study_windows.clear()
        for window in self.size_windows:
            if not window.closed:window.close()
        self.size_windows.clear()
        for window in self.process_windows:
            if not window.closed:window.close()
        self.process_windows.clear()
        for app in self.interaction_apps:
            if not app.closed:app.close()
        self.interaction_apps.clear()
        self.on_result = None;self.on_data = None;self.on_pending=None

    def set_busy(self,busy):
        self.busy=busy
        if busy:
            self.started_at=time.perf_counter();self.current_stage='开始求解';self.busy_caption=None
            self.run_button.configure(state='disabled',text=' Computing … 00:00')
            self.scan_button.configure(state='disabled');self.run_progress.start(50)
            self.run_note.set(' Running .按钮暂时禁用以避免重复启动; 进度见右侧日志.')
        else:
            elapsed=0 if self.started_at is None else int(time.perf_counter()-self.started_at)
            self.run_button.configure(state='normal',text='Run simulation');self.scan_button.configure(state='normal')
            self.run_progress.stop();self.run_note.set(f'本次任务已结束, 用时 {elapsed//60:02d}:{elapsed%60:02d}, 可以再次运行.')
            self.busy_caption=None

    def update_busy_caption(self):
        if not self.busy:return
        elapsed=0 if self.started_at is None else int(time.perf_counter()-self.started_at)
        stage=self.current_stage
        if 'Encoding' in stage:label='正在生成动画'
        elif 'Rendering' in stage:label='正在导出图像'
        elif 'Estimating Q Exactive' in stage:label='正在计算质谱采集率'
        elif 'two-phase' in stage or ' Emitter ' in stage:label='正在求解喷口两相流'
        elif 'Tracking' in stage or 'transport responses' in stage or 'cells:' in stage:label='正在计算细胞运输'
        elif 'scan' in stage.lower():label='正在进行参数扫描'
        else:label=' Computing '
        caption=f'{label}… {elapsed//60:02d}:{elapsed%60:02d}'
        if caption!=self.busy_caption:
            self.run_button.configure(text=caption);self.busy_caption=caption

    def record_ui_error(self,context,detail):
        """A failed preview/view must not stop queue polling or lose completion."""
        message=f'{context}: {detail}'
        self.log.configure(state='normal');self.log.insert('end',message+'\n');self.log.see('end');self.log.configure(state='disabled')
        directory=self.running_output or self.output
        if directory is not None:
            try:
                with (Path(directory)/'gui_errors.log').open('a',encoding='utf-8') as handle:handle.write(message+'\n')
            except OSError:pass
        self.status.set('计算/导出继续; 部分画面未能刷新, 详情见日志.' if self.busy else '结果已保留; 部分画面未能刷新, 详情见日志.')

    def inputs_changed(self,*args):
        if self.closed:return
        if self.rate_job is not None:self.root.after_cancel(self.rate_job)
        self.rate_job=self.root.after(250,self.update_rates)

    def publish_data(self,data):
        self.result_data=data
        self.trajectory_stream.set_data(data)
        for windows in (self.explorer_windows,self.state_windows,self.trajectory_windows,self.process_windows):
            for window in windows[:]:
                if window.closed:windows.remove(window)
                else:window.set_data(data)
        for app in self.interaction_apps:
            if not app.closed:app.session.adopt(data,reset_draft=True);app.refresh()
        if self.on_data is not None:self.on_data(data)

    def publish_pending(self,config):
        self.trajectory_stream.set_pending(config)
        for windows in (self.explorer_windows,self.state_windows,self.trajectory_windows,self.process_windows):
            for window in windows:
                if not window.closed:window.set_pending(config)
        for app in self.interaction_apps:
            if not app.closed:app.session.edit(config);app.refresh()
        if self.on_pending is not None:self.on_pending(config)

    def update_rates(self):
        if self.closed:return
        if self.rate_job is not None:
            self.root.after_cancel(self.rate_job);self.rate_job=None
        try:config=self.collect()
        except ValueError:
            self.rate_text.set("输入尚未完整或无效; 结果仍对应上次有效参数.")
            return
        from ..fluid.supply import convert_pressure, estimate_flow
        pressure=config.supply.sample_pressure_mbar
        self.pressure_equivalence.set(f'{pressure:g} mbar = {convert_pressure(pressure,"mbar","kPa"):g} kPa = {pressure/1000:g} bar = {convert_pressure(pressure,"mbar","psi"):.4g} psi')
        if config.supply.sample_control=='pressure':
            estimate=estimate_flow(config)
            self.supply_note.set(f'恒压估算 Q ≈ {estimate:.4g} μL/min; Run 后用出口 FEM 背压修正.流量输入在恒压模式不作为边界.')
            self.entry_widgets[('fluid','sample_flow_ul_min')].configure(state='readonly')
            solved=self.result_data.config if self.result_data is not None else None
            same=solved is not None and solved.supply==config.supply and solved.geometry==config.geometry and solved.fluid==config.fluid and solved.electrical==config.electrical and solved.chemistry==config.chemistry
            if not same:
                config.fluid.sample_flow_ul_min=max(estimate,1e-9)
                value=str(config.fluid.sample_flow_ul_min)
                if self.entries[('fluid','sample_flow_ul_min')].get()!=value:
                    self.entries[('fluid','sample_flow_ul_min')].set(value)
        else:
            self.entry_widgets[('fluid','sample_flow_ul_min')].configure(state='normal')
            self.supply_note.set('样品恒流; 鞘液始终恒流.压力反推为充分发展管流估算.')
        self.publish_pending(config)
        incoming=config.cell.concentration_per_ml*config.fluid.sample_flow_ul_min/60000
        text=f"当前输入:  Concentration  {config.cell.concentration_per_ml:g} cells/mL ×  Sample flow  {config.fluid.sample_flow_ul_min:g} μL/min →  Injection  {incoming:.2f} cells/s"
        if self.result_data is not None:
            from ..cell_state.analysis import outlet_rates
            from ..particle.events import concentration_only_change
            try:
                data=self.result_data.with_rate_inputs(config)
                if data is not self.result_data:self.publish_data(data)
            except ValueError as error:
                self.rate_text.set(text+f"\n事件统计未更新: {error}");return
            self.publish_pending(config)
            rates=outlet_rates(data.config,data.summary,data.rate_config)
            rate=rates['nozzle_cells_s']
            label='新流量预览 (暂用上次出管比例)' if data.rate_preview else '当前结果:  Steady model outlet rate '
            text+=f"\n{label} {rate:.2f} cells/s" if rate is not None else "\n当前结果: 出管率未知"
            for row in rates['observation_windows']:
                text+=f"; {row['duration_s']:g} s 窗口平均 {row['observed_nozzle_cells_s']:.2f} cells/s ({row['nozzle_exits']}  Count )"
            if not concentration_only_change(data.config,config):
                from .source_state import pending_message
                text+='\n'+pending_message(data.config,config)
            elif self.latest_result is None or self.latest_result.config.cell.concentration_per_ml!=config.cell.concentration_per_ml:
                text+="\n浓度已同步至连续源和打开的窗口;  Flow field /轨迹复用.已保存的主报告与图像保留原计算参数."
        self.rate_text.set(text)

    def get_data(self):
        from ..visualization.explorer_data import ExplorerData,find_results
        if self.result_data is None:
            if self.latest_result is not None:self.result_data=ExplorerData.from_result(self.latest_result,source=self.running_output or self.output or ' Current results ')
            else:
                source=self.output or find_results()
                if source is None:raise ValueError("请先运行一次流场与细胞运输模拟")
                self.load_results(source,update_parameters=False)
        # Read the form now as well as through the debounce, so clicking a view
        # immediately after typing cannot reopen a stale concentration snapshot.
        self.collect()
        self.update_rates()
        return self.result_data

    def load_figures(self,directory):
        from ..visualization.figures import FILENAMES
        path=Path(directory)
        for i,viewer in enumerate(self.viewers):
            file=path/f'{FILENAMES[FILE_MAP[i]]}.png'
            if file.exists():viewer.load(file)
            else:viewer.clear('本结果未导出此图.')
        gif=path/'cell_trajectories.gif'
        if gif.exists():self.trajectory_animation.play(gif)
        else:self.trajectory_animation.clear('本结果未导出 GIF. Select “有限球轨迹原图 /  Animation ”可从已有轨迹恢复, 无需重跑 CFD.')
        if (path/'two_phase.gif').exists():self.viewers[5].play(path/'two_phase.gif')

    def load_results(self,directory=None,update_parameters=True,data=None):
        from ..visualization.explorer_data import ExplorerData
        if self.busy and update_parameters:
            self.status.set(' Computing ; 完成后可加载其他已有结果.');return
        interactive=directory is None
        if directory is None:
            directory=filedialog.askdirectory(parent=self.root,title='加载已计算结果 (config.json / fields.npz)')
            if not directory:return
        try:data=data or ExplorerData.load(directory)
        except (ValueError,OSError,KeyError) as error:
            if interactive:messagebox.showerror('已有结果加载失败',str(error),parent=self.root);return
            raise
        self.output=Path(directory).resolve();self.latest_result=None;self.running_output=None
        if update_parameters:self.config=deepcopy(data.config);self.refresh()
        self.publish_data(data);self.load_figures(self.output);self.update_rates()
        self.status.set(f'已加载计算结果和原有图像/ Animation : {self.output}')

    def open_trajectories(self):
        from .trajectories import TrajectoryWindow
        try:
            data=self.get_data()
            for window in self.trajectory_windows:
                if not window.closed:window.window.lift();return window
            window=TrajectoryWindow(self.root,data);self.trajectory_windows.append(window)
            window.set_pending(self.collect())
            return window
        except Exception as error:messagebox.showerror('轨迹原图打开失败',str(error),parent=self.root)

    def collect(self):
        config = deepcopy(self.config)
        for (section, key), var in self.entries.items():
            target = getattr(config, section)
            if key=="spray_current_na" and var.get().strip().lower() in ("","none","null"):
                target.spray_current_na=None;continue
            # JSON may encode a physical float (including concentration) as an
            # integer. Parse by the declared field type, not its loaded value.
            declared=next(field.type for field in fields(target) if field.name==key)
            dtype = str if declared in (str,"str") else int if declared in (int,"int") else float
            try: setattr(target, key, dtype(var.get()))
            except ValueError: raise ValueError(f"{section}.{key}: 无效数值")
        config.spray.enabled = self.spray_enabled.get()
        config.spray.free_charge_enabled=self.free_charge_enabled.get()
        return config.validate()

    def apply_pressure_conversion(self):
        from ..fluid.supply import convert_pressure
        try:
            value=convert_pressure(float(self.pressure_value.get()),self.pressure_unit.get(),'mbar')
            if value<0:raise ValueError('供液表压不能为负')
            self.entries[('supply','sample_pressure_mbar')].set(f'{value:.10g}')
            self.entries[('supply','sample_control')].set('pressure');self.update_rates()
        except (ValueError,KeyError) as error:messagebox.showerror('压力换算',str(error),parent=self.root)

    def infer_sample_pressure(self):
        from ..fluid.supply import infer_pressure
        try:
            config=self.collect();target=float(self.target_flow.get())
            if target<=0:raise ValueError('目标流量必须为正')
            pressure=infer_pressure(config,target)
            if pressure<0:raise ValueError('该姿态的重力压头已超过目标阻力, 需要负表压; 请检查供液条件.')
            self.entries[('supply','sample_pressure_mbar')].set(f'{pressure:.10g}')
            self.entries[('supply','sample_control')].set('pressure');self.update_rates()
            self.supply_note.set(f'估算压力 {pressure:.4g} mbar; 已包含静压, 尚未包含 FEM 出口附加背压.')
        except ValueError as error:messagebox.showerror('反推压力',str(error),parent=self.root)

    def open_length_study(self):
        from .length_study import LengthStudyWindow
        try:
            window=LengthStudyWindow(self.root,self.collect());self.study_windows.append(window)
            return window
        except Exception as error:messagebox.showerror(' Length  / 剪切研究',str(error),parent=self.root)

    def open_size_study(self):
        from .size_study import SizeStudyWindow
        try:
            window=SizeStudyWindow(self.root,self.collect());self.size_windows.append(window);return window
        except Exception as error:messagebox.showerror('细胞尺寸对比',str(error),parent=self.root)

    def refresh(self):
        for (section, key), var in self.entries.items():
            value=getattr(getattr(self.config,section),key);var.set("" if value is None else str(value))
        self.spray_enabled.set(self.config.spray.enabled)
        self.free_charge_enabled.set(self.config.spray.free_charge_enabled)

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path: return
        try: self.config = Config.load(path); self.refresh()
        except Exception as exc: messagebox.showerror("参数错误", str(exc))

    def save(self):
        try:
            config = self.collect()
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path: config.save(path)
        except Exception as exc: messagebox.showerror("参数错误", str(exc))

    def advanced(self):
        try: config = self.collect()
        except Exception as exc: messagebox.showerror("参数错误", str(exc)); return
        win = tk.Toplevel(self.root); win.title("全部参数 / JSON (字段名包含单位)"); win.geometry("760x750")
        editor = tk.Text(win, wrap="none", font=("Consolas", 11)); editor.pack(fill="both", expand=True)
        editor.insert("1.0", json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
        def apply():
            try:
                self.config = Config.from_dict(json.loads(editor.get("1.0", "end")))
                self.refresh(); win.destroy()
            except Exception as exc: messagebox.showerror("参数错误", str(exc), parent=win)
        ttk.Button(win, text="验证并应用", command=apply).pack(pady=5)

    def start(self, scanning=False, output_directory=None):
        if self.busy: return
        try:
            config = self.collect()
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
            output = Path(output_directory) if output_directory is not None else Path("outputs")/f"gui_{stamp}"
            output.mkdir(parents=True, exist_ok=True)
            from ..visualization.figures import serialized_plotting
        except Exception as exc:messagebox.showerror('无法启动模拟',str(exc),parent=self.root);return
        self.config = config;self.running_output=output.resolve()
        # The worker owns a queue and plain data only, not Tk widgets/variables.
        # Keeping `self` in a worker closure can finalize Tk objects off-thread
        # after a parameter window is closed.
        messages = self.messages
        def progress(text): messages.put(("log", text))
        @serialized_plotting
        def stage(name, data):
            # Matplotlib is used only by this worker; Tk displays exported pixels.
            from ..visualization.figures import _new, _map, plt
            import numpy as np
            if name == "solvent":
                fig, ax = _new("Solved velocity / pressure", cols=2)
                _map(ax[0], data.radial_edges, data.axial_edges, np.hypot(data.velocity_r, data.velocity_z)*1000, "mm/s", active=data.active)
                _map(ax[1], data.radial_edges, data.axial_edges, data.pressure, "Pa", active=data.active)
                index = 0
                concentration, ca = _new("Computed methanol concentration")
                _map(ca[0], data.radial_edges, data.axial_edges, data.concentration, "Methanol fraction", active=data.active, zlim=(-1.3,0))
                cp=output/"preview_methanol.png";concentration.savefig(cp);plt.close(concentration)
                messages.put(("image", (3, cp)))
            elif name == "electric":
                fig, ax = _new("Solved electric field")
                _map(ax[0], data.radial_edges, data.axial_edges, np.hypot(data.field_r, data.field_z)/1e6, "kV/mm")
                index = 4
            elif name == "particles":
                from types import SimpleNamespace
                from ..visualization.figures import trajectories, exposure
                preview=SimpleNamespace(config=config,particles=data)
                for builder,index in [(trajectories,2),(exposure,3)]:
                    fig=builder(preview);path=output/f"preview_cells_{index}.png"
                    fig.savefig(path);plt.close(fig);messages.put(("image",(index,path)))
                return
            elif name == "spray":
                from types import SimpleNamespace
                from ..visualization.figures import spray
                fig=spray(SimpleNamespace(config=config,spray=data));index=5
            else: return
            path = output/f"preview_{name}.png"; fig.savefig(path); plt.close(fig)
            messages.put(("image", (index, path)))
        def worker():
            try:
                from ..engine import run
                from ..report.export import export
                result = run(config, progress, stage)
                messages.put(("result", result))
                if scanning:
                    from ..optimization.parameter_scan import scan
                    result.scan = scan(config, cells=512, progress=progress)
                path = export(result, output, animations=True, progress=progress)
                messages.put(("done", (path, result.particles.diagnostics)))
            except Exception:
                messages.put(("error", traceback.format_exc()))
        self.worker_thread=threading.Thread(target=worker,daemon=True)
        self.set_busy(True)
        try:self.worker_thread.start()
        except Exception as exc:
            self.worker_thread=None;self.set_busy(False)
            messagebox.showerror('无法启动模拟线程',str(exc),parent=self.root)

    def poll(self):
        if self.closed:return
        self.poll_job=None
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                try:
                    if kind == "log":
                        self.current_stage=payload;self.status.set(payload); self.log.configure(state="normal")
                        self.log.insert("end", payload+"\n"); self.log.see("end"); self.log.configure(state="disabled")
                    elif kind == "image": self.viewers[payload[0]].load(payload[1])
                    elif kind == "result":
                        self.latest_result = payload
                        if payload.config.supply.sample_control=='pressure':
                            # Report the actual solved Q while keeping the pressure
                            # as the boundary condition for subsequent runs.
                            self.entries[('fluid','sample_flow_ul_min')].set(str(payload.config.fluid.sample_flow_ul_min))
                        from ..visualization.explorer_data import ExplorerData
                        self.publish_data(ExplorerData.from_result(payload,source=self.running_output or ' Current results '))
                        if self.on_result is not None:self.on_result(payload)
                        self.update_rates()
                    elif kind == "done":
                        try:
                            self.output, d = payload
                            self.load_figures(self.output)
                            expected_rate=self.latest_result.events['steady_nozzle_cells_s']
                            self.status.set(f"完成: 轨迹样本 {d['exited']}/{d['count']}  Exited ;  Steady model outlet rate  {expected_rate:.2f} cells/s; MS 入孔/状态见脂质膜界面. Report : {self.output/'report.html'}")
                            self.update_rates()
                        finally:self.set_busy(False)
                    elif kind == "error":
                        self.set_busy(False)
                        self.record_ui_error('求解错误',payload)
                        self.status.set(" Computation failed , 详细诊断见弹窗.")
                        messagebox.showerror("求解错误", payload,parent=self.root)
                except Exception:
                    self.record_ui_error(f'界面消息 {kind} 处理失败',traceback.format_exc())
        except queue.Empty: pass
        finally:
            if not self.closed:
                # Even if a preview or completion callback failed, keep polling.
                # A finished worker must not leave the controls disabled.
                if self.busy and self.worker_thread is not None and not self.worker_thread.is_alive() and self.messages.empty():
                    self.set_busy(False)
                self.update_busy_caption()
                self.poll_job = self.root.after(100, self.poll)

    def open_report(self):
        if self.output is not None:
            import webbrowser
            webbrowser.open((self.output/"report.html").as_uri())

    def open_interactions(self):
        from ..workbench.app import WorkbenchApp
        from ..workbench.session import WorkbenchSession
        try:
            data=self.get_data()
            for app in self.interaction_apps:
                if not app.closed:app.root.lift();app.navigate('interactions');return app
            session=WorkbenchSession(data.config);session.adopt(data,reset_draft=True)
            window=tk.Toplevel(self.root);app=WorkbenchApp(window,session=session)
            self.interaction_apps.append(app)
            app.navigate('interactions');return app
        except Exception as error:messagebox.showerror('二维相互作用打开失败',str(error),parent=self.root)

    def open_process(self):
        from .process import ProcessWindow
        try:
            data=self.get_data()
            for window in self.process_windows:
                if not window.closed:window.window.lift();return window
            window=ProcessWindow(self.root,data,on_run=self.start);self.process_windows.append(window)
            window.set_pending(self.collect());return window
        except Exception as error:messagebox.showerror('连续细胞过程打开失败',str(error),parent=self.root)

    def open_cell_state(self,acquisition=False):
        from .cell_state import CellStateWindow
        try:
            data=self.get_data()
            window=CellStateWindow(self.root,data,on_run=self.start);self.state_windows.append(window)
            window.set_pending(self.collect())
            if acquisition:window.notebook.select(window.acquisition)
            return window
        except Exception as error:messagebox.showerror("细胞状态界面打开失败",str(error),parent=self.root)

    def open_explorer(self, page="device"):
        from .explorer import ExplorerWindow
        try:
            data=self.get_data()
            for viewer in self.explorer_windows:
                if not viewer.closed:
                    viewer.set_page(page);viewer.window.lift();return
            viewer = ExplorerWindow(self.root,data,page=page,on_parameters=self.root.lift,on_run=self.start)
            self.explorer_windows.append(viewer)
            viewer.set_pending(self.collect())
        except Exception as error:messagebox.showerror("可视化打开失败",str(error))


def launch(config_path=None,results=None,workers=None,particle_backend=None):
    from ..visualization.explorer_data import ExplorerData,find_results
    source=results or (find_results() if config_path is None else None)
    data=ExplorerData.load(source) if source is not None else None
    config = Config.load(config_path) if config_path else data.config if data is not None else Config()
    root = tk.Tk();app=Application(root,config)
    if data is not None:app.load_results(source,update_parameters=config_path is None,data=data)
    if workers is not None:app.entries[('numerics','workers')].set(str(workers))
    if particle_backend is not None:app.entries[('numerics','particle_backend')].set(particle_backend)
    if config_path is None and results is None:
        # Adopt the confirmed experimental boundary for new work; a loaded old
        # result keeps its own source inputs and is visibly marked pending.
        raw={} if source is None else json.loads((Path(source)/'config.json').read_text(encoding='utf-8-sig'))
        if 'supply' not in raw:
            app.entries[('supply','sample_control')].set('pressure')
            app.entries[('supply','sample_pressure_mbar')].set('400')
            app.entries[('supply','orientation')].set('horizontal')
            app.update_rates()
            app.status.set('新计算采用已确认的样品恒压 400 mbar,  Constant sheath flow ; 旧图对应原参数, 待 Run 更新.水平为单次计算情景, 多种摆放见长度 / 剪切研究.')
    root.mainloop()
