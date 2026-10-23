"""Tube-length and shear workspace with background flow calculations."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from .interface import Viewer


class LengthStudyWindow:
    def __init__(self, parent, config):
        self.config=deepcopy(config);self.closed=False;self.busy=False;self.job=None;self.output=None
        self.messages=queue.Queue();self.worker_thread=None
        self.window=tk.Toplevel(parent);self.window.title(' Sample capillary length  /  Orientation  /  Shear  /  Single-source rate ');self.window.geometry('1380x900')
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        controls=ttk.LabelFrame(self.window,text='从主界面复制当前参数; 研究结果另存, 不覆盖原 CFD',padding=10);controls.pack(fill='x',padx=10,pady=8)
        self.values={}
        items=[('lengths',' Capillary length  (cm, 逗号分隔)','5,10,20,40,70,100'),('pressure','样品表压 (mbar)',str(config.supply.sample_pressure_mbar)),
               ('cells','每情景轨迹样本','128'),('capture',' Capture rate  (1/s,  User-defined )',str(config.supply.wall_capture_rate_s)),
               ('gap','壁面接触间隙 (μm,  Specified )',str(config.supply.wall_capture_gap_um)),
               ('hazards','损伤系数情景 ( User-defined )','0,0.001,0.01'),('actual_it',' Actual  IT  Model settings  (ms)','5,10,30'),
               ('width',' Single-source packet width  (ms,  Specified )','0'),
               ('concentration',' Concentration  (cells/mL)',str(config.cell.concentration_per_ml)),
               ('flow','恒流模式样品 Q (μL/min)',str(config.fluid.sample_flow_ul_min)),
               ('delivery',' Exited → Packet delivery fraction  ( Specified )','1'),('overhead','控制附加时间 (ms,  Specified )','20'),
               ('workers','并行数量 (0=自动)',str(config.numerics.workers))]
        for i,(key,label,value) in enumerate(items):
            col=(i%4)*2;row=i//4
            ttk.Label(controls,text=label).grid(row=row,column=col,sticky='w',padx=4,pady=3)
            var=tk.StringVar(value=value);self.values[key]=var
            ttk.Entry(controls,textvariable=var,width=18).grid(row=row,column=col+1,padx=4)
        self.control=tk.StringVar(value=config.supply.sample_control)
        ttk.Label(controls,text='样品供液模式').grid(row=4,column=0,sticky='w')
        ttk.Combobox(controls,textvariable=self.control,values=('pressure','flow'),state='readonly',width=16).grid(row=4,column=1)
        self.ablations=tk.BooleanVar(value=True)
        ttk.Checkbutton(controls,text='70 cm 力学消融:  Gravity  /  Lift  / 润滑',variable=self.ablations).grid(row=4,column=2,columnspan=4,sticky='w')
        routebox=ttk.Frame(self.window);routebox.pack(fill='x',padx=15)
        self.orientations={}
        for key,label in [('horizontal',' Horizontal '),('down',' Vertical down '),('up',' Vertical up '),('horizontal_down',' Horizontal →向下'),('down_horizontal','向下→ Horizontal ')]:
            var=tk.BooleanVar(value=True);self.orientations[key]=var
            ttk.Checkbutton(routebox,text=label,variable=var).pack(side='left',padx=7)
        ttk.Label(self.window,text='每个流量点重新求解末端 CFD; 组合方向各占半程.恒流参考流量继承主界面.机械损伤情景不是实测活率, MS 输出是可分辨机会.',wraplength=1300).pack(anchor='w',padx=15,pady=6)
        action=ttk.Frame(self.window);action.pack(fill='x',padx=10)
        self.run_button=ttk.Button(action,text='计算全部情景',command=self.start);self.run_button.pack(side='left')
        ttk.Button(action,text='打开研究报告',command=self.open_report).pack(side='left',padx=8)
        self.progress=ttk.Progressbar(action,mode='indeterminate');self.progress.pack(side='left',fill='x',expand=True)
        self.status=tk.StringVar(value=' Ready .可先减少长度点或轨迹采样量, 检查设置.')
        ttk.Label(self.window,textvariable=self.status,wraplength=1320).pack(fill='x',padx=15,pady=5)
        tabs=ttk.Notebook(self.window);tabs.pack(fill='both',expand=True,padx=10,pady=6)
        self.viewers=[]
        for name in (' Length  /  Flow rate  /  Shear ',' Exited  / 四档质谱单细胞机会','沉降 /  3D transverse trajectories '):
            viewer=Viewer(tabs);tabs.add(viewer,text=name);self.viewers.append(viewer)
        tablebox=ttk.Frame(tabs);tabs.add(tablebox,text='情景数值 / 消融比较')
        columns=('length_cm','orientation','variant','actual_sample_flow_ul_min','mean_residence_s','wall_stress_pa','mean_shear_dose_pa_s','nozzle_cells_s','near_wall_exposed_fraction','nozzle_rate_delta_from_full_s')
        self.tree=ttk.Treeview(tablebox,columns=columns,show='headings')
        labels=(' Length  cm',' Orientation ','力学情景',' Actual  Q μL/min',' Mean transit time  s',' Wall shear  Pa',' Shear dose  Pa·s',' Exited  cells/s',' Near-wall fraction ','相对完整模型 Δrate')
        for key,label in zip(columns,labels):self.tree.heading(key,text=label);self.tree.column(key,width=130)
        xscroll=ttk.Scrollbar(tablebox,orient='horizontal',command=self.tree.xview);xscroll.pack(side='bottom',fill='x')
        yscroll=ttk.Scrollbar(tablebox,command=self.tree.yview);yscroll.pack(side='right',fill='y')
        self.tree.configure(xscrollcommand=xscroll.set,yscrollcommand=yscroll.set);self.tree.pack(fill='both',expand=True)
        self.columns=columns
        self.job=self.window.after(100,self.poll)

    def start(self, output_directory=None):
        if self.busy:return
        try:
            from ..optimization.length_study import StudyParameters
            from ..acquisition.q_exactive import AcquisitionParameters
            c=deepcopy(self.config);c.supply.sample_control=self.control.get()
            c.numerics.workers=int(self.values['workers'].get())
            c.cell.concentration_per_ml=float(self.values['concentration'].get())
            c.fluid.sample_flow_ul_min=float(self.values['flow'].get())
            c.supply.sample_pressure_mbar=float(self.values['pressure'].get())
            c.supply.wall_capture_rate_s=float(self.values['capture'].get());c.supply.wall_capture_gap_um=float(self.values['gap'].get())
            p=StudyParameters(lengths_cm=[float(v) for v in self.values['lengths'].get().split(',')],cells=int(self.values['cells'].get()),
                orientations=[k for k,v in self.orientations.items() if v.get()],ablations=self.ablations.get(),
                damage_coefficients=[float(v) for v in self.values['hazards'].get().split(',')]).validate()
            ap=AcquisitionParameters(actual_injection_times_ms=[float(v) for v in self.values['actual_it'].get().split(',')],signal_width_ms=float(self.values['width'].get()),
                signal_delivery_fraction=float(self.values['delivery'].get()),overhead_ms=float(self.values['overhead'].get())).validate()
            c.validate()
            output=Path(output_directory) if output_directory else Path('outputs')/'length_studies'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output.mkdir(parents=True,exist_ok=True)
        except Exception as error:messagebox.showerror('研究参数',str(error),parent=self.window);return
        messages=self.messages
        def worker():
            try:
                from ..optimization.length_study import analyze,export
                import json
                from ..report.export import clean,write_csv
                def checkpoint(state):
                    (output/'progress.json').write_text(json.dumps(clean(state),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
                    write_csv(output/'length_study_partial.csv',state['rows'])
                result=analyze(c,p,ap,progress=lambda s:messages.put(('log',s)),on_case=checkpoint)
                path=export(result,output);messages.put(('done',(result,path)))
            except Exception:messages.put(('error',traceback.format_exc()))
        self.worker_thread=threading.Thread(target=worker,daemon=True)
        self.busy=True;self.run_button.configure(state='disabled',text='正在计算情景…');self.progress.start(50)
        try:self.worker_thread.start()
        except Exception as error:self.finish();messagebox.showerror('启动失败',str(error),parent=self.window)

    def finish(self):
        self.busy=False;self.run_button.configure(state='normal',text='计算全部情景');self.progress.stop()

    def poll(self):
        if self.closed:return
        self.job=None
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=='log':self.status.set(payload)
                elif kind=='done':
                    try:
                        result,self.output=payload
                        for viewer,name in zip(self.viewers,('length_physics','length_rates','tube_paths')):viewer.load(self.output/f'{name}.png')
                        self.tree.delete(*self.tree.get_children())
                        for row in result['rows']:
                            self.tree.insert('','end',values=[f'{row[k]:.5g}' if isinstance(row.get(k),float) else row.get(k,'失败' if row['status']=='failed' else ' Unknown ') for k in self.columns])
                        failed=sum(r['status']!='ok' for r in result['rows'])
                        self.status.set(f'完成 {len(result["rows"])} 个情景, 失败 {failed}  Count . Report : {self.output}')
                    finally:self.finish()
                elif kind=='error':self.finish();self.status.set('求解失败, 可修改参数后重试.');messagebox.showerror('研究计算失败',payload,parent=self.window)
        except queue.Empty:pass
        except Exception as error:
            self.finish();self.status.set(f'界面更新失败, 数值结果保留: {error}')
        finally:
            if not self.closed:
                if self.busy and self.worker_thread is not None and not self.worker_thread.is_alive() and self.messages.empty():self.finish()
                self.job=self.window.after(100,self.poll)

    def open_report(self):
        if self.output:
            import webbrowser
            webbrowser.open((self.output/'length_study_report.html').as_uri())

    def close(self):
        if self.closed:return
        self.closed=True
        if self.job is not None:self.window.after_cancel(self.job)
        self.progress.stop()
        for viewer in self.viewers:viewer.close()
        self.window.destroy()
