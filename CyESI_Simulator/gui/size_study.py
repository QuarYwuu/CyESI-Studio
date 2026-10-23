"""Diameter-comparison figures and reproducible recalculation workspace."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .interface import Viewer
from ..visualization.size_comparison import FILENAMES


class SizeStudyWindow:
    def __init__(self,parent,config):
        self.config=deepcopy(config);self.closed=False;self.busy=False;self.job=None;self.output=None
        self.messages=queue.Queue();self.window=tk.Toplevel(parent)
        self.window.title('CyESI · 0.5 / 2 / 8 / 20 μm 细胞尺寸比较');self.window.geometry('1450x950')
        self.window.minsize(1050,720)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.window.bind('<Destroy>',lambda e:self.cleanup() if e.widget is self.window else None,add='+')
        ttk.Label(self.window,text=' Cell size : 运输,  Focusing , 暴露与质谱采集',font=('Microsoft YaHei UI',15,'bold')).pack(anchor='w',padx=14,pady=(10,6))
        self.inputs={};form=ttk.Frame(self.window,padding=10);form.pack(fill='x')
        fields=[('diameters',' Diameter  μm','0.5,2,8,20'),('cells','每组响应数','1024'),('brownian','布朗直径阈值 μm','2'),
            ('timeout','最长末端追踪 s','20'),('width','信号宽 ms ( Specified )','20')]
        for i,(key,label,value) in enumerate(fields):
            frame=ttk.Frame(form);frame.pack(side='left',padx=(0,12));ttk.Label(frame,text=label).pack(anchor='w')
            var=tk.StringVar(value=value);self.inputs[key]=var;ttk.Entry(frame,textvariable=var,width=15).pack()
        self.run_button=ttk.Button(form,text='按复制参数重新计算',command=self.start);self.run_button.pack(side='left',anchor='s',padx=6)
        ttk.Button(form,text='加载已有对比',command=self.choose).pack(side='left',anchor='s',padx=5)
        ttk.Button(form,text='打开 PDF',command=self.open_pdf).pack(side='right',anchor='s')
        self.source=tk.StringVar(value=(f'本次重算继承主界面:  Capillary length  {config.geometry.sample_length_mm:g} mm;  Orientation  {config.supply.orientation};  Fluid supply  {config.supply.sample_control}; '
            f' Pressure  {config.supply.sample_pressure_mbar:g} mbar;  Concentration  {config.cell.concentration_per_ml:g}/mL;  Voltage  {config.electrical.voltage_kv:g} kV.'))
        ttk.Label(self.window,textvariable=self.source,wraplength=1400).pack(fill='x',padx=14,pady=3)
        ttk.Label(self.window,text='每种直径重算有限球响应, 共用同一稀相载液场.超时轨迹先延长追踪, 仍未解析者单列;  Membrane model scenario , 信号峰宽和真实检测效率分别标注.',wraplength=1400).pack(fill='x',padx=14,pady=4)
        line=ttk.Frame(self.window);line.pack(fill='x',padx=14,pady=3)
        self.progress=ttk.Progressbar(line,mode='indeterminate');self.progress.pack(fill='x',expand=True)
        self.status=tk.StringVar(value='可先查看已有对比图; 重新计算将另存结果.')
        ttk.Label(self.window,textvariable=self.status,wraplength=1400).pack(fill='x',padx=14,pady=3)
        tabs=ttk.Notebook(self.window);tabs.pack(fill='both',expand=True,padx=10,pady=5)
        self.viewers=[]
        for title in ['总览 / 八项指标','长管轨迹','出口聚焦','时间 / 暴露分布',' Resolution settings  /  Single-source rate ','条件入口状态','2 μm 布朗对照']:
            viewer=Viewer(tabs);tabs.add(viewer,text=title);self.viewers.append(viewer)
        tableframe=ttk.Frame(tabs);tabs.add(tableframe,text='数值与分母')
        columns=['diameter_um','exited','samples','residence_p50_s','mean_residence_s','outlet_r95_um','outlet_cell_envelope_r95_um','nozzle_cells_s','censored_or_reverse']
        self.columns=columns;self.table=ttk.Treeview(tableframe,columns=columns,show='headings')
        headings=[' Diameter  μm','出管数','响应数',' Median transit  s',' Mean transit time  s',' Sphere centre  r95 μm',' Outer radius  r95+R μm',' Predicted outlet rate  /s',' Censored  /  Reverse flow ']
        for key,title in zip(columns,headings):self.table.heading(key,text=title);self.table.column(key,width=135,anchor='center')
        y=ttk.Scrollbar(tableframe,orient='vertical',command=self.table.yview);y.pack(side='right',fill='y')
        x=ttk.Scrollbar(tableframe,orient='horizontal',command=self.table.xview);x.pack(side='bottom',fill='x')
        self.table.configure(yscrollcommand=y.set,xscrollcommand=x.set);self.table.pack(fill='both',expand=True)
        existing=sorted(Path('outputs/size_studies').glob('*/size_summary.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        if existing:
            try:self.load_result(existing[0].parent)
            except (ValueError,OSError,KeyError):pass
        self.job=self.window.after(100,self.poll)

    def choose(self):
        path=filedialog.askdirectory(parent=self.window,title='选择含 size_summary.json 的尺寸对比目录')
        if path:
            try:self.load_result(path)
            except (ValueError,OSError,KeyError) as error:messagebox.showerror('尺寸对比',str(error),parent=self.window)

    def load_result(self,directory):
        directory=Path(directory).resolve();summary=json.loads((directory/'size_summary.json').read_text(encoding='utf-8'))
        names=FILENAMES+['size_07_brownian_control']
        for viewer,name in zip(self.viewers,names):
            path=directory/(name+'.png')
            if path.exists():viewer.load(path)
            else:viewer.clear('此结果未计算附加对照; 常规四尺寸图请看其他页.')
        self.table.delete(*self.table.get_children())
        for row in summary['rows']:
            fmt=lambda value:' Unknown ' if value is None else f'{value:.5g}' if isinstance(value,float) else str(value)
            self.table.insert('','end',values=[fmt(row.get(k)) for k in self.columns])
        self.output=directory
        c=summary['config']
        self.status.set(f'已显示保存结果: {directory} | 实际计算条件: {c["geometry"]["sample_length_mm"]:g} mm, {c["supply"]["orientation"]}, Q={c["fluid"]["sample_flow_ul_min"]:.5g} μL/min, C={c["cell"]["concentration_per_ml"]:g}/mL;  Per group  {summary["parameters"]["cells"]}  Response .')

    def start(self,output_directory=None):
        if self.busy:return
        try:
            from ..optimization.size_study import SizeParameters
            from ..acquisition.q_exactive import AcquisitionParameters
            config=deepcopy(self.config)
            p=SizeParameters(diameters_um=[float(s.strip()) for s in self.inputs['diameters'].get().split(',')],cells=int(self.inputs['cells'].get()),
                brownian_cutoff_um=float(self.inputs['brownian'].get()),max_tip_timeout_s=float(self.inputs['timeout'].get())).validate(config)
            ap=AcquisitionParameters(signal_width_ms=float(self.inputs['width'].get())).validate()
            output=Path(output_directory) if output_directory else Path('outputs/size_studies')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        except ValueError as error:messagebox.showerror('尺寸参数',str(error),parent=self.window);return
        messages=self.messages;self.busy=True;self.run_button.configure(state='disabled');self.progress.start(50)
        self.status.set('正在求解共用载液场; 0.5 μm 随机长管运输耗时可能为数分钟.')
        def work():
            try:
                from ..optimization.size_study import analyze,save_data
                from ..visualization.size_comparison import render
                notify=lambda s:messages.put(('progress',s))
                result=analyze(config,p,ap,progress=notify);save_data(result,output,source='GUI copied experimental configuration')
                render(result,output,progress=notify);messages.put(('done',output))
            except Exception:messages.put(('error',traceback.format_exc()))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=='progress':self.status.set(payload);continue
                self.busy=False;self.run_button.configure(state='normal');self.progress.stop()
                if kind=='done':self.load_result(payload)
                else:self.status.set(' Computation failed ; 可修改设置后重试.');messagebox.showerror('尺寸对比',payload,parent=self.window)
        except queue.Empty:pass
        self.job=self.window.after(100,self.poll)

    def open_pdf(self):
        if self.output:
            import os
            os.startfile(str(self.output/'cell_size_comparison.pdf'))

    def cleanup(self):
        if self.closed:return
        self.closed=True;self.progress.stop()
        if self.job is not None:
            try:self.window.after_cancel(self.job)
            except tk.TclError:pass
        self.job=None

    def close(self):
        if self.closed:return
        self.cleanup();self.window.destroy()
