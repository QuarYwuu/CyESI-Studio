"""Membrane and inlet workspace with background calculation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
from .interface import Viewer
from ..cell_state.parameters import StateParameters
from ..cell_state.analysis import outlet_rates


class CellStateWindow:
    def __init__(self,parent,data,on_run=None):
        self.window=tk.Toplevel(parent);self.window.title("CyESI · 脂质膜细胞 / Q Exactive 入口状态")
        self.window.geometry("1350x880");self.window.minsize(1030,700)
        self.data=data;self.parameters=StateParameters();self.messages=queue.Queue()
        self.on_run=on_run
        self.closed=False;self.busy=False;self.job=None;self.output=None;self.analysis_result=None;self.displayed_summary=None
        self.rate_text=tk.StringVar();self.refresh_rates()
        ttk.Label(self.window,textvariable=self.rate_text,
            wraplength=1280,font=("Microsoft YaHei UI",13,"bold")).pack(fill="x",padx=16,pady=(12,6))
        ttk.Label(self.window,text="条件情景: 水相内核＋ Lipid membrane ＋外部载细胞液滴; 滴径,  Charge , 膜通透性与溶解率需要校准.320°C 为入口壁温.",
            wraplength=1240).pack(fill="x",padx=16,pady=(0,8))
        self.pending_text=tk.StringVar();pending=ttk.Frame(self.window);pending.pack(fill='x',padx=16,pady=(0,5))
        ttk.Label(pending,textvariable=self.pending_text,foreground='#a75312',wraplength=1080).pack(side='left',fill='x',expand=True)
        if on_run is not None:ttk.Button(pending,text='应用参数并运行流场',command=on_run).pack(side='right')
        outer=ttk.Panedwindow(self.window,orient="horizontal");outer.pack(fill="both",expand=True,padx=12)
        left=ttk.Frame(outer,width=380);right=ttk.Frame(outer);outer.add(left,weight=0);outer.add(right,weight=1)
        form=ttk.LabelFrame(left,text="Q Exactive 与情景参数",padding=8);form.pack(fill="x")
        fields=[("inlet_diameter_mm"," Inlet ID  mm"),("transfer_tube_length_mm","转移管长 mm"),
            ("inlet_wall_temperature_c","壁面温度 °C"),("inlet_flow_l_min","吸气 L/min (空=估算)"),
            ("flow_factors","吸气倍率"),("drop_radii_um","液滴半径 μm"),
            ("charge_fractions"," Charge  / Rayleigh 极限"),("methanol_permeabilities_m_s"," Membrane permeability  m/s"),
            ("dissolution_rates_s","假设膜溶解 k /s"),("sample_tracks","运输轨迹采样量"),
            ("alignment_offset_um","喷口横向偏移 μm")]
        self.entries={}
        for row,(key,label) in enumerate(fields):
            ttk.Label(form,text=label).grid(row=row,column=0,sticky="w",pady=3)
            value=getattr(self.parameters,key)
            if isinstance(value,list):value=", ".join(str(x) for x in value)
            var=tk.StringVar(value="" if value is None else str(value));self.entries[key]=var
            ttk.Entry(form,textvariable=var,width=20).grid(row=row,column=1,padx=(8,0),pady=3)
        self.run_button=ttk.Button(left,text="计算状态与单细胞率",command=self.start);self.run_button.pack(fill="x",pady=8)
        ttk.Button(left,text="全部参数 JSON",command=self.advanced).pack(fill="x",pady=3)
        ttk.Button(left,text="打开情景报告",command=self.open_report).pack(fill="x",pady=3)
        ttk.Button(left,text="加载情景 JSON",command=self.load).pack(fill="x",pady=3)
        ttk.Button(left,text="查看已计算结果",command=self.load_result).pack(fill="x",pady=3)
        self.details=tk.Text(left,height=15,width=43,wrap="word",font=("Microsoft YaHei UI",10))
        self.details.pack(fill="both",expand=True,pady=8)
        self.details.insert("1.0","计算结果会区分: \n\n· 出管细胞率\n· 条件性载细胞滴入孔率\n· 单细胞滴占据纯度\n· 相邻事件时间隔离率\n\n液滴破碎, 液膜耗尽及膜模型失效时, 保留待解析范围.不得把情景通过比例解释为细胞存活率.")
        self.details.configure(state="disabled")
        notebook=ttk.Notebook(right);notebook.pack(fill="both",expand=True);self.notebook=notebook
        self.views=[]
        for name in ["入口细胞状态"," Single-source rate "]:
            view=Viewer(notebook);notebook.add(view,text=name);self.views.append(view)
        from .acquisition import AcquisitionPanel
        self.acquisition=AcquisitionPanel(notebook,data,state_provider=self.collect,on_run=self.on_run)
        notebook.add(self.acquisition,text=" MS acquisition  / 理论单细胞率")
        self.status=tk.StringVar(value=f"源数据: {data.source}")
        ttk.Label(self.window,textvariable=self.status,wraplength=1280).pack(fill="x",padx=16,pady=8)
        self.window.protocol("WM_DELETE_WINDOW",self.close)
        self.window.bind("<Destroy>",lambda e:self.cleanup() if e.widget is self.window else None,add="+")
        self.job=self.window.after(100,self.poll)
        ready=Path(data.source)/"cell_state"
        if data.rate_preview:
            self.run_button.configure(state='disabled')
            self.status.set('按新样品流量预览速率; 请先应用参数并运行流场,  Then compute membrane state .')
        elif (ready/"state_summary.json").exists():
            try:self.load_output(ready)
            except (ValueError,OSError,KeyError):pass

    def refresh_rates(self):
        rates=outlet_rates(self.data.config,self.data.summary,self.data.rate_config)
        nozzle=" Unknown " if rates['nozzle_cells_s'] is None else f"{rates['nozzle_cells_s']:.2f} cells/s"
        mode='出管预览 (暂用上次出管比例)' if self.data.rate_preview else ' Steady model outlet rate '
        self.rate_text.set(f" Concentration  {self.data.config.cell.concentration_per_ml:g} cells/mL     Continuous injection  {rates['injection_cells_s']:.2f} cells/s    {mode} {nozzle}    生物完整率 / MS 检出率尚未知")

    def set_pending(self,config):
        from .source_state import pending_message
        self.pending_text.set(pending_message(self.data.config,config))
        self.acquisition.set_pending(config)

    def clear_output(self,all_views=False):
        self.output=None;self.displayed_summary=None
        for view in self.views if all_views else self.views[1:]:
            view.source=None;view.photo=None
            view.label.configure(image="",text="参数已更新, 等待当前参数的计算结果")

    def set_data(self,data):
        if self.closed:return
        old=self.data;self.data=data;self.pending_text.set('');self.refresh_rates();self.acquisition.set_data(data)
        if old.source==data.source and old.config.to_dict()==data.config.to_dict() and old.rate_config.to_dict()==data.rate_config.to_dict():return
        if data.rate_preview:
            self.clear_output();self.run_button.configure(state='disabled')
            self.status.set('新样品流量的注入率与理论采集率已预览; 膜状态和有限窗统计需先重新计算流场.')
            return
        if not self.busy:self.run_button.configure(state='normal')
        from ..particle.events import concentration_only_change
        reusable=old.source==data.source and concentration_only_change(old.config,data.config)
        self.clear_output(all_views=not reusable)
        if not reusable:
            self.analysis_result=None
            self.details.configure(state="normal");self.details.delete("1.0","end")
            self.details.insert("1.0","流场或运输参数已更新, 请计算当前参数的细胞状态与入孔率.")
            self.details.configure(state="disabled")
            self.status.set("源数据已更新; 旧情景结果已失效.")
        elif self.analysis_result is not None:
            self.status.set("浓度已更新, 正在重算入孔事件率和单细胞滴占据率; 复用膜/飞行轨迹.")
            if not self.busy:self.launch_worker(self.parameters,self.analysis_result)
        else:self.status.set("浓度与连续事件统计已更新; 可计算细胞状态与条件入孔率.")

    def collect(self):
        values=self.parameters.to_dict()
        for key,var in self.entries.items():
            text=var.get().strip();old=values[key]
            if isinstance(old,list):value=[float(x.strip()) for x in text.replace(", ",",").split(",") if x.strip()]
            elif key=="sample_tracks":value=int(text)
            elif key=="inlet_flow_l_min" and not text:value=None
            else:value=float(text)
            values[key]=value
        return StateParameters(**values).validate()

    def refresh(self):
        for key,var in self.entries.items():
            value=getattr(self.parameters,key)
            var.set(", ".join(str(x) for x in value) if isinstance(value,list) else "" if value is None else str(value))

    def advanced(self):
        try:p=self.collect()
        except ValueError as e:messagebox.showerror("参数错误",str(e),parent=self.window);return
        dialog=tk.Toplevel(self.window);dialog.title(" Lipid membrane  / 微滴 / 入口全部参数");dialog.geometry("780x700")
        editor=tk.Text(dialog,wrap="none",font=("Consolas",11));editor.pack(fill="both",expand=True)
        editor.insert("1.0",json.dumps(p.to_dict(),ensure_ascii=False,indent=2))
        def apply():
            try:self.parameters=StateParameters(**json.loads(editor.get("1.0","end"))).validate();self.refresh();dialog.destroy()
            except (ValueError,TypeError) as e:messagebox.showerror("参数错误",str(e),parent=dialog)
        ttk.Button(dialog,text="验证并应用",command=apply).pack(pady=5)

    def load(self):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[("JSON","*.json")])
        if not path:return
        try:self.parameters=StateParameters.load(path);self.refresh()
        except (ValueError,TypeError) as e:messagebox.showerror("参数错误",str(e),parent=self.window)

    def start(self):
        if self.busy:return
        if self.data.rate_preview:self.status.set('请先应用样品流量并重新运行流场.');return
        try:self.parameters=self.collect()
        except ValueError as e:messagebox.showerror("参数错误",str(e),parent=self.window);return
        self.launch_worker(self.parameters)

    def launch_worker(self,parameters,previous=None):
        data=self.data;messages=self.messages
        source=Path(data.source)
        if not source.is_dir():source=Path("outputs")
        # A subdirectory does not overwrite the CFD result or its continuous stream.
        output=source/"cell_state"/datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        self.busy=True;self.run_button.configure(state="disabled")
        def worker():
            try:
                from ..cell_state.analysis import analyze,reweight_concentration
                from ..cell_state.report import export
                result=(analyze(data,parameters,progress=lambda text:messages.put(("log",text))) if previous is None
                        else reweight_concentration(previous,data))
                path=export(result,output);messages.put(("done",(path,result)))
            except Exception:messages.put(("error",traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def load_result(self):
        path=filedialog.askdirectory(parent=self.window,title="选择含 state_summary.json 的细胞状态结果文件夹")
        if not path:return
        try:self.load_output(path)
        except (ValueError,OSError,KeyError) as e:messagebox.showerror("结果加载失败",str(e),parent=self.window)

    def load_output(self,path):
        if self.data.rate_preview:raise ValueError("Sample flow changed. Recompute the flow solution before updating this preview.")
        path=Path(path)
        summary=json.loads((path/"state_summary.json").read_text(encoding="utf-8"))
        if Path(summary['source']).resolve()!=Path(self.data.source).resolve():
            raise ValueError("These parameters belong to another flow solution. Load the corresponding results.")
        from ..particle.events import concentration_only_change
        from ..cell_state.report import load_analysis
        if summary.get('source_config') is None or not concentration_only_change(summary['source_config'],self.data.config):
            raise ValueError("情景结果的源 CFD 参数与当前数据不同")
        self.parameters=StateParameters.load(path/"state_parameters.json");self.refresh()
        self.analysis_result=load_analysis(path)
        if summary['source_config']!=self.data.config.to_dict():
            self.clear_output(all_views=True)
            self.status.set("正在按当前浓度更新已保存情景的事件率和滴占据率.")
            if not self.busy:self.launch_worker(self.parameters,self.analysis_result)
        else:self.show_output(path,summary)

    def show_output(self,path,summary):
        self.output=Path(path);self.displayed_summary=summary;self.refresh_rates()
        for view,name in zip(self.views,["cell_state","single_cell_rates"]):view.load(self.output/f"{name}.png")
        self.details.configure(state="normal");self.details.delete("1.0","end")
        self.details.insert("1.0",f" Reference aspiration estimate : {summary['inlet']['baseline_ambient_flow_l_min']:.3g} L/min\n\n情景终态计数 (并非概率): \n"+
            "\n".join(f"{key}: {value}" for key,value in summary['scenario_counts'].items())+
            "\n\n完整活细胞率与 MS 检出率未校准, 保持未知.\n\n全部参数, 入口状态和单细胞率 CSV 可从报告复核.")
        self.details.configure(state="disabled");self.status.set(str(self.output/"cell_state_report.html"))

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=="log":self.status.set(payload)
                elif kind=="done":
                    path,result=payload;self.busy=False;self.run_button.configure(state="normal")
                    from ..particle.events import concentration_only_change
                    from ..cell_state.report import summarize
                    same_source=result.source==self.data.source
                    if self.data.rate_preview:
                        self.analysis_result=result;self.run_button.configure(state='disabled')
                        self.status.set('原参数情景已完成; 新样品流量需要先重新计算流场.')
                    elif same_source and result.source_config==self.data.config.to_dict():
                        self.analysis_result=result;self.show_output(path,summarize(result))
                    elif same_source and concentration_only_change(result.source_config,self.data.config):
                        # Discard completed work when its inputs have changed.
                        self.analysis_result=result
                        self.launch_worker(self.parameters,result)
                    else:self.status.set("计算期间源数据已改变; 请计算当前参数的细胞状态.")
                elif kind=="error":
                    self.busy=False;self.run_button.configure(state="normal");self.status.set("情景计算失败")
                    messagebox.showerror("情景计算失败",payload,parent=self.window)
        except queue.Empty:pass
        self.job=self.window.after(100,self.poll)

    def open_report(self):
        if self.output:
            import webbrowser
            webbrowser.open((self.output/"cell_state_report.html").as_uri())

    def cleanup(self):
        if self.closed:return
        self.closed=True
        if self.job:
            try:self.window.after_cancel(self.job)
            except tk.TclError:pass
            self.job=None
        for view in self.views:view.close()
        self.acquisition.cleanup()
        self.on_run=None

    def close(self):
        self.cleanup();self.window.destroy()
