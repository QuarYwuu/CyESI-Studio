"""Interactive, inexpensive acquisition estimates; no CFD rerun required."""
from dataclasses import fields
from datetime import datetime
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from ..acquisition.q_exactive import AcquisitionParameters,analyze,sample_timeline
from ..cell_state.analysis import outlet_rates


class AcquisitionPanel(ttk.Frame):
    def __init__(self,parent,data,state_provider=None,on_run=None):
        super().__init__(parent,padding=9)
        self.data=data;self.parameters=AcquisitionParameters();self.result=None;self.output=None
        self.state_provider=state_provider;self.on_run=on_run
        self.closed=False;self.job=None;self.poll_job=None;self.exporting=False;self.messages=queue.Queue();self.process_windows=[]
        self.source_text=tk.StringVar();self.summary_text=tk.StringVar();self.status=tk.StringVar()
        ttk.Label(self,textvariable=self.source_text,wraplength=1050,font=('Microsoft YaHei UI',10,'bold')).pack(fill='x',pady=(0,6))
        ttk.Label(self,text='Full MS  Continuous acquisition . Actual  IT 可能短于 Maximum IT; 下表是采样机会与来源纯度, 真实检出率还需信噪比和识别效率校准.',
            wraplength=1050).pack(fill='x')
        form=ttk.LabelFrame(self,text='方法已知: Maximum IT 30 ms, Microscans 1; 其余为可改情景',padding=6);form.pack(fill='x',pady=6)
        self.entries={}
        inputs=[('max_injection_time_ms','Maximum IT (ms)'),('actual_injection_times_ms',' Actual  IT  Model settings  (ms)'),
            ('microscans','Microscans'),('overhead_ms','控制/转移附加时间 (ms)'),
            ('signal_width_ms','离子信号持续宽度 (ms)'),('extra_record_delay_ms','额外谱间等待 (ms)'),
            ('signal_delivery_fraction',' Exited →离子信号比例 (0–1)'),('single_event_identification_efficiency','单事件识别效率 (空= Unknown )'),
            ('purity_target','非空谱纯度目标 (0–1)'),('observation_durations_s','观察时间情景 (s)')]
        for i,(key,label) in enumerate(inputs):
            row,column=divmod(i,2);column*=2
            ttk.Label(form,text=label).grid(row=row,column=column,sticky='w',padx=(2,6),pady=2)
            value=getattr(self.parameters,key)
            var=tk.StringVar(value=', '.join(f'{x:g}' for x in value) if isinstance(value,list) else '' if value is None else str(value))
            self.entries[key]=var
            ttk.Entry(form,textvariable=var,width=16).grid(row=row,column=column+1,sticky='ew',padx=(0,12),pady=2)
        form.columnconfigure(1,weight=1);form.columnconfigure(3,weight=1)
        bar=ttk.Frame(self);bar.pack(fill='x')
        ttk.Button(bar,text='更新理论速率',command=self.recalculate).pack(side='left')
        self.export_button=ttk.Button(bar,text='导出表格 / 图 /  Report ',command=self.export);self.export_button.pack(side='left',padx=5)
        ttk.Button(bar,text='打开采集报告',command=self.open_report).pack(side='left',padx=5)
        ttk.Button(bar,text='加载采集参数',command=self.load).pack(side='left',padx=5)
        ttk.Button(bar,text=' Import measured acquisition gates  CSV',command=self.import_timing).pack(side='left',padx=5)
        ttk.Button(bar,text=' Continuous cell transport  / 采集回收动画',command=self.open_process).pack(side='left',padx=5)
        table=ttk.Frame(self);table.pack(fill='both',expand=True,pady=7)
        columns=[('resolution',' Resolution @200',85),('it',' Actual IT ms',77),('transient',' Transient  ms',72),
            ('cycle',' Scan cycle  ms',82),('hz',' Scan /s',64),('purity','非空谱单细胞%',105),
            ('mixed','全部谱混合%',100),('unique',' Unique single-source /s',105),('minute','去重/min',86),('spectra','纯谱/s',77)]
        self.table=ttk.Treeview(table,columns=[x[0] for x in columns],show='headings',height=9,selectmode='browse')
        for key,title,width in columns:
            self.table.heading(key,text=title);self.table.column(key,width=width,minwidth=width,anchor='e',stretch=True)
        vertical=ttk.Scrollbar(table,orient='vertical',command=self.table.yview)
        horizontal=ttk.Scrollbar(table,orient='horizontal',command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        self.table.grid(row=0,column=0,sticky='nsew');vertical.grid(row=0,column=1,sticky='ns');horizontal.grid(row=1,column=0,sticky='ew')
        table.columnconfigure(0,weight=1);table.rowconfigure(0,weight=1)
        self.table.bind('<<TreeviewSelect>>',lambda e:self.select_row())
        ttk.Label(self,textvariable=self.summary_text,wraplength=1050,justify='left').pack(fill='x',pady=4)
        self.timeline=tk.Canvas(self,height=144,bg='#f1f5f8',highlightthickness=0);self.timeline.pack(fill='x')
        self.timeline.bind('<Configure>',lambda e:self.draw_timeline())
        ttk.Label(self,textvariable=self.status,wraplength=1050).pack(fill='x',pady=3)
        self.bind('<Destroy>',lambda e:self.cleanup() if e.widget is self else None,add='+')
        for var in self.entries.values():var.trace_add('write',self.schedule)
        self.recalculate();self.poll_job=self.after(100,self.poll)

    def collect(self):
        values=self.parameters.to_dict()
        for key,var in self.entries.items():
            text=var.get().strip();declared=next(f.type for f in fields(self.parameters) if f.name==key)
            if isinstance(values[key],list):value=[float(x.strip()) for x in text.replace(', ',',').split(',') if x.strip()]
            elif not text and key=='single_event_identification_efficiency':value=None
            elif declared is int:value=int(text)
            else:value=float(text)
            values[key]=value
        return AcquisitionParameters(**values).validate()

    def import_timing(self,path=None,output_directory=None):
        path=path or filedialog.askopenfilename(parent=self,title='导入真实注入门起点与实际 IT ( Reduced  RAW 读出时间)',filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            from ..acquisition.measured_timing import load_gates,analyze_gates,export
            p=self.collect()
            if p.microscans!=1:raise ValueError('实际采集门导入目前要求 Microscans=1')
            rate=outlet_rates(self.data.config,self.data.summary,self.data.rate_config)['nozzle_cells_s']
            result=analyze_gates(load_gates(path),rate,p.signal_width_ms,p.signal_delivery_fraction,p.single_event_identification_efficiency,source=path)
            target=Path(output_directory) if output_directory else Path('outputs')/'measured_acquisition'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output=export(result,target)
            self.status.set(f'已导入 {result["gate_count"]}  acquisition gates ; 条件去重单细胞机会 {result["unique_pure_opportunities_s"]:.4g}/s. Report : {output}')
            import webbrowser
            webbrowser.open((output/'measured_timing_report.html').as_uri())
        except (ValueError,OSError,KeyError) as error:messagebox.showerror(' Acquisition gates  CSV',str(error),parent=self)

    def schedule(self,*args):
        if self.closed:return
        if self.job is not None:self.after_cancel(self.job)
        self.job=self.after(250,self.recalculate)

    def set_data(self,data):
        self.data=data;self.recalculate()
        for window in self.process_windows:
            if not window.closed:window.set_data(data)

    def open_process(self):
        from .process import ProcessWindow
        p=self.collect();selection=self.table.selection()
        row=self.result['rows'][int(selection[0])] if self.result and selection else None
        window=ProcessWindow(self.winfo_toplevel(),self.data,p,
            resolution=row['resolution_at_mz200'] if row else 17500,
            actual_it=row['actual_injection_ms'] if row else p.actual_injection_times_ms[-1],
            state_parameters=self.state_provider() if self.state_provider else None,on_run=self.on_run)
        self.process_windows.append(window);return window

    def set_pending(self,config):
        for window in self.process_windows:
            if not window.closed:window.set_pending(config)

    def recalculate(self):
        if self.closed:return
        if self.job is not None:self.after_cancel(self.job);self.job=None
        try:
            p=self.collect();nozzle=outlet_rates(self.data.config,self.data.summary,self.data.rate_config)['nozzle_cells_s']
            result=analyze(nozzle,p,self.data.source,self.data.config.to_dict(),self.data.rate_config.to_dict(),self.data.rate_preview)
        except (ValueError,TypeError) as error:
            self.result=None;self.output=None
            self.table.delete(*self.table.get_children());self.timeline.delete('all');self.summary_text.set('')
            self.status.set(f'输入未完成或无效: {error}');return
        old=self.result
        changed=old!=result
        if changed:self.output=None
        self.parameters=p;self.result=result
        selection=self.table.selection();selected=int(selection[0]) if selection else (len(p.actual_injection_times_ms)-1)*4
        self.table.delete(*self.table.get_children())
        for i,row in enumerate(result['rows']):
            purity='—' if row['nonempty_single_cell_purity'] is None else f"{row['nonempty_single_cell_purity']*100:.1f}"
            self.table.insert('', 'end',iid=str(i),values=(f"{row['resolution_at_mz200']:,}",f"{row['actual_injection_ms']:g}",
                f"{row['transient_ms']:g}",f"{row['record_cycle_ms']:.1f}",f"{row['spectra_per_s']:.2f}",purity,
                f"{row['mixed_spectrum_probability']*100:.1f}",f"{row['unique_pure_single_cell_events_s']:.3f}",
                f"{row['unique_pure_single_cell_events_s']*60:.1f}",f"{row['pure_single_cell_spectra_s']:.3f}"))
        self.table.selection_set(str(min(selected,len(result['rows'])-1)));self.table.see(self.table.selection()[0])
        self.source_text.set(f"源浓度 {self.data.config.cell.concentration_per_ml:g} cells/mL;  Predicted outlet rate  {nozzle:.2f}/s →  Specified  {p.signal_delivery_fraction:.0%} 形成离子包: {result['ion_packet_rate_s']:.2f}/s")
        if self.data.rate_preview:self.source_text.set(self.source_text.get()+'; 新流量预览: 暂用上次出管比例, 运输待重算')
        self.status.set('四档瞬态来自 Thermo 官方资料; 附加时间与矩形信号宽度未校准.点击一行查看采集门和细胞信号.')
        self.select_row()

    def select_row(self):
        selection=self.table.selection()
        if self.result is None or not selection:return
        row=self.result['rows'][int(selection[0])]
        counts='; '.join(f"{x['duration_s']:g} s 期望 {x['unique_pure_single_cell_events']:.1f}  Count " for x in row['expected_windows'])
        measured=' Detection calibration pending : 尚无单事件识别效率' if row['conditional_detected_unique_cells_s'] is None else f"按输入识别效率,  Detection model  {row['conditional_detected_unique_cells_s']:.3f}/s"
        recovery='—' if row['nozzle_pure_recovery_fraction'] is None else f"{row['nozzle_pure_recovery_fraction']:.1%}"
        self.summary_text.set(f"选中: {row['resolution_at_mz200']:,},  Actual  IT {row['actual_injection_ms']:g} ms.去重单细胞机会 {row['unique_pure_single_cell_events_s']:.3f}/s (占出管 {recovery}); {counts}.\n"
            f"{measured}.纯谱数量与不同细胞数量分别统计; 长信号跨谱时只算一个细胞.")
        self.draw_timeline()

    def draw_timeline(self):
        if self.closed:return
        canvas=self.timeline;canvas.delete('all');selection=self.table.selection()
        if self.result is None or not selection:return
        row=self.result['rows'][int(selection[0])]
        duration=max(1.2,2*row['record_cycle_ms']*1e-3)
        sampled=sample_timeline(row,self.parameters,self.result['ion_packet_rate_s'],duration)
        width=max(350,canvas.winfo_width());left=90;right=width-15
        def x(t):return left+(right-left)*max(0.,min(duration,t))/duration
        canvas.create_text(7,14,text='同一条 Poisson 信号流的采集示例; 不作为速率估计',anchor='w',fill='#40576d')
        canvas.create_text(7,46,text='离子收集门',anchor='w')
        pure=set();mixed=set();captured=set()
        for scan in sampled['scans']:
            ids=scan['cell_ids'];captured.update(ids)
            if len(ids)==1:pure.update(ids)
            elif ids:mixed.update(ids)
            color='#229d91' if len(ids)==1 else '#d58353' if ids else '#a4b5c5'
            for a,b in scan['gates']:canvas.create_rectangle(x(a),33,x(b),60,fill=color,outline='')
        canvas.create_text(7,92,text='细胞来源包',anchor='w')
        for i,t in enumerate(sampled['arrivals_s']):
            color='#168979' if i in pure else '#cc7646' if i in captured else '#a4b5c5'
            end=t+sampled['signal_width_s']
            canvas.create_line(x(t),78,x(t),101,fill=color,width=2)
            if end>t:canvas.create_line(x(t),90,x(end),90,fill=color,width=3)
        for i in range(6):
            tick=duration*i/5
            canvas.create_text(x(tick),114,text=f'{tick:.2f}s',anchor='n',fill='#53697d')
        canvas.create_text(7,136,text='绿: 有纯谱归属的细胞;  Orange : 仅混合; 灰: 未采到.细胞跨多张纯谱时仍只计一次.',anchor='w',fill='#53697d')

    def load(self):
        path=filedialog.askopenfilename(parent=self,filetypes=[('JSON','*.json')])
        if not path:return
        try:
            self.parameters=AcquisitionParameters.load(path)
            for key,var in self.entries.items():
                value=getattr(self.parameters,key)
                var.set(', '.join(f'{x:g}' for x in value) if isinstance(value,list) else '' if value is None else str(value))
            self.recalculate()
        except (ValueError,TypeError,OSError) as error:messagebox.showerror('采集参数加载失败',str(error),parent=self)

    def export(self):
        self.recalculate()
        if self.result is None or self.exporting:return
        result=self.result;messages=self.messages
        source=Path(self.data.source);source=source if source.is_dir() else Path('outputs')
        output=source/'acquisition'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.exporting=True;self.export_button.configure(state='disabled');self.status.set('正在导出采集速率表, 图与报告…')
        def worker():
            try:
                from ..acquisition.report import export
                path=export(result,output);messages.put(('done',(path,result)))
            except Exception:messages.put(('error',traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,payload=self.messages.get_nowait();self.exporting=False;self.export_button.configure(state='normal')
                if kind=='done':
                    path,snapshot=payload
                    if snapshot==self.result:self.output=path;self.status.set(str(path/'acquisition_report.html'))
                    else:self.status.set('已导出旧参数快照; 当前参数已改变, 请重新导出.')
                else:messagebox.showerror('采集报告导出失败',payload,parent=self)
        except queue.Empty:pass
        self.poll_job=self.after(100,self.poll)

    def open_report(self):
        if self.output:
            import webbrowser
            webbrowser.open((self.output/'acquisition_report.html').as_uri())
        else:self.export()

    def cleanup(self):
        if self.closed:return
        self.closed=True
        for window in self.process_windows:
            if not window.closed:window.close()
        for job in (self.job,self.poll_job):
            if job is not None:
                try:self.after_cancel(job)
                except tk.TclError:pass
        self.job=None;self.poll_job=None
