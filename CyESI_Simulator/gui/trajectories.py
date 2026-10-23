"""Restore the original numerical trajectory view from saved CFD arrays."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk
from .interface import Viewer
from .source_state import pending_message


class TrajectoryWindow:
    def __init__(self,parent,data):
        self.window=tk.Toplevel(parent);self.window.title('CyESI ·  Continuous cell stream  /  Finite-sphere trajectories ')
        self.window.geometry('1180x760');self.window.minsize(850,550)
        self.data=data;self.closed=False;self.busy=False;self.job=None;self.revision=0;self.messages=queue.Queue()
        self.pending_config=None;self.output=None
        self.source_text=tk.StringVar();self.status=tk.StringVar();self.pending_text=tk.StringVar()
        ttk.Label(self.window,text='细胞轨迹 | 稳态连续流 / 原始响应对比',
            font=('Microsoft YaHei UI',13,'bold')).pack(fill='x',padx=14,pady=(12,4))
        ttk.Label(self.window,textvariable=self.source_text,wraplength=1100).pack(fill='x',padx=14)
        ttk.Label(self.window,textvariable=self.pending_text,foreground='#a75312',wraplength=1100).pack(fill='x',padx=14,pady=4)
        notebook=ttk.Notebook(self.window);notebook.pack(fill='both',expand=True,padx=10,pady=6)
        from .trajectory_stream import TrajectoryStreamPanel
        self.continuous=TrajectoryStreamPanel(notebook,data);notebook.add(self.continuous,text=' Continuous feed  / 全程物理时钟')
        self.animation=Viewer(notebook);notebook.add(self.animation,text='有限球轨迹动画 (原图)')
        self.static=Viewer(notebook);notebook.add(self.static,text='轨迹 / 出口分布')
        bar=ttk.Frame(self.window);bar.pack(fill='x',padx=14,pady=6)
        self.restore_button=ttk.Button(bar,text='从已有轨迹恢复动画',command=self.restore);self.restore_button.pack(side='left')
        ttk.Button(bar,text='打开动画文件',command=self.open_file).pack(side='left',padx=6)
        ttk.Label(bar,text='原图为同龄响应对比, 可能重合; 连续进样页保留长管位置与真实到达时差.').pack(side='left',padx=8)
        ttk.Label(self.window,textvariable=self.status,wraplength=1100).pack(fill='x',padx=14,pady=(0,8))
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.window.bind('<Destroy>',lambda e:self.cleanup() if e.widget is self.window else None,add='+')
        self.set_data(data);self.job=self.window.after(100,self.poll)

    def set_pending(self,config):
        self.pending_config=config;self.pending_text.set(pending_message(self.data.config,config))
        self.continuous.set_pending(config)

    def set_data(self,data):
        if self.closed:return
        old=self.data;self.data=data
        self.continuous.set_data(data)
        self.source_text.set(f'数据源: {data.source}; 保存 {len(data.ids)} 条完整响应; 连续页按不同注入时刻回放, 原 GIF 按同龄对齐.')
        if self.pending_config is not None:self.set_pending(self.pending_config)
        # Concentration and rate previews do not change solved trajectories.
        same_tracks=(old.fields['trajectory_time_s'] is data.fields['trajectory_time_s']
            and old.config.geometry==data.config.geometry and old.config.cell.diameter_um==data.config.cell.diameter_um)
        if same_tracks and self.output is not None:return
        self.revision+=1;self.output=None
        for view in (self.animation,self.static):
            view.clear('正在加载已计算的轨迹…')
        source=Path(data.source)
        static=source/'figure_04_trajectories.png'
        if static.exists():self.static.load(static)
        else:self.static.label.configure(text='此数据源尚未导出静态轨迹图; 动画可直接从数值轨迹恢复.')
        gif=source/'cell_trajectories.gif'
        if gif.exists():
            self.output=gif.resolve();self.animation.play(gif);self.status.set(str(gif.resolve()))
        else:self.restore()

    def restore(self):
        if self.closed or self.busy:return
        data=self.data;messages=self.messages;revision=self.revision
        # Keep generated views separate from source CFD and saved reports.
        output=Path('outputs')/'trajectory_views'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        f=data.fields
        particles=SimpleNamespace(saved_ids=data.ids,trajectory_time=f['trajectory_time_s'],
            trajectory_r=f['trajectory_r_m'],trajectory_z=f['trajectory_z_m'])
        result=SimpleNamespace(config=data.config,particles=particles)
        self.busy=True;self.restore_button.configure(state='disabled')
        self.status.set('正在从已有轨迹生成原版有限球动画,  Using the current  CFD…')
        def worker():
            try:
                from ..visualization.figures import save_trajectory_animation
                path=save_trajectory_animation(result,output)
                if path is None:raise ValueError('保存的轨迹帧数不足, 无法恢复动画')
                messages.put(('done',(revision,path.resolve())))
            except Exception:messages.put(('error',(revision,traceback.format_exc())))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,(revision,payload)=self.messages.get_nowait()
                self.busy=False;self.restore_button.configure(state='normal')
                if revision!=self.revision:
                    if self.output is None:self.restore()
                    continue
                if kind=='done':self.output=payload;self.animation.play(payload);self.status.set(str(payload))
                else:self.status.set('动画恢复失败: '+payload)
        except queue.Empty:pass
        self.job=self.window.after(100,self.poll)

    def open_file(self):
        if self.output:
            import webbrowser
            webbrowser.open(self.output.as_uri())

    def cleanup(self):
        if self.closed:return
        self.closed=True
        if self.job is not None:
            try:self.window.after_cancel(self.job)
            except tk.TclError:pass
        self.job=None;self.animation.close();self.static.close();self.continuous.close()

    def close(self):
        self.cleanup();self.window.destroy()
