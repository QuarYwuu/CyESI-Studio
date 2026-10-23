"""Two interactive native scenes, built on exported or live simulation data."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk, Image
from ..visualization.explorer_data import ExplorerData, ExplanationSettings, find_results
from ..visualization.explorer_scene import SceneRenderer, BG, PANEL, WHITE, MUTED, GAS, AMBER


def hexcolor(color): return "#"+"".join(f"{v:02x}" for v in color)


class ExplorerWindow:
    def __init__(self,parent,data,page="device",standalone=False,on_parameters=None,on_run=None):
        self.window=parent if standalone else tk.Toplevel(parent)
        self.window.title("CyESI · 装置与喷口可视化")
        width=min(1460,self.window.winfo_screenwidth()-60)
        height=min(1000,self.window.winfo_screenheight()-85)
        self.window.geometry(f"{width}x{height}");self.window.minsize(min(1040,width),min(760,height))
        self.window.configure(bg=hexcolor(BG))
        self._on_parameters=on_parameters;self._on_run=on_run;self._parameter_window=None;self.state_windows=[]
        self.trajectory_windows=[]
        self.pending_config=None;self.pending_text=tk.StringVar()
        self.data=data;self.renderer=SceneRenderer(data);self.selected=data.default_cell
        self.page=page;self.explanation=False;self.settings=ExplanationSettings()
        self.progress=.78 if page=="device" else 0.
        self.saved_progress={"device":.78,"nozzle":0.,"explanation":.69}
        self.phase=0.;self.playing=True;self.dirty=True;self.closed=False;self.photo=None;self.last_image=None
        self.last_size=None;self.last_tick=time.perf_counter();self.last_render=0.;self.job=None;self._updating=False
        self.display_scale=1.;self.display_offset=(0.,0.)
        self.show_field=tk.BooleanVar(value=True);self.show_cells=tk.BooleanVar(value=True)
        self.show_streams=tk.BooleanVar(value=True);self.loop=tk.BooleanVar(value=True)
        self.progress_var=tk.DoubleVar(value=self.progress*100);self.speed=tk.StringVar(value="1×")
        self.cell_var=tk.StringVar(value=f" Cell  #{data.ids[self.selected]}")
        self.source_var=tk.StringVar();self.time_var=tk.StringVar();self.help_var=tk.StringVar(value="点击代表细胞可切换跟踪对象; 播放速度经过分段缩放, 时间读数采用物理时间.")
        self.assumption_vars={name:tk.StringVar(value=str(value)) for name,value in vars(self.settings).items() if name!="log_radius_spread"}
        self._build()
        pending=tk.Frame(self.window,bg='#fff0d8');pending.pack(fill='x',padx=23,pady=2,before=self.window.winfo_children()[0])
        tk.Label(pending,textvariable=self.pending_text,bg='#fff0d8',fg='#8c4a10',wraplength=1100,anchor='w',justify='left').pack(side='left',fill='x',expand=True)
        ttk.Button(pending,text='应用参数并运行',command=self.apply_parameters).pack(side='right')
        self.pending_bar=pending;self.pending_bar.pack_forget()
        self._update_source();self._update_page_style()
        self.window.protocol("WM_DELETE_WINDOW",self.close)
        self.window.bind("<Destroy>",lambda e:self._cleanup() if e.widget is self.window else None,add="+")
        self.window.bind("<space>",lambda e:self.toggle_play() if not isinstance(e.widget,(tk.Entry,ttk.Entry,ttk.Spinbox)) else None)
        self.window.bind("<Escape>",lambda e:self.window.attributes("-fullscreen",False))
        self.window.bind("<F11>",lambda e:self.window.attributes("-fullscreen",not bool(self.window.attributes("-fullscreen"))))
        self.job=self.window.after(40,self._tick)

    def label(self,parent,text=None,variable=None,size=11,color=WHITE,**kwargs):
        return tk.Label(parent,text=text,textvariable=variable,font=("Microsoft YaHei UI",size),
            bg=parent.cget("bg"),fg=hexcolor(color),**kwargs)

    def button(self,parent,text,command,color=WHITE):
        return tk.Button(parent,text=text,command=command,bg="#183349",fg=hexcolor(color),
            activebackground="#244b65",activeforeground="white",relief="flat",bd=0,
            padx=13,pady=8,font=("Microsoft YaHei UI",10),cursor="hand2")

    def _build(self):
        root=self.window
        top=tk.Frame(root,bg=hexcolor(BG));top.pack(fill="x",padx=20,pady=(12,5))
        self.label(top,"CyESI  /  VISUAL LAB",size=17,color=GAS).pack(side="left")
        nav=tk.Frame(top,bg=hexcolor(BG));nav.pack(side="right")
        self.device_button=self.button(nav,"① 全装置与细胞运输",lambda:self.set_page("device"))
        self.device_button.pack(side="left",padx=4)
        self.nozzle_button=self.button(nav,"②  Emitter , 液滴与细胞",lambda:self.set_page("nozzle"))
        self.nozzle_button.pack(side="left",padx=4)
        self.button(nav,'③ 有限球轨迹原图',self.open_trajectories).pack(side='left',padx=4)
        bar=tk.Frame(root,bg=hexcolor(PANEL));bar.pack(fill="x",padx=20,pady=4)
        self.label(bar,variable=self.source_var,size=10,color=MUTED,anchor="w").pack(side="left",padx=12,fill="x",expand=True)
        self.button(bar,"加载结果",self.load_directory).pack(side="left",padx=2)
        self.button(bar,"修改参数 / 重算",self.open_parameters).pack(side="left",padx=2)
        self.button(bar,"导出画面",self.export_frame).pack(side="left",padx=2)
        self.button(bar,"阅读说明",self.show_help).pack(side="left",padx=2)
        self.button(bar," Cell state  /  Single-source rate ",self.open_cell_state).pack(side="left",padx=2)
        self.scene=tk.Label(root,bg=hexcolor(BG),bd=0,highlightthickness=0)
        self.scene.pack(fill="both",expand=True,padx=2)
        self.scene.bind("<Configure>",lambda e:self.mark_dirty())
        self.scene.bind("<Button-1>",self.pick_cell)
        self.scene.bind("<Motion>",self.hover)
        modes=tk.Frame(root,bg=hexcolor(BG));modes.pack(fill="x",padx=20,pady=(2,3))
        self.mode_box=tk.Frame(modes,bg=hexcolor(BG));self.mode_box.pack(side="left")
        self.numeric_button=self.button(self.mode_box,"数值结果",lambda:self.set_explanation(False),GAS)
        self.numeric_button.pack(side="left",padx=(0,4))
        self.explain_button=self.button(self.mode_box," Reference models  ( Specified )",lambda:self.set_explanation(True),AMBER)
        self.explain_button.pack(side="left")
        layers=tk.Frame(modes,bg=hexcolor(BG));layers.pack(side="right")
        for text,var in [(" Electric field ",self.show_field),(" Continuous cell stream ",self.show_cells),("流线",self.show_streams),("轨迹重播",self.loop)]:
            tk.Checkbutton(layers,text=text,variable=var,command=self.mark_dirty,bg=hexcolor(BG),
                fg=hexcolor(MUTED),selectcolor="#183349",activebackground=hexcolor(BG),
                activeforeground="white",font=("Microsoft YaHei UI",10),bd=0).pack(side="left",padx=5)
        self.assumptions=tk.Frame(root,bg="#302619")
        descriptions=[("jet_diameter_um"," Specified jet diameter  μm"),("lysis_rate_s"," Specified  k Lysis  /s"),
                      ("extraction_rate_s"," Specified  k Extraction  /s"),("extra_exposure_ms","继续暴露 ms")]
        for name,label in descriptions:
            self.label(self.assumptions,label,size=10,color=AMBER).pack(side="left",padx=(12,4),pady=7)
            entry=tk.Entry(self.assumptions,textvariable=self.assumption_vars[name],width=7,
                bg="#473723",fg="#ffe2ad",insertbackground="white",relief="flat",font=("Consolas",11))
            entry.pack(side="left",padx=(0,8))
            entry.bind("<Return>",lambda e:self.apply_assumptions())
            entry.bind("<FocusOut>",lambda e:self.apply_assumptions())
        self.button(self.assumptions,"应用假设",self.apply_assumptions,AMBER).pack(side="right",padx=6)
        controls=tk.Frame(root,bg=hexcolor(PANEL));controls.pack(fill="x",padx=20,pady=(2,3))
        self.play_button=self.button(controls," Pause ",self.toggle_play,GAS);self.play_button.pack(side="left")
        self.button(controls,"从头",lambda:self.seek(0)).pack(side="left",padx=(3,10))
        self.label(controls," Cell ",size=10,color=MUTED).pack(side="left",padx=3)
        self.cell_combo=ttk.Combobox(controls,textvariable=self.cell_var,state="readonly",width=13,
            values=[f" Cell  #{i}" for i in self.data.ids])
        self.cell_combo.pack(side="left",padx=(3,10));self.cell_combo.bind("<<ComboboxSelected>>",self.select_cell)
        self.slider=tk.Scale(controls,variable=self.progress_var,from_=0,to=100,resolution=.05,orient="horizontal",
            command=self.slider_changed,showvalue=False,bg=hexcolor(PANEL),fg="white",troughcolor="#29465c",
            highlightthickness=0,activebackground=hexcolor(GAS),sliderrelief="flat",bd=0,width=8,length=300)
        self.slider.pack(side="left",expand=True,fill="x",padx=7)
        self.speed_combo=ttk.Combobox(controls,textvariable=self.speed,state="readonly",width=6,
            values=["0.25×","0.5×","1×","2×","4×"])
        self.speed_combo.pack(side="left",padx=8)
        self.button(controls,"末帧",lambda:self.seek(1)).pack(side="left")
        bottom=tk.Frame(root,bg=hexcolor(BG));bottom.pack(fill="x",padx=23,pady=(3,10))
        self.label(bottom,variable=self.time_var,size=11,color=GAS).pack(side="left")
        self.label(bottom,variable=self.help_var,size=9,color=MUTED,anchor="e").pack(side="right",fill="x",expand=True,padx=14)

    def _update_source(self):
        c=self.data.config
        name=Path(self.data.source).name if self.data.source else "当前结果"
        state=" Current unspecified ,  Open-circuit limit " if c.electrical.liquid_model=="resistive" and c.electrical.spray_current_na is None else c.electrical.liquid_model
        self.source_var.set(f"{name} |  Concentration  {c.cell.concentration_per_ml:g} cells/mL |  Sample / Sheath  {c.fluid.sample_flow_ul_min:g}/{c.fluid.sheath_flow_ul_min:g} μL/min |  Electrode  {c.electrical.voltage_kv:g} kV | {state}")
        if self.data.rate_preview:self.source_var.set(self.source_var.get()+f' | 事件源按新样品流量 {self.data.rate_config.fluid.sample_flow_ul_min:g} 预览, 流场待重算')

    def _key(self): return "device" if self.page=="device" else "explanation" if self.explanation else "nozzle"

    def set_page(self,page):
        self.saved_progress[self._key()]=self.progress
        self.page=page;self.progress=self.saved_progress[self._key()]
        self.renderer.cache.clear();self._update_page_style();self.mark_dirty()

    def set_explanation(self,value):
        self.saved_progress[self._key()]=self.progress
        self.explanation=bool(value);self.progress=self.saved_progress[self._key()]
        self.renderer.cache.clear();self._update_page_style();self.mark_dirty()

    def _update_page_style(self):
        self.device_button.configure(bg="#245c65" if self.page=="device" else "#183349")
        self.nozzle_button.configure(bg="#245c65" if self.page=="nozzle" else "#183349")
        self.numeric_button.configure(bg="#245c65" if not self.explanation else "#183349")
        self.explain_button.configure(bg="#6b4b22" if self.explanation else "#183349")
        if self.page=="device":self.mode_box.pack_forget()
        elif not self.mode_box.winfo_manager():self.mode_box.pack(side="left")
        if self.page=="nozzle" and self.explanation:
            if not self.assumptions.winfo_manager():self.assumptions.pack(fill="x",padx=20,pady=2,before=self.play_button.master)
        else:self.assumptions.pack_forget()

    def mark_dirty(self): self.dirty=True

    def seek(self,value):
        self.progress=float(value);self.playing=False;self.play_button.configure(text=" Play ")
        self.mark_dirty()

    def slider_changed(self,value):
        if self._updating:return
        self.progress=float(value)/100;self.mark_dirty()

    def toggle_play(self):
        self.playing=not self.playing;self.last_tick=time.perf_counter()
        self.play_button.configure(text=" Pause " if self.playing else " Play ")
        self.mark_dirty()

    def select_cell(self,event=None):
        self.selected=self.cell_combo.current();self.mark_dirty()

    def pick_cell(self,event):
        if self.page!="device":return
        ex=(event.x-self.display_offset[0])/self.display_scale
        ey=(event.y-self.display_offset[1])/self.display_scale
        for hit in reversed(self.renderer.hits):
            if (ex-hit["x"])**2+(ey-hit["y"])**2<=hit["radius"]**2:
                self.selected=hit["index"];self.cell_var.set(f" Cell  #{self.data.ids[self.selected]}")
                self.progress=self.data.progress_at_time(self.selected,hit["time_s"])
                self.playing=False;self.play_button.configure(text=" Play ");self.mark_dirty();break

    def hover(self,event):
        ex=(event.x-self.display_offset[0])/self.display_scale
        ey=(event.y-self.display_offset[1])/self.display_scale
        selectable=any((ex-hit["x"])**2+(ey-hit["y"])**2<=hit["radius"]**2 for hit in self.renderer.hits)
        self.scene.configure(cursor="hand2" if selectable else "")

    def apply_assumptions(self):
        try:
            values={k:float(var.get()) for k,var in self.assumption_vars.items()}
            updated=ExplanationSettings(**values).validate()
            if updated.jet_diameter_um>=self.data.config.geometry.sheath_id_um:
                raise ValueError("假设射流直径应小于中管内径")
            self.settings=updated;self.help_var.set("假设已应用; 它们只影响机理演示, 不改变数值结果或检测效率.")
            self.mark_dirty()
        except ValueError as error:self.help_var.set(f"假设参数无效: {error}")

    def _redraw(self):
        available=(self.scene.winfo_width(),self.scene.winfo_height())
        if min(available)<=2:available=(1320,760)
        size=(max(980,available[0]),max(560,available[1]))
        if size!=self.last_size:self.renderer.cache.clear();self.last_size=size
        self.last_image=self.renderer.render(*size,page=self.page,progress=self.progress,selected=self.selected,
            phase=self.phase,show_field=self.show_field.get(),show_cells=self.show_cells.get(),
            show_streams=self.show_streams.get(),explanation=self.explanation,settings=self.settings)
        self.display_scale=min(available[0]/size[0],available[1]/size[1],1.)
        display_size=tuple(max(1,round(value*self.display_scale)) for value in size)
        self.display_offset=((available[0]-display_size[0])/2,(available[1]-display_size[1])/2)
        picture=self.last_image if display_size==size else self.last_image.resize(display_size,Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(picture,master=self.window)
        self.scene.configure(image=self.photo)
        self._updating=True;self.progress_var.set(self.progress*100);self._updating=False
        if self.page=="device":
            time_s=self.data.progress_time(self.selected,self.progress)
            self.time_var.set(f" Continuous source  {self.phase:.2f} s  ·  选中轨迹样本年龄 {time_s:.3f} s (分段慢放)")
        elif self.explanation:self.time_var.set(f"机理阶段 {self.progress:.0%}  ·  不对应真实喷雾时间")
        elif self.data.spray is not None:
            sp=self.data.spray;frame=round(self.progress*(len(sp["time_s"])-1))
            self.time_var.set(f"数值时间 {sp['time_s'][frame]*1e3:.4f} ms  ·  帧 {frame+1}/{len(sp['time_s'])}")
        else:self.time_var.set("未加载两相计算数据")
        self.dirty=False

    def _tick(self):
        if self.closed:return
        now=time.perf_counter();elapsed=min(now-self.last_tick,.15);self.last_tick=now
        if self.playing:
            speed=float(self.speed.get().replace("×",""))
            self.phase+=elapsed*speed
            self.progress+=elapsed*speed/(20 if self.page=="device" else 14)
            if self.progress>=1:
                if self.loop.get():self.progress%=1
                else:self.progress=1.;self.playing=False;self.play_button.configure(text=" Play ")
            self.dirty=True
        if self.dirty and now-self.last_render>.065:
            self._redraw();self.last_render=now
        self.job=self.window.after(30,self._tick)

    def set_data(self,data):
        if self.closed:return
        self.data=data;self.renderer=SceneRenderer(data);self.selected=data.default_cell
        self.cell_combo.configure(values=[f" Cell  #{i}" for i in data.ids]);self.cell_var.set(f" Cell  #{data.ids[self.selected]}")
        for windows in (self.state_windows,self.trajectory_windows):
            for window in windows[:]:
                if window.closed:windows.remove(window)
                else:window.set_data(data)
        if self.pending_config is not None:self.set_pending(self.pending_config)
        self._update_source();self.mark_dirty()

    def set_pending(self,config):
        from .source_state import pending_message
        self.pending_config=config;self.pending_text.set(pending_message(self.data.config,config))
        if self.pending_text.get():self.pending_bar.pack(fill='x',padx=23,pady=2,before=self.window.winfo_children()[0])
        else:self.pending_bar.pack_forget()
        for windows in (self.state_windows,self.trajectory_windows):
            for window in windows:
                if not window.closed:window.set_pending(config)

    def apply_parameters(self):
        if self._on_run is not None:self._on_run();return
        self.open_parameters()
        if self._parameter_window is not None and self._parameter_window.winfo_exists():self._parameter_application.start()

    def load_directory(self):
        path=filedialog.askdirectory(parent=self.window,title="选择包含 config.json 和 fields.npz  Data ")
        if not path:return
        try:self.set_data(ExplorerData.load(path))
        except Exception as error:messagebox.showerror("结果加载失败",str(error),parent=self.window)

    def export_frame(self):
        if self.last_image is None:self._redraw()
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        name=f"cyesi_{self._key()}_{stamp}.png"
        path=filedialog.asksaveasfilename(parent=self.window,defaultextension=".png",initialfile=name,filetypes=[("PNG","*.png")])
        if path:
            self.last_image.save(path)
            self.help_var.set(f"已导出: {Path(path).name} (含数据/假设标记)")

    def open_parameters(self):
        if self._on_parameters is not None:self._on_parameters();return
        if self._parameter_window is not None and self._parameter_window.winfo_exists():
            self._parameter_window.lift();return
        from .interface import Application
        self._parameter_window=tk.Toplevel(self.window)
        self._parameter_application=Application(self._parameter_window,self.data.config,on_data=self.set_data,on_pending=self.set_pending)
        if (Path(self.data.source)/'fields.npz').exists():
            self._parameter_application.load_results(self.data.source,update_parameters=False,data=self.data)
        else:
            self._parameter_application.result_data=self.data
            self._parameter_application.update_rates()

    def open_cell_state(self):
        from .cell_state import CellStateWindow
        window=CellStateWindow(self.window,self.data,on_run=self.apply_parameters);self.state_windows.append(window)
        if self.pending_config is not None:window.set_pending(self.pending_config)
        return window

    def open_trajectories(self):
        from .trajectories import TrajectoryWindow
        for window in self.trajectory_windows:
            if not window.closed:window.window.lift();return window
        window=TrajectoryWindow(self.window,self.data);self.trajectory_windows.append(window)
        if self.pending_config is not None:window.set_pending(self.pending_config)
        return window

    def show_help(self):
        window=tk.Toplevel(self.window);window.title("可视化数据与假设");window.geometry("780x590")
        text=tk.Text(window,wrap="word",font=("Microsoft YaHei UI",11),padx=20,pady=15)
        text.pack(fill="both",expand=True)
        text.insert("1.0",
            "全装置页\n\n蓝色为含 140 mM 甲酸铵的水相, 橙色为甲醇.颜色连续过渡来自已求解的稳态浓度场."
            "连续细胞源按浓度与流量进行 Poisson  Injection ,  Events  ID 持续增加.运输分布由保存的有限球轨迹采样, 轨迹采样量不限制连续源的细胞总数."
            "细胞越过中管口后保留在下游待解析计数中; 目前没有外部细胞/液滴捕获轨迹.选中的高亮对象是可单独回放的轨迹样本."
            "70 cm 长管折叠展示, 剖面径向放大, 细胞显示尺寸也经过放大.\n\n"
            "电场来自已解电位.resistive 模式包括离子扩散电位, 远端电阻与石英电渗滑移; 未提供电流时仅为开路极限."
            "气体箭头读取两相速度历史, 标明速度比例.气相压力可在参数主界面的气体速度/压力页查看.\n\n"
            "喷口数值页\n\n液体界面取自 spray.npz 的实际体积分数, 0.5 轮廓用于显示边界."
            "液滴分布只统计实际分离且达到显示分辨要求的连通区.没有脱落液滴时不会补画."
            "当前喷雾求解没有细胞膜模型, 也没有 MS 捕获模型.\n\n"
            "机理演示页\n\n这是单独的假设情景, 不是计算得到的喷雾.49.3° 锥体为理想 Taylor 参考形态; "
            "滴径采用无电荷, 无黏性圆柱射流的 Rayleigh 体积参考, 并加入标明的展示离散度."
            "示意液滴的数量不表示实际喷雾通量, 也不表示每滴都含细胞.\n\n"
            "膜完整性假设: dI/dt = -k Lysis  C² I.胞内残留假设: dM/dt = -k Extraction  C(1-I)M."
            "I 和 M 是归一化变量; 释放量 = 1-M.实际出管前的暴露历史来自轨迹, 出管后的继续暴露时长和速率均为用户假设."
            "这些值未经生物实验校准, 不能解释为实测裂解率, 提取率或检测效率.\n\n"
            f"Current source: {self.data.source}\n\nSee README.md for installation and usage.")
        text.configure(state="disabled")

    def _cleanup(self):
        if self.closed:return
        self.closed=True
        for window in self.state_windows:
            if not window.closed:window.close()
        self.state_windows.clear()
        for window in self.trajectory_windows:
            if not window.closed:window.close()
        self.trajectory_windows.clear()
        self._on_run=None;self._on_parameters=None
        if self.job is not None:
            try:self.window.after_cancel(self.job)
            except tk.TclError:pass
            self.job=None
        self.photo=None

    def close(self):
        self._cleanup()
        self.window.destroy()


def launch(results=None,page="device"):
    root=tk.Tk();root.withdraw()
    path=Path(results) if results else find_results()
    if path is None:
        path=filedialog.askdirectory(parent=root,title="选择已计算的结果文件夹; 尚无结果时请先运行主界面")
        if not path:root.destroy();return
    try:data=ExplorerData.load(path)
    except Exception as error:
        messagebox.showerror("无法加载结果",str(error),parent=root);root.destroy();return
    root.deiconify();ExplorerWindow(root,data,page=page,standalone=True);root.mainloop()
