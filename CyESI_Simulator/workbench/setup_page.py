from copy import deepcopy
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
from ..config import Config
from ..gui.interface import INPUTS
from ..fluid.supply import convert_pressure,infer_pressure
from .widgets import *


class SetupPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.session=app.session;self.vars={};self.inputs={};self.muted=False
        self.columnconfigure(0,weight=3);self.columnconfigure(1,weight=2);self.rowconfigure(0,weight=1)
        left=Card(self,'实验参数','参数草稿与已计算结果分开保存.修改浓度即时更新稳态统计; 物理场按需计算.')
        left.grid(row=0,column=0,sticky='nsew',padx=(0,10))
        self.tabs=ttk.Notebook(left,style='WB.TNotebook');self.tabs.pack(fill='both',expand=True,padx=8,pady=6)
        groups=[('几何',list(INPUTS.values())[0]),(' Fluid supply ',list(INPUTS.values())[1]+list(INPUTS.values())[2]),
            (' Electric field ',list(INPUTS.values())[3]),(' Cell ',list(INPUTS.values())[4]),(' Value ',list(INPUTS.values())[5])]
        combos={('supply','sample_control'):['pressure','flow'],('supply','orientation'):
            ['horizontal','down','up','horizontal_down','down_horizontal','custom','legacy'],
            ('numerics','particle_backend'):['auto','numba','numpy']}
        for name,specs in groups:
            pane=ScrollFrame(self.tabs);self.tabs.add(pane,text=name);body=pane.body
            body.columnconfigure(0,weight=1);body.columnconfigure(1,weight=1)
            for i,(section,key,title) in enumerate(specs):
                cell=tk.Frame(body,bg=WHITE);cell.grid(row=i//2,column=i%2,sticky='ew',padx=10,pady=9)
                label(cell,title,size=9,color=MUTED).pack(anchor='w',pady=(0,4))
                value=getattr(getattr(self.session.draft,section),key)
                v=tk.StringVar(value='' if value is None else str(value));self.vars[(section,key)]=v
                choices=combos.get((section,key))
                widget=(ttk.Combobox(cell,textvariable=v,values=choices,state='readonly',style='WB.TCombobox',width=15)
                    if choices else ttk.Entry(cell,textvariable=v,style='WB.TEntry',width=16))
                widget.pack(fill='x');self.inputs[(section,key)]=widget
                for target in (cell,widget):target.bind('<MouseWheel>',pane.wheel,add='+')
                v.trace_add('write',self.changed)
            if name==' Cell ':
                controls=tk.Frame(body,bg=PALE);controls.grid(row=(len(specs)+1)//2,column=0,columnspan=2,sticky='ew',padx=10,pady=12)
                self.brownian_override=self.session.draft.cell.brownian_enabled
                self.brownian_enabled=tk.BooleanVar(value=self.session.draft.cell.use_brownian)
                self.brownian_switch=ttk.Checkbutton(controls,text='启用布朗运动 (长管与解析末端)',
                    variable=self.brownian_enabled,command=self.toggle_brownian)
                self.brownian_switch.pack(anchor='w',padx=10,pady=(10,5))
                self.brownian_note=tk.StringVar()
                label(controls,textvariable=self.brownian_note,size=9,color=MUTED,wraplength=390,justify='left').pack(fill='x',padx=10,pady=4)
                button(controls,'恢复按粒径阈值',self.reset_brownian).pack(anchor='w',padx=10,pady=(3,10))
            if name==' Fluid supply ':self.pressure_tools(body,len(specs)//2+1)
            if name==' Value ':
                self.spray_enabled=tk.BooleanVar(value=self.session.draft.spray.enabled)
                ttk.Checkbutton(body,text='计算喷口两相流 (实验性)',variable=self.spray_enabled,
                    command=self.changed).grid(row=len(specs)//2+1,column=0,columnspan=2,sticky='w',padx=10,pady=9)
                self.free_charge_enabled=tk.BooleanVar(value=self.session.draft.spray.free_charge_enabled)
                ttk.Checkbutton(body,text='自由电荷输运 / RC供电 / Maxwell Feedback ',variable=self.free_charge_enabled,
                    command=self.changed).grid(row=len(specs)//2+2,column=0,columnspan=2,sticky='w',padx=10,pady=6)
                label(body,'更多物性, 分段摆放与数值设置可在完整 JSON 中编辑.',size=9,color=MUTED,
                    wraplength=400).grid(row=len(specs)//2+3,column=0,columnspan=2,padx=10,pady=8,sticky='w')
        actions=tk.Frame(left,bg=WHITE);actions.pack(fill='x',padx=12,pady=12)
        button(actions,'载入参数',self.load_config).pack(side='left')
        button(actions,'保存参数',self.save_config).pack(side='left',padx=3)
        button(actions,' Complete  JSON',self.advanced).pack(side='right')
        right=Card(self,'参数即时预览','这里响应参数草稿; 流场图片始终标明实际计算条件.')
        right.grid(row=0,column=1,sticky='nsew')
        self.preview=tk.Canvas(right,bg=WHITE,highlightthickness=0,height=170);self.preview.pack(fill='x',padx=10)
        self.preview.bind('<Configure>',lambda e:self.refresh())
        self.preview_text=tk.StringVar();label(right,size=11,textvariable=self.preview_text,wraplength=370,justify='left').pack(fill='x',padx=18,pady=12)
        self.diff_text=tk.StringVar();label(right,size=9,color=MUTED,textvariable=self.diff_text,wraplength=370,
            justify='left').pack(fill='x',padx=18,pady=10)
        button(right,'应用参数 · 计算所需模块',app.calculate,primary=True).pack(fill='x',padx=18,pady=12)
        self.error=tk.StringVar();label(right,size=9,color=AMBER,textvariable=self.error,wraplength=370,justify='left').pack(fill='x',padx=18)
        self.refresh()

    def pressure_tools(self,body,row):
        tools=tk.Frame(body,bg=PALE);tools.grid(row=row,column=0,columnspan=2,sticky='ew',padx=10,pady=12)
        label(tools,'压力换算与恒压供液',bold=True).pack(anchor='w',padx=10,pady=8)
        line=tk.Frame(tools,bg=PALE);line.pack(fill='x',padx=8,pady=4)
        self.pressure=tk.StringVar(value='400');self.unit=tk.StringVar(value='mbar')
        ttk.Entry(line,textvariable=self.pressure,width=7,style='WB.TEntry').pack(side='left')
        ttk.Combobox(line,textvariable=self.unit,values=['mbar','kPa','bar','psi','Pa'],width=6,
            style='WB.TCombobox',state='readonly').pack(side='left',padx=4)
        button(line,'换算并写入',self.convert).pack(side='left')
        self.pressure_note=tk.StringVar();label(tools,textvariable=self.pressure_note,size=9,color=MUTED,
            wraplength=400).pack(anchor='w',padx=10,pady=6)
        line=tk.Frame(tools,bg=PALE);line.pack(fill='x',padx=8,pady=4)
        self.target=tk.StringVar(value='1')
        ttk.Entry(line,textvariable=self.target,width=7,style='WB.TEntry').pack(side='left')
        label(line,'μL/min',size=9).pack(side='left',padx=4)
        button(line,'目标流量反推压力',self.infer).pack(side='left')
        label(tools,'反推为充分发展流近似; 最终流量还需闭合末端背压.\n恒压模式中的样品流量由求解确定.',
            size=9,color=MUTED,wraplength=400,justify='left').pack(anchor='w',padx=10,pady=8)

    def collect(self):
        c=deepcopy(self.session.draft)
        for (section,key),v in self.vars.items():
            old=getattr(getattr(c,section),key);value=v.get().strip()
            if old is None:value=None if not value else float(value)
            elif isinstance(old,int):value=int(value)
            elif isinstance(old,float):value=float(value)
            setattr(getattr(c,section),key,value)
        c.spray.enabled=self.spray_enabled.get()
        c.spray.free_charge_enabled=self.free_charge_enabled.get()
        c.cell.brownian_enabled=self.brownian_override
        return c.validate()

    def toggle_brownian(self):
        self.brownian_override=bool(self.brownian_enabled.get());self.changed()

    def reset_brownian(self):
        self.brownian_override=None
        c=self.session.draft.cell;self.brownian_enabled.set(c.diameter_um<=c.brownian_below_um)
        self.changed()

    def changed(self,*args):
        if not self.muted:self.app.schedule_draft()

    def sync(self):
        self.muted=True
        for (section,key),v in self.vars.items():
            value=getattr(getattr(self.session.draft,section),key);v.set('' if value is None else str(value))
        self.spray_enabled.set(self.session.draft.spray.enabled)
        self.free_charge_enabled.set(self.session.draft.spray.free_charge_enabled)
        self.brownian_override=self.session.draft.cell.brownian_enabled
        self.brownian_enabled.set(self.session.draft.cell.use_brownian);self.muted=False;self.refresh()

    def refresh(self):
        c=self.session.draft;g=c.geometry;f=c.fluid;info=self.session.draft_preview()
        if hasattr(self,'brownian_note'):
            enabled=c.cell.diameter_um<=c.cell.brownian_below_um if self.brownian_override is None else self.brownian_override
            self.brownian_enabled.set(enabled)
            mode=f' Size threshold :  Diameter  ≤ {c.cell.brownian_below_um:g} μm 时开启.' if self.brownian_override is None else '手动开启: 不受粒径阈值限制.' if enabled else '手动关闭: 所有粒径均不加入布朗运动.'
            self.brownian_note.set(mode+'\n Apply with “应用 / 计算”; 开启后随机运输计算可能更慢.')
        sample=self.inputs.get(('fluid','sample_flow_ul_min'))
        if sample:sample.configure(state='readonly' if c.supply.sample_control=='pressure' else 'normal')
        approximate='充分发展流预估 (待末端背压校核)' if info['approximate'] else '恒流输入'
        volume=np.pi*g.a**2*g.sample_length_mm*1e-3*1e12
        transit=volume/(max(info['flow_ul_min'],1e-15)*1000/60)
        self.preview_text.set(f' Sample flow  {info["flow_ul_min"]:.5g} μL/min\n{approximate}\n\n入口事件强度 {info["rate_cells_s"]:.4g}  Count /s\n体积 / 流量时间 {transit:.4g} s\n\n Sample capillary  {g.sample_length_mm/10:g} cm ·  Cell  {c.cell.diameter_um:g} μm\n Methanol sheath  {c.chemistry.sheath_methanol_fraction:.0%}')
        if self.session.base:
            from .compute import physical_inputs
            from ..gui.source_state import LABELS
            a=physical_inputs(self.session.base.config);b=physical_inputs(c)
            changes=[f'{LABELS.get((section,key),key)}: {old} → {b[section].get(key)}' for section,values in a.items()
                for key,old in values.items() if old!=b[section].get(key)]
            self.diff_text.set('待应用变化: \n'+'\n'.join(changes[:6]) if changes else '当前物理参数已有结果.浓度与采集设置可即时比较.')
        else:self.diff_text.set('尚未载入物理结果.设置完成后计算一次, 即可在各页面复用.')
        if hasattr(self,'pressure_note'):
            p=c.supply.sample_pressure_mbar
            self.pressure_note.set(f'{p:g} mbar = {p/10:g} kPa = {p/1000:g} bar')
        canvas=self.preview;canvas.delete('all');w=max(270,canvas.winfo_width());cy=83;x0=20;end=w-30
        for radial,color in [(45,'#d7e6ef'),(31,'#ede4fa'),(15,'#deeffc')]:
            canvas.create_rectangle(x0,cy-radial,end,cy+radial,fill=color,outline=WHITE,width=3)
        end_inner=end-45;canvas.create_rectangle(x0,cy-15,end_inner,cy+15,fill='#cbd9e4',outline='')
        canvas.create_rectangle(x0,cy-8,end_inner,cy+8,fill='#e4f2ff',outline='')
        radius=min(7,7*c.cell.diameter_um/8);canvas.create_oval(end_inner-65-radius,cy-radius,end_inner-65+radius,cy+radius,fill=BLUE,outline=WHITE)
        canvas.create_line(end_inner,cy-38,end_inner,cy+44,fill=MUTED,dash=(3,3))
        canvas.create_text(end_inner,cy+62,text=f' Sample recess  {g.inner_recess_mm:g} mm',font=(FONT,9),fill=MUTED)
        canvas.create_text(end,20,text=' Sheath outlet  z=0',font=(FONT,9),fill=TEAL,anchor='e')

    def convert(self):
        try:self.vars[('supply','sample_pressure_mbar')].set(str(convert_pressure(float(self.pressure.get()),self.unit.get(),'mbar')))
        except ValueError as e:self.error.set(str(e))
    def infer(self):
        try:self.vars[('supply','sample_pressure_mbar')].set(f'{infer_pressure(self.collect(),float(self.target.get())):.8g}')
        except ValueError as e:self.error.set(str(e))
    def load_config(self):
        path=filedialog.askopenfilename(parent=self.app.root,filetypes=[('JSON','*.json')])
        if path:
            try:self.session.edit(Config.load(path));self.sync();self.app.refresh()
            except (ValueError,OSError,KeyError) as e:self.error.set(str(e))
    def save_config(self):
        path=filedialog.asksaveasfilename(parent=self.app.root,defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:self.collect().save(path)
    def advanced(self):
        window=tk.Toplevel(self.app.root);window.title(' Experiment parameters  JSON');window.geometry('780x690')
        text=tk.Text(window,font=('Consolas',10),wrap='none');text.pack(fill='both',expand=True,padx=12,pady=12)
        text.insert('1.0',json.dumps(self.session.draft.to_dict(),ensure_ascii=False,indent=2))
        def apply():
            try:self.session.edit(Config.from_dict(json.loads(text.get('1.0','end'))));self.sync();self.app.refresh();window.destroy()
            except (ValueError,TypeError,KeyError) as e:messagebox.showerror('参数无效',str(e),parent=window)
        button(window,'应用到参数草稿',apply,primary=True).pack(pady=10)
