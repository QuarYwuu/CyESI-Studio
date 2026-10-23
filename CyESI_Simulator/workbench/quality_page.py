"""Numerical diagnostics and model-scope summaries."""
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk
from .widgets import Card,label,button,table,WHITE,BG,INK,MUTED,TEAL,BLUE,AMBER,FONT


STATUS={'pass':' Pass ','warning':'需核查','failed':' Fail ','missing':'缺少记录','assumption':' Model parameters '}
COLORS={'pass':TEAL,'warning':AMBER,'failed':'#be4d59','missing':MUTED,'assumption':BLUE}


class QualityPage(tk.Frame):
    def __init__(self,parent,app):
        super().__init__(parent,bg=BG);self.app=app;self.key=None;self.assessment=None;self.rows={}
        head=Card(self,'模型核查与适用范围','把数值守恒,  Resolution , 运输覆盖与未闭合边界放在一起; 通过筛查不等于实验验证.')
        head.pack(fill='x',pady=(0,10))
        row=tk.Frame(head,bg=WHITE);row.pack(fill='x',padx=16,pady=8)
        self.summary=tk.StringVar(value='载入数值结果后生成核查.');label(row,textvariable=self.summary,size=10,bold=True,wraplength=780,justify='left').pack(side='left',fill='x',expand=True)
        button(row,'导出核查记录',self.export,primary=True).pack(side='right')
        self.funnel=tk.StringVar();label(head,textvariable=self.funnel,size=9,color=MUTED,wraplength=950,justify='left').pack(fill='x',padx=16,pady=(0,12))
        card=Card(self);card.pack(fill='both',expand=True)
        row=tk.Frame(card,bg=WHITE);row.pack(fill='x',padx=12,pady=8)
        label(row,'核查状态',size=9,color=MUTED).pack(side='left',padx=(0,8))
        self.filter=tk.StringVar(value='全部');choice=ttk.Combobox(row,textvariable=self.filter,
            values=['全部','需核查 /  Fail ','缺少记录 /  Model parameters ',' Pass '],state='readonly',style='WB.TCombobox',width=22)
        choice.pack(side='left');choice.bind('<<ComboboxSelected>>',lambda e:self.fill_table())
        label(row,'选择一项查看依据和下一步.',size=9,color=MUTED).pack(side='right')
        box,self.table=table(card,['stage','label','status','value'],['过程 / 区域','核查内容',' Status ',' Value '],[130,340,110,170],height=8)
        box.pack(fill='both',expand=True,padx=12,pady=8)
        for status,color in COLORS.items():self.table.tag_configure(status,foreground=color)
        self.table.bind('<<TreeviewSelect>>',self.select)
        # Reserve the explanation before allocating the expanding table, even
        # on smaller screens. Long audit lists keep their own scrollbar.
        detail_box=tk.Frame(card,bg=WHITE);detail_box.pack(side='bottom',fill='x',padx=12,pady=(0,12),before=box)
        self.detail=tk.Text(detail_box,height=5,wrap='word',font=(FONT,10),bg='#f6f9fc',fg=INK,relief='flat',padx=12,pady=10,state='disabled')
        self.detail.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(detail_box,orient='vertical',command=self.detail.yview);scrollbar.pack(side='right',fill='y')
        self.detail.configure(yscrollcommand=scrollbar.set)

    def refresh(self):
        s=self.app.session;data=s.data
        if data is None:return
        key=(id(data.fields),id(data.summary),data.config.cell.concentration_per_ml)
        if key!=self.key:
            from ..diagnostics.model_assessment import assess_model
            self.assessment=assess_model(data);self.key=key;self.fill_table()
        a=self.assessment;counts=a['counts']
        self.summary.set(' · '.join(f'{STATUS[status]} {counts.get(status,0)}' for status in STATUS))
        from ..cell_state.analysis import outlet_rates
        rates=outlet_rates(data.config,data.summary)
        injection=rates['injection_cells_s'];nozzle=rates['nozzle_cells_s']
        nozzle_text=' Unknown ' if nozzle is None else f'{nozzle:.4g}  Count /s'
        pure=s.metrics()['pure'] if getattr(s,'views_ready',True) else None
        pure_text='统计更新中' if pure is None else f'{pure:.4g}  Count /s'
        upper=rates.get('unresolved_nozzle_upper_cells_s')
        interval='' if upper is None or nozzle is None or abs(upper-nozzle)<1e-12 else f'; 把未解析响应全部视为后续可出管时, 上限 {upper:.4g}  Count /s'
        self.funnel.set(f' Input rate  {injection:.4g}  Count /s →  Predicted outlet rate  {nozzle_text} → 理论去重纯来源机会 {pure_text}.\n'
            '出管比例可能受截断影响'+interval+'; 采集机会包含离子包输送假设.真实入口捕获, 完整率和 MS 检出率需实验校准.')

    def fill_table(self):
        if self.assessment is None:return
        self.table.delete(*self.table.get_children());self.rows={};choice=self.filter.get()
        allowed={'全部':set(STATUS),'需核查 /  Fail ':{'warning','failed'},
            '缺少记录 /  Model parameters ':{'missing','assumption'},' Pass ':{'pass'}}[choice]
        priority={'failed':0,'warning':1,'missing':2,'assumption':3,'pass':4}
        for i,row in sorted(enumerate(self.assessment['checks']),key=lambda item:priority[item[1]['status']]):
            if row['status'] not in allowed:continue
            key=str(i);self.rows[key]=row;value=row.get('value');unit=row.get('unit','')
            rendered='—' if value is None else f'{value:.5g}' if isinstance(value,(int,float)) else '['+', '.join(f'{x:.5g}' for x in value)+']' if isinstance(value,list) and all(isinstance(x,(int,float)) for x in value) else str(value)
            self.table.insert('','end',iid=key,values=[row.get('stage',''),row['label'],STATUS[row['status']],rendered+(' '+unit if value is not None else '')],tags=(row['status'],))
        ids=self.table.get_children()
        if ids:self.table.selection_set(ids[0]);self.select()
        else:self.set_detail('当前筛选没有记录.')

    def set_detail(self,text):
        self.detail.configure(state='normal');self.detail.delete('1.0','end');self.detail.insert('end',text);self.detail.configure(state='disabled')

    def select(self,e=None):
        selected=self.table.selection()
        if not selected:return
        row=self.rows[selected[0]]
        detail=f'{row["label"]} · {STATUS[row["status"]]}\n\n{row.get("detail","")}'
        if row.get('value') is not None:detail+='\n\n Value : '+json.dumps(row['value'],ensure_ascii=False)+' '+row.get('unit','')
        if row.get('recorded_value') is not None:detail+='\n记录值: '+str(row['recorded_value'])
        if row.get('screening_tolerance') is not None:detail+='\n软件筛查阈值: '+str(row['screening_tolerance'])
        if row.get('recommendation'):detail+='\n\n建议: '+str(row['recommendation'])
        if row.get('source'):detail+='\n依据: '+str(row['source'])
        self.set_detail(detail)

    def export(self):
        if self.assessment is None:self.app.message('请先载入数值结果.');return
        record=self.assessment;directory=self.app.new_directory('audits')
        def work(notify):
            from ..report.export import write_csv
            directory.mkdir(parents=True,exist_ok=True)
            (directory/'model_assessment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            write_csv(directory/'model_checks.csv',record['checks']);return directory
        self.app.run_task('导出模型核查记录',work,lambda path:self.app.message('核查已导出: '+str(path)))
