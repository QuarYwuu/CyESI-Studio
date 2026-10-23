"""Separate native workflow UI; only the current page redraws during playback."""
from copy import deepcopy,copy
from datetime import datetime,timezone,timedelta
from pathlib import Path
import json
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk,filedialog
from .widgets import *
from .session import WorkbenchSession,catalog
from .compute import physical_key,digest,save_numeric,reusable_output


PAGES=[('overview',' Overview ','总览'),('setup',' Experiment ','设置'),('transport','运输与聚焦','运输'),
       ('interactions',' Interactions and adhesion ',' Interactions '),
       ('spray','喷雾与入口',' Inlet '),('acquisition','采集与回收',' Acquisition '),('compare','对比与研究',' Comparison '),
       ('quality','模型核查','核查'),('library',' Results ','资料库')]


class TaskCancelled(Exception):
    """Cooperative stop at a solver stage boundary; completed caches survive."""


class WorkbenchApp:
    def __init__(self,root,session=None,output_root='outputs/workbench'):
        self.root=root;self.session=session or WorkbenchSession();self.pages={};self.current='overview';self.closed=False
        self.busy=False;self.job_name=None;self.messages=queue.Queue();self.logs=[];self.tick_job=None;self.poll_job=None
        self.draft_job=None;self.method_job=None;self.last_tick=time.perf_counter();self.children={}
        self.restored_animation=None;self.completed={};self.exports={};self.worker=None;self.last_export=None
        self.cancel_event=threading.Event();self.prep_running=False;self.prep_request=None;self.prep_key=None;self.ready_key=None
        self.failed_view_key=None
        self.session.views_ready=False;self.session.process_error=None
        self.output_root=Path(output_root)
        root.title('CyESI Workbench · 多物理场研究工作台');root.configure(bg=BG)
        width=min(1400,root.winfo_screenwidth()-60);height=min(950,root.winfo_screenheight()-80)
        root.geometry(f'{width}x{height}');root.minsize(1080,740);styles(root)
        self.nav=tk.Frame(root,bg=NAV,width=205);self.nav.pack(side='left',fill='y');self.nav.pack_propagate(False)
        logo=tk.Canvas(self.nav,bg=NAV,highlightthickness=0,height=62);logo.pack(fill='x',padx=18,pady=(20,0))
        for radius,color in [(23,'#45647f'),(16,'#8c68d3'),(8,'#5bd6c7')]:logo.create_oval(26-radius,31-radius,26+radius,31+radius,outline=color,width=3)
        logo.create_line(25,31,48,31,fill='#5bd6c7',width=2)
        label(self.nav,'CyESI',size=22,bold=True,color=WHITE).pack(anchor='w',padx=24)
        label(self.nav,'MULTIPHYSICS WORKBENCH',size=8,color='#8ea7bd').pack(anchor='w',padx=24,pady=(2,25))
        self.nav_buttons={}
        for i,(key,title,short) in enumerate(PAGES):
            if key in ('setup','quality'):label(self.nav,'实验流程' if key=='setup' else '数据与成果',size=8,color='#7694ad').pack(anchor='w',padx=24,pady=(16,8))
            b=tk.Button(self.nav,text=f'{i:02d}   {title}',font=(FONT,11),fg='#c6d6e5',bg=NAV,activebackground=NAV_ACTIVE,
                activeforeground=WHITE,anchor='w',relief='flat',bd=0,padx=20,pady=13,command=lambda k=key:self.navigate(k),cursor='hand2')
            b.pack(fill='x',padx=10,pady=2);self.nav_buttons[key]=b
        bottom=tk.Frame(self.nav,bg=NAV);bottom.pack(side='bottom',fill='x',padx=20,pady=20)
        label(bottom,'当前计算来源',size=8,color='#7694ad').pack(anchor='w')
        self.source=tk.StringVar(value='尚未载入');label(bottom,textvariable=self.source,size=9,color='#c6d6e5',wraplength=165,justify='left').pack(anchor='w',pady=(4,8))
        label(bottom,'旧版内容独立保留',size=8,color='#7694ad').pack(anchor='w')
        main=tk.Frame(root,bg=BG);main.pack(side='left',fill='both',expand=True)
        header=tk.Frame(main,bg=WHITE);header.pack(fill='x')
        row=tk.Frame(header,bg=WHITE);row.pack(fill='x',padx=22,pady=(15,8))
        title=tk.Frame(row,bg=WHITE);title.pack(side='left')
        label(title,'CyESI  /  研究工作台',size=9,color=MUTED).pack(anchor='w')
        self.title=tk.StringVar();label(title,textvariable=self.title,size=19,bold=True).pack(anchor='w',pady=(3,0))
        tools=tk.Frame(row,bg=WHITE);tools.pack(side='right',anchor='center')
        button(tools,'载入结果',self.choose_result).pack(side='left',padx=3)
        button(tools,'保存基准',self.pin).pack(side='left',padx=3)
        self.run_button=button(tools,'应用 / 计算',self.calculate,primary=True);self.run_button.pack(side='left',padx=3)
        self.steps=tk.Canvas(header,bg=WHITE,highlightthickness=0,height=31);self.steps.pack(fill='x',padx=22,pady=(0,10))
        self.steps.bind('<Configure>',lambda e:self.draw_steps());self.steps.bind('<Button-1>',self.step_clicked)
        self.metrics=tk.Frame(main,bg=BG);self.metrics.pack(fill='x',padx=20,pady=(15,10))
        self.cards=[]
        for i,(title,unit,color) in enumerate([('入口事件强度',' Count /s',BLUE),('模型出管率',' Count /s',TEAL),('全程平均运输','s',INK),('理论纯来源机会',' Count /s · 去重',PURPLE)]):
            self.metrics.columnconfigure(i,weight=1,uniform='metrics')
            card=Metric(self.metrics,title,unit,color);card.grid(row=0,column=i,sticky='ew',padx=(0 if i==0 else 5,0 if i==3 else 5));self.cards.append(card)
        self.banner=tk.Frame(main,bg='#e1efec');self.banner.pack(fill='x',padx=20,pady=(0,10))
        self.banner_text=tk.StringVar();self.banner_label=label(self.banner,textvariable=self.banner_text,size=9,color='#257367',wraplength=1000,justify='left')
        self.banner_label.pack(fill='x',padx=12,pady=8);self.banner.bind('<Configure>',lambda e:self.banner_label.configure(wraplength=max(500,e.width-24)))
        self.content=tk.Frame(main,bg=BG);self.content.pack(fill='both',expand=True,padx=20,pady=(0,12))
        footer=tk.Frame(main,bg=WHITE);footer.pack(side='bottom',fill='x',before=self.content)
        self.status=tk.StringVar(value=' Ready ; 可载入已有结果.');self.status_label=label(footer,textvariable=self.status,size=9,color=MUTED,wraplength=850)
        self.status_label.pack(side='left',fill='x',expand=True,padx=18,pady=10)
        self.progress=ttk.Progressbar(footer,mode='indeterminate',length=120)
        self.cancel_button=button(footer,'停止任务',self.cancel_task,state='disabled');self.cancel_button.pack(side='right',padx=6)
        button(footer,' Simulation records ',self.show_log).pack(side='right',padx=6)
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.navigate('overview');self.tick_job=root.after(80,self.tick);self.poll_job=root.after(100,self.poll)

    def new_directory(self,kind):
        stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        return self.output_root/kind/stamp

    def navigate(self,key):
        if self.closed:return
        if self.current in self.pages:self.pages[self.current].pack_forget()
        self.current=key
        if key not in self.pages:
            from .pages import OverviewPage,TransportPage,AcquisitionPage
            from .setup_page import SetupPage
            from .research_pages import SprayPage,ComparePage,LibraryPage
            from .quality_page import QualityPage
            from .interaction_page import InteractionPage
            cls=dict(overview=OverviewPage,setup=SetupPage,transport=TransportPage,spray=SprayPage,
                acquisition=AcquisitionPage,compare=ComparePage,quality=QualityPage,library=LibraryPage,interactions=InteractionPage)[key]
            self.pages[key]=cls(self.content,self)
        self.pages[key].pack(fill='both',expand=True)
        self.title.set(next(title for k,title,_ in PAGES if k==key))
        for k,b in self.nav_buttons.items():b.configure(bg=NAV_ACTIVE if key==k else NAV,fg=WHITE if key==k else '#c6d6e5')
        self.refresh();self.draw_steps()

    def draw_steps(self):
        self.steps.delete('all');w=max(600,self.steps.winfo_width());self.step_hits=[]
        sequence=[row for row in PAGES if row[0] in ('setup','transport','interactions','spray','acquisition','compare')]
        total=len(sequence)
        for i,(key,title,short) in enumerate(sequence):
            x=10+i*(w-25)/total;color=TEAL if self.current==key else '#9aabbc'
            if i<total-1:self.steps.create_line(x+30,15,x+(w-25)/total-10,15,fill=LINE)
            self.steps.create_oval(x,5,x+20,25,fill=color,outline='')
            self.steps.create_text(x+10,15,text=str(i+1),font=(FONT,8),fill=WHITE)
            self.steps.create_text(x+29,15,text=short,font=(FONT,9),fill=color,anchor='w');self.step_hits.append((x,x+(w-25)/total,key))
    def step_clicked(self,event):
        for left,right,key in self.step_hits:
            if left<=event.x<right:self.navigate(key);return

    def refresh(self):
        if self.closed:return
        s=self.session
        try:
            ready=self.request_views()
            metrics=s.metrics() if ready else dict.fromkeys(['injection','nozzle','transit','pure'])
            for card,key in zip(self.cards,['injection','nozzle','transit','pure']):card.set(metrics[key])
            self.cards[-1].note.configure(text=f' Count /s · R={s.resolution:,} · IT={s.actual_it_ms:g} ms')
            if s.acquisition.single_event_identification_efficiency is not None and metrics['pure'] is not None:
                self.cards[-1].note.configure(text=f'R={s.resolution:,} · IT={s.actual_it_ms:g} ms\n情景识别 {metrics["pure"]*s.acquisition.single_event_identification_efficiency:.4g}  Count /s')
            if s.data:
                self.source.set(Path(s.data.source).name)
                if s.pending:
                    info=s.draft_preview();self.banner.configure(bg='#fff0d9');self.banner_label.configure(bg='#fff0d9',fg=AMBER)
                    self.banner_text.set(f'参数草稿待应用.当前图和指标仍来自已解条件; 草稿入口强度预估 {info["rate_cells_s"]:.4g}  Count /s,  Flow rate  {info["flow_ul_min"]:.5g} μL/min.')
                else:
                    self.banner.configure(bg='#e1efec');self.banner_label.configure(bg='#e1efec',fg='#257367')
                    self.banner_text.set(f'已解结果: {Path(s.data.source).name}  ·  各页面共用此结果  ·   Concentration  / 采集更新无需重算 CFD')
                    if not s.data.summary.get('coupling',{}).get('converged',True):
                        self.banner.configure(bg='#fff0d9');self.banner_label.configure(bg='#fff0d9',fg=AMBER)
                        self.banner_text.set(self.banner_text.get()+' · 耦合尚未达到容差, 需收敛校核')
                if not ready:
                    self.banner_text.set(self.banner_text.get()+(' · 统计视图准备失败, 请查看计算记录或修改参数后重试' if self.failed_view_key==s.view_key() else ' · 统计视图在后台更新; 动画暂候'))
                elif s.process_error:self.banner_text.set(self.banner_text.get()+' · 连续动画: '+s.process_error)
            else:self.banner_text.set('先载入已有结果; 或设置实验条件后计算一次, 再按流程观察.')
            if self.current in self.pages and (ready or self.current in ('setup','spray','quality','library') or s.data is None):
                self.pages[self.current].refresh()
        except (ValueError,KeyError,TypeError) as error:self.message(str(error))

    def frozen_session(self):
        frozen=copy(self.session);frozen.draft=deepcopy(self.session.draft)
        frozen.acquisition=deepcopy(self.session.acquisition);frozen.pins=list(self.session.pins)
        return frozen

    def request_views(self):
        """Coalesce edits; prepare only immutable snapshots outside Tk."""
        s=self.session
        if s.data is None:s.views_ready=True;return True
        key=s.view_key()
        if key==self.failed_view_key:s.views_ready=False;return False
        if key==self.ready_key and (s.process_error or s.peek_process() is not None):
            self.prep_request=None;s.views_ready=True;return True
        s.views_ready=False
        if key==self.prep_key:self.prep_request=None
        elif self.prep_request is None or key!=self.prep_request[0]:
            self.prep_request=(key,self.frozen_session())
        if not self.prep_running:self.start_view_preparation()
        return False

    def start_view_preparation(self):
        if self.closed or self.prep_request is None:return
        key,frozen=self.prep_request;self.prep_request=None;self.prep_key=key;self.prep_running=True
        messages=self.messages
        def work():
            try:messages.put(('views',(key,frozen.prepare_views(),None)))
            except Exception:messages.put(('views',(key,None,traceback.format_exc())))
        threading.Thread(target=work,daemon=True,name='CyESI-views').start()

    def finish_view_preparation(self,payload):
        key,result,trace=payload;self.prep_running=False;self.prep_key=None
        if self.session.data is not None and key==self.session.view_key():
            if trace:
                self.failed_view_key=key;self.session.views_ready=False;self.session.process_error='统计视图准备失败'
                self.message(trace)
            else:
                self.ready_key=key;self.failed_view_key=None;self.session.accept_views(result,key)
                self.session.views_ready=True;self.session.process_error=result['process_error']
            self.refresh()
        self.start_view_preparation()

    def message(self,text):
        if self.closed:return
        self.status.set(text);self.logs.append(text);self.logs=self.logs[-300:]

    def schedule_draft(self):
        if self.draft_job is not None:self.root.after_cancel(self.draft_job)
        self.draft_job=self.root.after(300,self.apply_draft)
    def apply_draft(self):
        if self.draft_job is not None:self.root.after_cancel(self.draft_job)
        self.draft_job=None
        if 'setup' not in self.pages:return True
        try:
            self.session.edit(self.pages['setup'].collect());self.pages['setup'].error.set('');self.refresh();return True
        except (ValueError,TypeError,KeyError) as e:self.pages['setup'].error.set('输入待修正: '+str(e));return False
    def schedule_method(self):
        if self.method_job is not None:self.root.after_cancel(self.method_job)
        self.method_job=self.root.after(300,self.apply_method)
    def apply_method(self):
        if self.method_job is not None:self.root.after_cancel(self.method_job)
        self.method_job=None
        if 'acquisition' not in self.pages:return True
        try:
            p,res,it=self.pages['acquisition'].collect();self.session.set_method(p,res,it);self.refresh();return True
        except (ValueError,TypeError,KeyError) as e:self.message('采集输入待修正: '+str(e));return False

    def choose_result(self):
        path=filedialog.askdirectory(parent=self.root,title='选择含 config.json / fields.npz 的结果目录')
        if path:self.load_result(path)
    def load_result(self,path,preserve_draft=False):
        if self.busy:self.message('当前任务完成后可切换计算来源.');return
        s=self.session;draft_at_start=s.draft.to_dict()
        def work(notify):
            notify('读取数值场与轨迹');data=s.read(path)
            notify('恢复可复用的载液与电场');s.solver.seed_saved(data)
            return data
        def complete(data):
            # Edits made during loading remain an explicit pending draft.
            s.adopt(data,reset_draft=not preserve_draft and s.draft.to_dict()==draft_at_start);self.restored_animation=None
            self.completed.clear()
            if 'setup' in self.pages:self.pages['setup'].sync()
            self.refresh();self.message('已载入数值结果; 查看各页面不会重算物理场.')
        self.run_task('后台载入结果 · 可继续浏览',work,complete)
    def pin(self):
        try:
            if not self.apply_draft() or not self.apply_method():return
            self.session.pin();self.refresh();self.message('已保留当前已解配置为对比基准.')
        except ValueError as e:self.message(str(e))

    def run_task(self,name,compute,complete):
        if self.busy:self.message('当前任务进行中; 相同请求不重复启动.');return False
        self.busy=True;self.job_name=name;self.cancel_event=threading.Event()
        self.run_button.configure(state='disabled');self.cancel_button.configure(state='normal')
        self.progress.pack(side='right',padx=6,before=self.cancel_button);self.progress.start(50);self.message(name)
        messages=self.messages
        stop=self.cancel_event
        def notify(text):
            if stop.is_set():raise TaskCancelled()
            messages.put(('progress',text))
        def work():
            try:
                result=compute(notify)
                if stop.is_set():raise TaskCancelled()
                messages.put(('done',(complete,result)))
            except TaskCancelled:messages.put(('cancelled',None))
            except Exception:messages.put(('error',traceback.format_exc()))
        self.worker=threading.Thread(target=work,daemon=True);self.worker.start();return True

    def cancel_task(self):
        if not self.busy:return
        self.cancel_event.set();self.cancel_button.configure(state='disabled')
        self.message('已请求停止; 当前数值阶段结束后退出, 保留已完成的缓存.')
    def calculate(self):
        if self.busy:self.message('计算正在进行; 不会重复启动.');return
        if not self.apply_draft():return
        s=self.session;point=deepcopy(s.draft).validate()
        if s.data is not None and not s.pending and reusable_output(s.data):
            self.failed_view_key=None;self.refresh();self.message('相同物理条件已有结果; 已复用, 并更新浓度 / 采集统计.');return
        key=physical_key(point)
        if key in self.completed:
            s.adopt(self.completed[key]);self.refresh();self.message('该工况已计算; 直接恢复缓存结果.');return
        directory=self.new_directory('runs')
        def work(notify):
            # Metadata lookup and archive decompression also stay off Tk.
            from ..config import Config
            notify('查找相同物理工况的已有结果')
            for row in catalog():
                notify('检查已保存工况: '+row['name'])
                try:
                    if physical_key(Config.load(Path(row['path'])/'config.json'))==key:
                        data=s.read(row['path']);s.solver.seed_saved(data)
                        if reusable_output(data):return data
                        notify('已有记录缺少当前所需模块; 保留并复用载液 /  Electric field ')
                except (ValueError,OSError,KeyError):continue
            result=s.solver.run(point,notify);notify('保存独立数值记录');data=save_numeric(result,directory)
            (directory/'reuse.json').write_text(json.dumps(s.cache_summary(),ensure_ascii=False,indent=2),encoding='utf-8');return data
        def complete(data):
            self.completed[key]=data;s.adopt(data)
            if 'setup' in self.pages:self.pages['setup'].sync()
            self.refresh();self.message('计算完成; 所有页面已切换到同一份新结果.')
        self.run_task('计算所需模块 · 可继续浏览已有结果',work,complete)

    def toggle_play(self):self.session.playing=not self.session.playing
    def tick(self):
        self.tick_job=None
        if self.closed:return
        now=time.perf_counter();elapsed=min(.25,now-self.last_tick);self.last_tick=now
        if self.session.playing and self.session.data is not None and self.session.views_ready and not self.session.process_error and self.current in ('overview','transport','acquisition'):
            self.session.now=min(self.session.duration_s,self.session.now+elapsed*self.session.speed)
            if self.session.has_interactions():
                study=self.session.interaction_result
                if self.session.now>=study.time_s[-1] and self.session.now-elapsed*self.session.speed<study.time_s[-1]:
                    self.session.now=float(study.time_s[-1]);self.session.playing=False
            if self.session.now>=self.session.duration_s:self.session.playing=False
            try:self.pages[self.current].draw()
            except (ValueError,KeyError,TypeError,IndexError) as e:self.session.playing=False;self.message('回放暂停: '+str(e))
        if self.current=='spray' and self.current in self.pages:
            page=self.pages['spray']
            if page.phase_play:
                page.phase_value=(page.phase_value+elapsed/8)%1
                page.draw()
        self.tick_job=self.root.after(80,self.tick)
    def poll(self):
        self.poll_job=None
        if self.closed:return
        try:
            while True:
                kind,payload=self.messages.get_nowait()
                if kind=='views':self.finish_view_preparation(payload);continue
                if kind=='progress':self.message(payload);continue
                self.busy=False;self.job_name=None;self.progress.stop();self.progress.pack_forget();self.run_button.configure(state='normal');self.cancel_button.configure(state='disabled')
                if kind=='done':
                    callback,result=payload
                    try:callback(result)
                    except Exception:self.message('结果已保存; 页面刷新失败: '+traceback.format_exc())
                elif kind=='cancelled':self.message('任务已停止; 当前显示结果仍可使用.')
                else:self.message('任务失败, 可修改后重试: '+payload)
        except queue.Empty:pass
        self.poll_job=self.root.after(100,self.poll)

    def restore_animation(self):
        if self.session.data is None:return
        from types import SimpleNamespace
        from ..visualization.figures import save_trajectory_animation
        data=self.session.data;f=data.fields
        result=SimpleNamespace(config=data.config,particles=SimpleNamespace(saved_ids=data.ids,
            trajectory_time=f['trajectory_time_s'],trajectory_r=f['trajectory_r_m'],trajectory_z=f['trajectory_z_m']))
        directory=self.new_directory('figures');source=id(data.fields)
        def complete(path):
            if self.session.data is not None and id(self.session.data.fields)==source:
                self.restored_animation=path;self.refresh();self.message('原动画已从现有轨迹恢复; 未重算 CFD.')
        self.run_task('恢复有限球原动画',lambda notify:save_trajectory_animation(result,directory),complete)

    def export_report(self):
        if self.session.data is None:self.message('请先载入结果.');return
        if not self.apply_draft() or not self.apply_method():return
        from .export import export_report
        s=self.session;key=(id(s.data.fields),s.data.config.cell.concentration_per_ml,digest(s.acquisition.to_dict()),
            s.resolution,s.actual_it_ms,s.duration_s,s.startup,tuple((id(p.data.fields),p.data.config.cell.concentration_per_ml) for p in s.pins),physical_key(s.draft))
        if key in self.exports and (self.exports[key]/'workbench_report.html').exists():self.last_export=self.exports[key];self.message('报告已经生成, 复用: '+str(self.last_export));return
        # Freeze the report inputs while the interactive draft remains editable.
        frozen=self.frozen_session()
        directory=self.new_directory('exports')
        def complete(path):self.exports[key]=path;self.last_export=path;self.message('工作台报告已导出: '+str(path/'workbench_report.html'))
        self.run_task('导出对比图 / CSV / 工作台报告',lambda notify:export_report(frozen,directory,notify),complete)

    def open_source_report(self):
        if self.session.data:
            import webbrowser
            source=Path(self.session.data.source)
            paths=[source/'report.html',source/'workbench_report.html']
            path=next((p for p in paths if p.exists()),None)
            if path:webbrowser.open(path.resolve().as_uri())
            else:self.message('此记录只保存数值; 可使用“导出工作台报告”.')
    def open_legacy(self):
        args=[sys.executable,'main.py','--gui']
        if self.session.data:args+=['--results',self.session.data.source]
        subprocess.Popen(args,cwd=str(Path(__file__).resolve().parents[2]))
    def advanced_tool(self,kind):
        previous=self.children.get(kind)
        if previous is not None and not previous.closed:previous.window.deiconify();previous.window.lift();return previous
        if kind=='size':
            from ..gui.size_study import SizeStudyWindow
            view=SizeStudyWindow(self.root,deepcopy(self.session.draft))
        elif kind=='length':
            from ..gui.length_study import LengthStudyWindow
            view=LengthStudyWindow(self.root,deepcopy(self.session.draft))
        elif self.session.data is None:self.message('请先载入运输结果.');return
        elif kind=='state':
            from ..gui.cell_state import CellStateWindow
            view=CellStateWindow(self.root,self.session.data)
        elif kind=='process':
            from ..gui.process import ProcessWindow
            view=ProcessWindow(self.root,self.session.data,self.session.acquisition,
                self.session.resolution,self.session.actual_it_ms,state_parameters=self.session.state_parameters)
        elif kind=='explorer':
            from ..gui.explorer import ExplorerWindow
            view=ExplorerWindow(self.root,self.session.data,page='nozzle')
        else:return
        self.children[kind]=view;return view

    def show_log(self):
        window=tk.Toplevel(self.root);window.title('计算与复用记录');window.geometry('800x500')
        text=tk.Text(window,font=(FONT,10),wrap='word',bg=WHITE,fg=INK);text.pack(fill='both',expand=True,padx=12,pady=12)
        text.insert('end','\n'.join(self.logs)+'\n\n复用计数: \n'+json.dumps(self.session.cache_summary(),ensure_ascii=False,indent=2));text.configure(state='disabled')
    def close(self):
        if self.closed:return
        self.closed=True;self.session.playing=False;self.progress.stop()
        self.cancel_event.set();self.prep_request=None
        for name in ('tick_job','poll_job','draft_job','method_job'):
            job=getattr(self,name)
            if job is not None:
                try:self.root.after_cancel(job)
                except tk.TclError:pass
                setattr(self,name,None)
        for child in list(self.children.values()):
            if not child.closed:child.close()
        self.root.destroy()


def launch(results=None,config_path=None):
    from ..config import Config
    session=WorkbenchSession(Config.load(config_path) if config_path else None)
    root=tk.Tk();app=WorkbenchApp(root,session)
    rows=catalog();source=results or (rows[0]['path'] if rows else None)
    if source:app.load_result(source,preserve_draft=bool(config_path))
    if config_path:
        session.edit(Config.load(config_path));app.refresh()
    root.mainloop()
