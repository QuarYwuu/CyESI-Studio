"""Standalone workbench reports. Source result folders remain read-only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import html
import json
import csv
from ..report.export import clean,write_csv
from .pages import draw_fields,draw_resolution,metric_string


class StaticPlot:
    def __init__(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        self.figure=Figure(figsize=(11,4.5),dpi=120,facecolor='white',layout='constrained')
        self.canvas=FigureCanvasAgg(self.figure)
    def axes(self,n=1):
        self.figure.clear();return self.figure.subplots(1,n,squeeze=False)[0]
    def show(self):self.canvas.draw()
    def save(self,path):self.figure.savefig(path,dpi=160)


def export_report(session,directory,progress=lambda s:None):
    from ..visualization.figures import serialized_plotting
    @serialized_plotting
    def generate():
        root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
        if session.data is None:raise ValueError('请先载入计算结果')
        data=session.data;pin=session.current_pin;current=session.metrics();baseline=session.metrics(pin.data) if pin else None
        preparation=session.prepare_views();model=preparation['model']
        from ..diagnostics.model_assessment import assess_model
        assessment=assess_model(data)
        state=dict(source=data.source,solved_config=(session.base or data).config.to_dict(),view_config=data.config.to_dict(),draft_config=session.draft.to_dict(),
            draft_pending=session.pending,acquisition=session.acquisition.to_dict(),resolution=session.resolution,
            actual_it_ms=session.actual_it_ms,metrics=current,baseline_metrics=baseline,
            process_parameters=dict(duration_s=session.duration_s,startup=session.startup),
            process_summary=model.summary() if model is not None else dict(status='unavailable',reason=preparation['process_error']),
            model_assessment=assessment,
            source_warnings=data.summary.get('warnings',[]),coupling=data.summary.get('coupling'),
            baseline=None if pin is None else dict(name=pin.name,source=pin.data.source,config=pin.data.config.to_dict()),
            reuse=session.cache_summary(),actual_ms_detection_rate=None,biologically_intact_entry_fraction=None)
        if session.has_interactions():
            from .interaction_storage import save_study
            save_study(session,root/'interactions')
            analysis=session.interaction_acquisition()
            state['interaction_study']=dict(parameters=session.interaction_result.parameters.to_dict(),
                diagnostics=session.interaction_result.diagnostics,conditional_acquisition=analysis)
            rows=analysis['ledger']
            with (root/'interaction_acquisition.csv').open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ['event_id','packet_onset_s','baseline_status','status','captured_fraction','pure_captured_fraction'])
                writer.writeheader();writer.writerows(rows)
        (root/'workbench_summary.json').write_text(json.dumps(clean(state),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(root/'acquisition.csv',session.acquisition_result()['rows'])
        (root/'model_assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        write_csv(root/'model_checks.csv',assessment['checks'])
        ledger=[] if model is None else model.ledger
        write_csv(root/'cell_event_ledger.csv',ledger)
        r=None if model is None else model.records
        feed=[] if r is None else [dict(event_id=int(i),injection_s=a,trajectory_sample_id=int(data.ids[k]),nozzle_s=b)
            for i,a,k,b in zip(r['event_id'],r['injection_s'],r['template_id'],r['nozzle_s'])]
        write_csv(root/'continuous_feed.csv',feed)
        # Write a CSV header for empty cohorts.
        for name,rows,columns in [('cell_event_ledger.csv',ledger,['event_id','packet_onset_s','status','scan_count','pure_scan_count','captured_fraction','pure_captured_fraction']),
                                  ('continuous_feed.csv',feed,['event_id','injection_s','trajectory_sample_id','nozzle_s'])]:
            if not rows:
                with (root/name).open('w',encoding='utf-8-sig',newline='') as handle:csv.writer(handle).writerow(columns)
        rows=[dict(metric=key,baseline=None if baseline is None else baseline[key],current=value,
            difference=None if baseline is None or baseline[key] is None or value is None else value-baseline[key]) for key,value in current.items()]
        write_csv(root/'comparison.csv',rows)
        progress('导出已解场与共同色标对比')
        plot=StaticPlot();draw_fields(plot,data,' Methanol ',pin.data if pin else None);plot.save(root/'solvent_comparison.png')
        plot=StaticPlot();draw_fields(plot,data,'流速',pin.data if pin else None);plot.save(root/'velocity_comparison.png')
        progress('导出四档采集分辨率对比')
        plot=StaticPlot();draw_resolution(plot,session,pin.data if pin else None);plot.save(root/'acquisition_comparison.png')
        c=data.config;p=session.acquisition
        timestamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+08:00')
        cards=''.join(f'<div class="card"><span>{title}</span><strong>{metric_string(current[key])}</strong><small>{unit}</small></div>'
            for key,title,unit in [('injection',' Input rate ',' Count /s'),('nozzle',' Predicted outlet rate ',' Count /s'),('transit','全程平均运输','s'),('pure','理论去重纯来源机会',' Count /s')])
        table=''.join(f'<tr><td>{html.escape(row["metric"])}</td><td>{metric_string(row["baseline"])}</td><td>{metric_string(row["current"])}</td><td>{metric_string(row["difference"])}</td></tr>' for row in rows)
        notice='<p class="notice">草稿仍有未应用物理参数; 下方结果对应已计算配置.</p>' if session.pending else ''
        quality=''.join('<li>'+html.escape(str(w))+'</li>' for w in data.summary.get('warnings',[]))
        from .quality_page import STATUS
        checks=''.join('<tr><td>'+html.escape(row['label'])+'</td><td>'+STATUS[row['status']]+'</td><td>'+html.escape(str(row.get('detail','')))+'</td></tr>' for row in assessment['checks'])
        process_notice='' if model is not None else '<p class="notice">连续动画 / 事件账本未生成: '+html.escape(str(preparation['process_error']))+'; 总体理论指标不受动画容量限制.CSV 仅保留列名.</p>'
        text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyESI Workbench</title><style>body{{font:15px/1.7 system-ui,sans-serif;color:#18324d;background:#eef2f7;max-width:1150px;margin:36px auto;padding:0 24px}}h1{{font-size:32px;margin-bottom:0}}.sub,small{{color:#708195}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:20px;margin:16px 0}}strong{{display:block;font-size:30px;color:#0b9185}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #e5eaf1;text-align:left}}a{{color:#0b9185}}.notice{{background:#fff0d9;padding:12px;border-radius:8px}}@media(max-width:650px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style>
<h1>CyESI Workbench</h1><p class="sub">实验 → 运输与聚焦 → 喷雾与入口 →  Acquisition  →  Comparison  · {timestamp}</p>{notice}
<div class="cards">{cards}</div><section><h2>计算来源与采集方法</h2><p>{html.escape(data.source)}</p>
<p> Sample capillary  {c.geometry.sample_length_mm/10:g} cm;  Actual  Q={c.fluid.sample_flow_ul_min:.6g} μL/min;  Concentration  {c.cell.concentration_per_ml:g}/mL;  Cell  {c.cell.diameter_um:g} μm.</p>
<p>Q Exactive Full MS; R={session.resolution:,};  Actual  IT={session.actual_it_ms:g} ms; Maximum IT={p.max_injection_time_ms:g} ms; Microscans={p.microscans}; 峰宽={p.signal_width_ms:g} ms ( Model settings );  Delivery fraction ={p.signal_delivery_fraction:g} ( Model settings ).</p>
<p>不同页面共用源数据; 浓度和采集时序更新复用稀相物理响应. Measured  MS 检出率与生物完整率未知.外液暴露不直接等于提取效率.</p></section>
<section><h2>共同单位与色标的对比</h2><p> Reference : {html.escape(pin.name if pin else '尚未设置')}.采集比较统一采用当前方法.</p><table><tr><th> Metrics </th><th> Reference </th><th> Current </th><th>变化</th></tr>{table}</table>
<img src="velocity_comparison.png" alt="速度场对比"><img src="solvent_comparison.png" alt="甲醇场对比"></section>
<section><h2> Resolution settings </h2><img src="acquisition_comparison.png" alt="理论采集机会"></section>
<section><h2>模型核查</h2><p>数值守恒与适用范围的自动筛查, 不能代替网格独立性与实验校准.</p><table><tr><th>内容</th><th> Status </th><th>依据与限制</th></tr>{checks}</table><a href="model_assessment.json">核查 JSON</a> · <a href="model_checks.csv">核查 CSV</a></section>
<section><h2> Source data </h2>{process_notice}<a href="workbench_summary.json"> Parameters  /  Source  / 缓存记录</a> · <a href="comparison.csv">指标对比 CSV</a> · <a href="acquisition.csv">采集情景 CSV</a> · <a href="continuous_feed.csv"> Continuous source </a> · <a href="cell_event_ledger.csv"> Per-cell acquisition ledger </a>
<p>该报告不重新求解 CFD.两相出口稳定性, 膜参数与入口捕获仍需校准和网格验证; 单个条件飞行不构成总体入孔效率.</p></section></html>'''
        if quality:text=text.replace('</html>','<section><details><summary>源结果的求解与边界说明</summary><ul>'+quality+'</ul></details></section></html>')
        if session.has_interactions():
            text=text.replace('</html>','<section><h2>共用物理模块: 二维细胞扰动与黏附</h2><p>实际连续编号与共同物理时钟; 局部接触, 结合, 脱附及实际出管重新匹配采集门.上方全程理论率仍为独立运输基线, 局部观察不直接等于整体回收率.</p><a href="interactions/interactions.json"> Parameters  / 守恒 / 局部采集诊断</a> · <a href="interactions/interactions.npz">轨迹与状态数组</a> · <a href="interaction_acquisition.csv">实际出管与基线采集对照</a></section></html>')
        (root/'workbench_report.html').write_text(text,encoding='utf-8')
        return root
    return generate()
