import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="CyESI Multiphysics Simulator")
    parser.add_argument("--gui", action="store_true", help="Open the native Python GUI (default if no batch flags)")
    parser.add_argument('--workbench',action='store_true',help='Open the independent structured workflow and comparison workbench')
    parser.add_argument('--studio',action='store_true',help='Open the unified 2-D / 3-D scientific workspace (independent of legacy apps)')
    parser.add_argument('--studio-port',type=int,default=0,help='Local Studio HTTP port; 0 selects an available port')
    parser.add_argument('--studio-no-browser',action='store_true',help='Start Studio without automatically opening a browser')
    parser.add_argument("--explore", action="store_true", help="Open the whole-device and nozzle visual workspaces")
    parser.add_argument("--results", type=Path, help="Source CFD results folder for visualization or cell-state analysis")
    parser.add_argument('--scene3d',action='store_true',help='Open the independent continuous 3-D numerical workspace')
    parser.add_argument('--scene3d-port',type=int,default=0,help='Local 3-D HTTP port; 0 selects an available port')
    parser.add_argument('--scene3d-no-browser',action='store_true',help='Start the local 3-D workspace without opening a browser')
    parser.add_argument('--scene3d-saved-only',action='store_true',help='Initially display read-only saved results without automatically solving new 3-D fields')
    parser.add_argument("--scene", choices=["device","nozzle"], default="device", help="Initial visual scene")
    parser.add_argument("--cell-state",action="store_true",help="Compute conditional lipid-shell / Q Exactive entry states from exported CFD")
    parser.add_argument('--acquisition',action='store_true',help='Estimate Q Exactive Full MS timing, occupancy and unique single-cell opportunities')
    parser.add_argument('--process-view',action='store_true',help='Open linked long-tube / focusing / MS collection movie')
    parser.add_argument('--process-report',action='store_true',help='Export linked process figures, event ledger and GIF from saved CFD')
    parser.add_argument('--process-duration',type=float,default=60.,help='Process-view observation window in seconds')
    parser.add_argument('--process-resolution',type=int,default=17500,help='Selected Q Exactive resolution for process report')
    parser.add_argument('--process-it',type=float,default=30.,help='Actual injection time for process report (ms)')
    parser.add_argument('--process-width-ms',type=float,help='Assumed rectangular ion packet width; default 20 ms example, not calibrated')
    parser.add_argument('--length-study',action='store_true',help='Compute tube length / orientation / shear and Q Exactive rate scenarios')
    parser.add_argument('--study-config',type=Path,help='Length / orientation study JSON')
    parser.add_argument('--acquisition-config',type=Path,help='Instrument timing / signal-packet scenario JSON for --acquisition mode')
    parser.add_argument('--acquisition-timing-csv',type=Path,help='Actual accumulation_start_s / actual_it_ms gates for --acquisition')
    parser.add_argument("--state-config",type=Path,help="Explicit lipid-shell / inlet scenario JSON")
    parser.add_argument("--run", action="store_true", help="Run a headless simulation and export all figures")
    parser.add_argument("--config", type=Path, help="JSON config, merged onto documented defaults")
    parser.add_argument("--write-default", type=Path, help="Write a complete editable default config and exit")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--cells", type=int, help="Transport response sample count; continuous arrival count is set by flow/concentration")
    parser.add_argument('--workers',type=int,help='CPU workers: 0 automatic CPU/memory budget, 1 serial, 2-32 explicit')
    parser.add_argument('--particle-backend',choices=('auto','numpy','numba'),help='Compiled particle integration or NumPy reference')
    parser.add_argument("--scan", action="store_true", help="Run all 19 requested one-at-a-time sweep cases")
    parser.add_argument("--full-factorial", action="store_true", help="Run all 500 parameter combinations")
    parser.add_argument("--scan-cells", type=int, default=512, help="Cells per scan point (default 512; baseline stays 10000)")
    parser.add_argument("--no-spray", action="store_true", help="Skip experimental two-phase transient")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF encoding")
    args = parser.parse_args()
    if args.write_default:
        Config().save(args.write_default); print(args.write_default.resolve()); return
    if args.studio:
        from .studio.server import launch
        launch(args.results,args.studio_port,not args.studio_no_browser);return
    if args.scene3d:
        from .scene3d.server import launch
        launch(args.results,args.scene3d_port,not args.scene3d_no_browser,not args.scene3d_saved_only);return
    if args.workbench:
        from .workbench.app import launch
        launch(args.results,args.config);return
    if args.process_view:
        from .gui.process import launch
        from .acquisition.q_exactive import AcquisitionParameters
        from .visualization.process_data import ProcessParameters
        from .cell_state.parameters import StateParameters
        ap=AcquisitionParameters.load(args.acquisition_config) if args.acquisition_config else AcquisitionParameters(signal_width_ms=20.)
        if args.process_width_ms is not None:ap.signal_width_ms=args.process_width_ms
        launch(args.results,ap,ProcessParameters(duration_s=args.process_duration,resolution=args.process_resolution,actual_it_ms=args.process_it),
            StateParameters.load(args.state_config) if args.state_config else None);return
    if args.process_report:
        from .visualization.explorer_data import ExplorerData,find_results
        from .visualization.process_data import ProcessModel,ProcessParameters,selected_flight
        from .visualization.process_report import export
        from .acquisition.q_exactive import AcquisitionParameters
        from .cell_state.parameters import StateParameters
        source=args.results or find_results()
        if source is None:parser.error('Run CFD first, or provide --results')
        ap=AcquisitionParameters.load(args.acquisition_config) if args.acquisition_config else AcquisitionParameters(signal_width_ms=20.)
        if args.process_width_ms is not None:ap.signal_width_ms=args.process_width_ms
        model=ProcessModel(ExplorerData.load(source),ap,ProcessParameters(duration_s=args.process_duration,
            resolution=args.process_resolution,actual_it_ms=args.process_it))
        if model.default_event is not None:
            print('Computing one conditional lipid-carrier inlet case',flush=True)
            try:model.flight=selected_flight(model,model.default_event,StateParameters.load(args.state_config) if args.state_config else None)
            except ValueError as error:print('Conditional inlet case unavailable: '+str(error),flush=True)
        output=args.output or Path(source)/'process_views'/datetime.now().strftime('%Y%m%d_%H%M%S')
        path=export(model,output,not args.no_animation,progress=lambda s:print(s,flush=True))
        print(f'Linked process report: {path/"process_report.html"}');return
    if args.length_study:
        from .optimization.length_study import analyze,export,StudyParameters
        from .acquisition.q_exactive import AcquisitionParameters
        config=Config.load(args.config) if args.config else Config()
        if args.workers is not None:config.numerics.workers=args.workers
        if args.particle_backend is not None:config.numerics.particle_backend=args.particle_backend
        if args.config is None:
            config.supply.sample_control='pressure';config.supply.orientation='horizontal'
        parameters=StudyParameters.load(args.study_config) if args.study_config else StudyParameters()
        ap=AcquisitionParameters.load(args.acquisition_config) if args.acquisition_config else AcquisitionParameters()
        output=args.output or Path('outputs')/'length_studies'/datetime.now().strftime('%Y%m%d_%H%M%S')
        output.mkdir(parents=True,exist_ok=True)
        def checkpoint(state):
            import json
            from .report.export import clean,write_csv
            (output/'progress.json').write_text(json.dumps(clean(state),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            write_csv(output/'length_study_partial.csv',state['rows'])
        result=analyze(config,parameters,ap,progress=lambda s:print(s,flush=True),on_case=checkpoint)
        path=export(result,output)
        print(f'Length / shear report: {path/"length_study_report.html"}')
        if any(r['status']!='ok' for r in result['rows']):raise SystemExit(2)
        return
    if args.acquisition:
        from .visualization.explorer_data import ExplorerData,find_results
        from .cell_state.analysis import outlet_rates
        from .acquisition.q_exactive import AcquisitionParameters,analyze
        from .acquisition.report import export
        source=args.results or find_results()
        if source is None:parser.error('Run CFD first, or provide --results')
        data=ExplorerData.load(source)
        parameters=AcquisitionParameters.load(args.acquisition_config) if args.acquisition_config else AcquisitionParameters()
        if args.acquisition_timing_csv:
            from .acquisition.measured_timing import load_gates,analyze_gates,export as export_gates
            if parameters.microscans!=1:parser.error('Actual gate CSV mode requires Microscans=1')
            result=analyze_gates(load_gates(args.acquisition_timing_csv),outlet_rates(data.config,data.summary)['nozzle_cells_s'],
                parameters.signal_width_ms,parameters.signal_delivery_fraction,parameters.single_event_identification_efficiency,source=args.acquisition_timing_csv)
            path=export_gates(result,args.output or Path(source)/'measured_acquisition')
            print(f'Actual accumulation gate report: {path/"measured_timing_report.html"}');return
        result=analyze(outlet_rates(data.config,data.summary)['nozzle_cells_s'],parameters,data.source,data.config.to_dict())
        path=export(result,args.output or Path(source)/'acquisition')
        print(f"Acquisition report: {path/'acquisition_report.html'}")
        for row in result['rows']:
            print(f"R={row['resolution_at_mz200']:6d}, IT={row['actual_injection_ms']:g} ms: unique pure cells={row['unique_pure_single_cell_events_s']:.3f}/s; not calibrated detection")
        return
    if args.cell_state:
        from .visualization.explorer_data import ExplorerData,find_results
        from .cell_state.parameters import StateParameters
        from .cell_state.analysis import analyze
        from .cell_state.report import export as export_state
        source=args.results or find_results()
        if source is None:parser.error("Run CFD first, or provide --results")
        parameters=StateParameters.load(args.state_config) if args.state_config else StateParameters()
        result=analyze(ExplorerData.load(source),parameters,progress=lambda x:print(x,flush=True))
        path=export_state(result,args.output or Path(source)/"cell_state")
        print(f"Cell-state report: {path/'cell_state_report.html'}")
        print(f"Expected nozzle rate: {result.outlet_rates['nozzle_cells_s']:.3f} cells/s; MS detection rate: unknown")
        return
    if args.explore:
        from .gui.explorer import launch
        launch(args.results,args.scene);return
    if args.gui or not (args.run or args.scan or args.full_factorial):
        from .gui.interface import launch
        launch(args.config,args.results,args.workers,args.particle_backend); return
    config = Config.load(args.config) if args.config else Config()
    if args.workers is not None:config.numerics.workers=args.workers
    if args.particle_backend is not None:config.numerics.particle_backend=args.particle_backend
    if args.cells is not None: config.cell.count = args.cells
    if args.no_spray: config.spray.enabled = False
    config.validate()
    if args.scan_cells < 1: parser.error("--scan-cells must be positive")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("outputs")/stamp
    from .engine import run
    from .report.export import export
    result = run(config, progress=lambda x: print(x, flush=True))
    if args.scan or args.full_factorial:
        from .optimization.parameter_scan import scan
        result.scan = scan(config, args.full_factorial, args.scan_cells, progress=lambda x: print(x, flush=True))
    root = export(result, output, not args.no_animation, progress=print)
    print(f"Report: {root/'report.html'}")
    print(f"Transport samples: {config.cell.count}; exited samples: {result.particles.diagnostics['exited']}; continuous source: {result.events['rate_cells_s']:.3f} cells/s")
    failed = sum(row["status"] != "ok" for row in (result.scan or []))
    if failed:
        print(f"WARNING: {failed} scan cases failed; inspect scan.csv")
        raise SystemExit(2)


if __name__ == "__main__": main()
