import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Real2Sim with Astra · 本地刚体工作台")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=7865)
    serve.add_argument("--data", default="runs")
    sub.add_parser("doctor")
    isaac=sub.add_parser("isaac",help="Export existing reviewed project to native Isaac USD and validate")
    isaac.add_argument("project",type=Path)
    isaac.add_argument("--seconds",type=float,default=10)
    iview=sub.add_parser("view-isaac",help="Open an exported Isaac USD")
    iview.add_argument("scene",type=Path)
    recheck=sub.add_parser("recheck",help="Reuse existing assets, adjust table, and revalidate")
    recheck.add_argument("project",type=Path)
    recheck.add_argument("--auto-layout",action="store_true",help="Infer support graph and iteratively repair rigid poses with Astra")
    recheck.add_argument("--seconds",type=float,default=10)
    for name in ('width','depth','x','y'):recheck.add_argument('--table-'+name,type=float)
    view = sub.add_parser("view")
    view.add_argument("scene", type=Path)
    build = sub.add_parser("build",help="Single RGB file or calibrated RGB-D capture folder to MuJoCo / Isaac Sim")
    build.add_argument("input",type=Path,nargs="?")
    build.add_argument("--resume",type=Path,help="Reuse selected masks after failed 3D reconstruction; preserves prior logs")
    build.add_argument("--data",default="runs")
    build.add_argument("--target",choices=["mujoco","isaac"],default="mujoco")
    build.add_argument("--selection",type=Path)
    review=build.add_mutually_exclusive_group()
    review.add_argument("--refine-assets",dest="refine_assets",action="store_true",help="Review all selected assets (default)")
    review.add_argument("--skip-asset-review",dest="refine_assets",action="store_false",help="Explicit ablation only; export remains unreviewed")
    build.set_defaults(refine_assets=True)
    build.add_argument("--seconds",type=float,default=10)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
    elif args.command == "build":
        if args.target=="isaac":
            from .isaac import runtime_python
            runtime_python()
        if not .1 <= args.seconds <= 30:
            parser.error("--seconds must be between 0.1 and 30")
        if bool(args.input)==bool(args.resume):
            parser.error('Provide exactly one input path or --resume PROJECT')
        if args.resume and args.selection:
            parser.error('--resume reuses saved selection; do not pass --selection')
        from .pipeline import build,resume_reconstruction
        if args.resume:
            result=resume_reconstruction(args.resume,args.refine_assets,args.seconds)
        else:
            result=build(args.input,args.data,json.loads(args.selection.read_text()) if args.selection else None,args.refine_assets,args.seconds)
        if args.target=='isaac':
            import uuid
            from .isaac import export_isaac
            from .store import Store
            store=Store(args.resume.resolve().parent if args.resume else args.data)
            result=export_isaac(store,result,store.path(result.id)/'jobs'/('isaac_'+uuid.uuid4().hex[:8]),args.seconds,print)
        if args.target=='isaac' and not result.reports['isaac']['validation']['dynamics_check_pass']:
            raise SystemExit('Isaac exported with failed native dynamics validation; inspect validation.json')
        if not result.reports['validation']['physics_check_pass']:
            raise SystemExit("Exported with failed physical validation; inspect validation.json")
    elif args.command == "recheck":
        import uuid
        from .store import Store
        from .pipeline import recheck_existing
        from .models import TableRecord
        if not .1<=args.seconds<=30:parser.error('--seconds must be between 0.1 and 30')
        project=args.project.resolve();store=Store(project.parent);scene=store.load(project.name)
        table=scene.table.model_dump()
        for key in ('width','depth'):
            value=getattr(args,'table_'+key)
            if value is not None:table[key]=value;table['calibration_source']='user'
        for i,key in enumerate(('x','y')):
            value=getattr(args,'table_'+key)
            if value is not None:table['center_xy'][i]=value;table['calibration_source']='user'
        scene.table=TableRecord.model_validate(table)
        result=recheck_existing(store,scene,project/'jobs'/('recheck_'+uuid.uuid4().hex[:8]),args.seconds,auto_layout=args.auto_layout)
        if not result.reports['validation']['physics_check_pass']:raise SystemExit('Exported with failed physical validation; inspect validation.json')
    elif args.command == "isaac":
        import uuid
        from .store import Store
        from .pipeline import generate_target
        if not .1<=args.seconds<=30:parser.error('--seconds must be 0.1–30')
        project=args.project.resolve();store=Store(project.parent);scene=store.load(project.name)
        scene=generate_target(store,scene,project/'jobs'/('isaac_'+uuid.uuid4().hex[:8]),target='isaac',seconds=args.seconds,progress=print)
        print(json.dumps(scene.reports['isaac'],ensure_ascii=False))
        if not scene.reports['isaac']['validation']['dynamics_check_pass']:raise SystemExit('Isaac native dynamics validation failed; inspect validation.json')
    elif args.command == "view-isaac":
        import subprocess
        from .isaac import runtime_python,worker_path
        raise SystemExit(subprocess.call([str(runtime_python()),str(worker_path()),'--view',str(args.scene.resolve())]))
    elif args.command == "doctor":
        from .adapters import status
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "view":
        import mujoco.viewer
        # Use the native viewer's simulation loop and playback controls.
        mujoco.viewer.launch_from_path(str(args.scene.resolve()))


if __name__ == "__main__":
    main()
