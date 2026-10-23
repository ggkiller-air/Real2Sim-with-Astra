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
    view = sub.add_parser("view")
    view.add_argument("scene", type=Path)
    build = sub.add_parser("build",help="Single RGB file or calibrated RGB-D capture folder to MuJoCo")
    build.add_argument("input",type=Path)
    build.add_argument("--data",default="runs")
    build.add_argument("--selection",type=Path)
    build.add_argument("--refine-assets",action="store_true")
    build.add_argument("--seconds",type=float,default=10)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
    elif args.command == "build":
        if not .1 <= args.seconds <= 30:
            parser.error("--seconds must be between 0.1 and 30")
        from .pipeline import build
        result=build(args.input,args.data,json.loads(args.selection.read_text()) if args.selection else None,args.refine_assets,args.seconds)
        if not result.reports['validation']['physics_check_pass']:
            raise SystemExit("Exported with failed physical validation; inspect validation.json")
    elif args.command == "doctor":
        from .adapters import status
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "view":
        import mujoco.viewer
        # Use the native viewer's simulation loop and playback controls.
        mujoco.viewer.launch_from_path(str(args.scene.resolve()))


if __name__ == "__main__":
    main()
