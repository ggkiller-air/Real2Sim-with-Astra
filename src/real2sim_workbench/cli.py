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
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.data), host="127.0.0.1", port=args.port)
    elif args.command == "doctor":
        from .adapters import status
        print(json.dumps(status(), ensure_ascii=False, indent=2))
    elif args.command == "view":
        import time
        import mujoco
        import mujoco.viewer
        model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep-(time.monotonic()-start)))


if __name__ == "__main__":
    main()
