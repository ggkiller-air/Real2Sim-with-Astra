# Real2Sim with Astra

从图片出发，选择物体、纠正分割、编辑刚体布局，并导出可检查的 MuJoCo 场景。项目提供本地图形界面、共享 Python/API 实现，以及给 Astra 的操作 skill。

**v0.1 只处理 MuJoCo 普通刚体。** 特殊结构仅保存标注；后续再加入关节、软体和 Isaac Lab。尚未完成单图到真实三维资产的端到端验证。

## 当前能力

| 模块 | 状态 |
| --- | --- |
| 本地界面 | 图片上传、项目保存、掩码叠加、候选勾选、框和正负点、三维预览 |
| SAM3 | 独立模型进程；真实文字候选和实例修正已在本机运行 |
| SAM 3D Objects | 官方源码获取脚本和适配器；缺权重权限，真实推理未验证 |
| 几何输入 | OBJ / GLB / STL / PLY 导入；显式占位盒预演 |
| 布局 | 桌面四点映射、框底部接触近似、支撑层级、位置/尺寸/朝向编辑 |
| 布局优化 | 保守 AABB 的有界 XY 搜索；非 REST3D 复现，不优化相机或深度 |
| MuJoCo | 独立自由刚体、质量和摩擦、短时仿真报告、含资产 ZIP 导出 |
| 特殊结构 | 铰链/抽屉/布料/软体标签与备注，当前均按普通刚体处理 |

导入网格使用单个凸包碰撞，不能保留容器内腔。默认尺寸和物理参数是先验，不是从图片测得。检查通过只表示当前近似下的短时稳定，不能代替视觉对照、物理标定或任务验证。

## 启动

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。SAM3 另需可用的 NVIDIA/CUDA 模型环境。核心界面、网格导入和 MuJoCo 检查不依赖 SAM3。

```bash
git clone https://github.com/ggkiller-air/Real2Sim-with-Astra.git
cd Real2Sim-with-Astra
bash scripts/setup.sh
cp .env.example .env.local
# 编辑 .env.local，填写本机 SAM3 Python 和权重路径；未安装时可暂留空。
bash scripts/run.sh
```

打开 **http://127.0.0.1:7865**。端口可通过 `bash scripts/run.sh --port 7866` 修改。应用只监听本机；图片和推理在本地处理。

模型环境与权重获取见 [docs/models.md](docs/models.md)。浏览器端 Three.js 已随仓库提供，无需 Node 或 CDN。

## 操作流程

1. 上传图片，输入英文物体描述获得候选；或框选物体并添加正负点。
2. 勾选需要的实例。点击清单后可修正该物体掩码；“新物体提示”创建新实例。
3. 导入已有网格；具备 SAM 3D 环境后可调用适配器。没有三维资产时，“生成占位预演”只用于验证流水线。
4. 按远左、远右、近右、近左标桌面四角，填写桌面尺寸。编辑物体尺寸、位姿与支撑物并保存。
5. 运行布局优化，对照原图；运行物理检查查看穿插、位移、旋转、速度和预期支撑接触。
6. 下载 MuJoCo ZIP，解压后加载 `scene.xml`。原生查看器：`.venv/bin/real2sim view /absolute/path/to/scene.xml`。

## 验证与 agent 入口

```bash
.venv/bin/python -m pytest -q
.venv/bin/real2sim doctor
.venv/bin/python scripts/rigid_example.py --output runs/rigid-example
```

`doctor` 只检查配置，不宣称模型已经成功加载。示例是合成刚体场景，不是图片重建评测。真实运行证据保存在 `runs/<id>/jobs/`，失败记录也保留。

Astra 工作流见 [skills/real2sim/SKILL.md](skills/real2sim/SKILL.md)，共享接口见 [docs/api.md](docs/api.md)。

## 代码结构

- `models.py`, `store.py`：场景记录、来源与持久化。
- `adapters.py`, `workers/`：独立 SAM3 / SAM 3D 进程。
- `geometry.py`：网格导入、平面映射和布局搜索。
- `physics.py`：MuJoCo 导出及物理检查。
- `server.py`, `cli.py`：共享 API、本地服务及查看器。
- `src/real2sim_workbench/web/`：可随 Python 包分发的界面与 Three.js。
- `scripts/`, `environments/`, `tests/`, `docs/`：运行工具、依赖记录、验证和设计资料。

`runs/`、`.env.local`、`vendor/` 和权重不进入 Git。第三方组件见 [THIRD_PARTY.md](THIRD_PARTY.md)。自有代码许可尚未选择。
