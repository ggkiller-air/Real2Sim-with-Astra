# Real2Sim with Astra

上传单张图片后自动分割，少量增删物体并选择属性，一键生成和检查 MuJoCo 场景。项目提供本地图形界面、共享 Python/API 实现，以及给 Astra 的操作 skill。

**v0.1 只处理 MuJoCo 普通刚体。** 特殊结构仅保存标注；后续再加入关节、软体和 Isaac Lab。真实 SAM 3D 推理已在本机 RTX 4090 通过；端到端验证记录见 [docs/validation.json](docs/validation.json)。

## 当前能力

| 模块 | 状态 |
| --- | --- |
| 本地界面 | 紧凑单页；自动分割、物体卡片、删除/撤回、正负点/框选预览确认、属性自动保存、三维预览 |
| SAM3 | 独立模型进程；自动类别扫描、掩码去重、机器人排除；文字和框/点 API 保留 |
| SAM 3D Objects | ModelScope 主权重；独立 CUDA 环境；真实彩色网格、相机位姿与共同预测深度 |
| 几何输入 | OBJ / GLB / STL / PLY 导入；显式占位盒预演 |
| 布局 | 界面自动简化背景为支撑平面；SAM3 桌面掩码 + MoGe 平面拟合；共同相机/尺度，自动支撑面 |
| 资产优化 | Astra 参考图驱动的锤子/按钮盒参数化候选；同视角前后对照，保留原资产，尚不自动替换场景 |
| 布局优化 | 真实凸体接触消穿插、重力落稳；Astra 多视角提出位置建议或稳定姿态搜索目标，经渲染与物理复检决定采纳 |
| MuJoCo | 彩色显示网格与 CoACD 碰撞体分离、质量和摩擦、原生多视角、物理报告、含资产 ZIP |
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

打开 **http://127.0.0.1:7865**。端口可通过 `bash scripts/run.sh --port 7866` 修改。应用只监听本机；SAM3/SAM 3D 在本机处理；Astra 视觉检查通过已登录的 Codex 服务调用，发送参考图、生成视角与场景摘要。

模型环境与权重获取见 [docs/models.md](docs/models.md)。浏览器端 Three.js 已随仓库提供，无需 Node 或 CDN。

## 操作流程

1. 上传单张图片，自动扫描常见桌面物体、合并重复掩码，并排除检测到的机械臂/夹爪。自动扫描可能漏检，不保证类别无关的全物体覆盖。
2. 图片下方列出保留的物体卡片。删除多余项，可撤回本次会话中的删除；漏掉的用“点击选物体”加正点，右键或 Shift+点击加负点排除背景；预览后确认。也保留框选。
3. 在卡片上选择类型与重量，自动保存。点击名称可选填尺寸或导入已有网格，不需要调整坐标、桌面四角等复杂参数。
4. 点击“生成占位场景”（SAM 3D 就绪时显示“生成场景”），自动生成彩色几何、共同深度布局、接触优化、落桌和 3 秒 MuJoCo 检查，随后采集视角并用 Astra 检查与修正。任一步失败停止流程；占位盒不代表真实三维资产。
5. 查看“检查结果”和“多视角与 Astra”，再下载 MuJoCo ZIP。解压后加载 `scene.xml`；原生查看器：`.venv/bin/real2sim view /absolute/path/to/scene.xml`。

背景只生成简化桌面和地面，不重建机械臂或远处设备。单图预测深度和假设桌宽不等于尺度标定，默认尺寸与物理参数仍是估计。多视角与深度输入暂未接入。

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
- `geometry.py`, `reconstruction.py`：颜色/碰撞处理、共同深度与支撑坐标。
- `asset_refinement.py`, `prompts/astra_asset_refinement.txt`：独立资产候选、参数约束和前后对照。
- `review.py`, `pose_search.py`, `prompts/astra_scene_review.txt`：Astra 调用、约束建议和测量验收。
- `physics.py`：MuJoCo 导出及物理检查。
- `server.py`, `cli.py`：共享 API、本地服务及查看器。
- `src/real2sim_workbench/web/`：可随 Python 包分发的界面与 Three.js。
- `scripts/`, `environments/`, `tests/`, `docs/`：运行工具、依赖记录、验证和设计资料。

`runs/`、`.env.local`、`vendor/` 和权重不进入 Git。第三方组件见 [THIRD_PARTY.md](THIRD_PARTY.md)。自有代码许可尚未选择。
