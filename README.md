# Real2Sim with Astra

把桌面照片重建为可在 MuJoCo 或 Isaac Sim 中加载的普通刚体场景，支持两条输入路径：

| 输入路径 | 传入内容 | 尺度与摆放来源 |
| --- | --- | --- |
| 单图 | 一张 RGB 图片 | SAM 3D 共同预测点图与支撑面；绝对尺度仍有假设 |
| 深度 / RGB-D | RGB、原始深度与标定信息组成的目录 | 实测米制深度、相机配准与支撑面 |

共同流程：**SAM3 分割 → SAM 3D 资产初稿 → Astra 审查与候选修正 → 场景组装和物理检查 → 原生视角及最终复核**。SAM3 负责分割，SAM 3D Objects 负责三维生成；Astra 通过结构化方案调用受约束的模板/清理代码，并非只有一份 prompt。

首个验收版本标签为 **`v0`**（Python 包版本 `0.1.0`），内部方法版本为 `rigid_v0.4`。发布范围与已知限制见 [v0 发布说明](docs/release-v0.md)。输出包含真实网格、纹理和凸碰撞体，但**完成运行不等于重建合格**：可能仍有形状、摆放或稳定性问题，报告会保留这些失败。铰链、抽屉、软体只保存标注；Isaac Lab 尚未接入。

## 1. 运行前准备

当前重建环境面向 **Linux + NVIDIA GPU**，已在 RTX 4090 24 GB 上实际运行；这不是最低显存保证，也没有完成第二台干净机器的安装验收。需要：

- Git、[uv](https://docs.astral.sh/uv/)；安装 SAM 3D 环境的脚本还需要 Conda 和编译工具。
- 可用的 NVIDIA 驱动。SAM3 与 SAM 3D 使用独立 Python / PyTorch / CUDA 运行时。
- SAM3 源码环境与有权限使用的权重。
- SAM 3D 源码、独立环境和权重。本项目提供获取及安装脚本。
- 已安装、登录且可调用 `gpt-6-astra` 的 Codex CLI；默认资产审查和最终复核均需要它。
- 足够磁盘空间：SAM 3D 主权重约 13.1 GB，此外还有其他权重、CUDA 环境、编译缓存及每次生成的资产。

查看 MuJoCo 交互窗口需要桌面显示环境；无显示服务器上可以先查看保存的 PNG 和报告。离屏渲染仍需要可工作的 OpenGL/EGL 驱动。

## 2. 克隆与安装

以下命令均从仓库根目录执行。

```bash
git clone https://github.com/ggkiller-air/Real2Sim-with-Astra.git
cd Real2Sim-with-Astra
bash scripts/setup.sh
```

`setup.sh` 创建 `.venv/`，安装核心、MuJoCo 和测试依赖。**它不会自动安装 SAM3、SAM 3D 或下载模型权重。**

### 2.1 配置 SAM3

在独立 Python 环境中按照 [SAM3 官方安装说明](https://github.com/facebookresearch/sam3#installation) 安装代码，并按照其 [权重获取说明](https://github.com/facebookresearch/sam3#getting-started) 申请权限、下载 checkpoint。这里使用 SAM3 图像分割接口；不要把 SAM 3D 权重或 SAM 3.1 视频权重当作本项目的 SAM3 checkpoint。

完成后记下：

- 该环境的 Python **绝对路径**；
- 下载的 SAM3 checkpoint 文件 **绝对路径**。

本项目历史验证环境为 SAM3 0.1.4 / Torch 2.7.1 CUDA 12.6；官方最新版的安装要求可能不同。SAM3 当前没有随本仓库提供一键安装脚本，详细边界见 [模型环境说明](docs/models.md)。

### 2.2 安装 SAM 3D Objects

```bash
# 获取固定版本源码。
bash scripts/fetch_models.sh

# 下载并校验 ModelScope 主权重，支持保留中断下载。
.venv/bin/python scripts/download_sam3d_modelscope.py

# 创建隔离 Python 3.11 / Torch 2.5.1 CUDA 12.1 环境并编译依赖。
bash scripts/setup_sam3d.sh
```

源码默认在 `vendor/sam-3d-objects/`，权重在 `vendor/sam-3d-objects/checkpoints/modelscope/`，运行时在 `.venv-sam3d/`。首次推理还可能获取 MoGe、DINO 等辅助模型。

安装脚本默认编译架构为 RTX 4090 的 `8.9`。其他显卡应先确认对应架构，并在执行前设置 `TORCH_CUDA_ARCH_LIST`；不要直接把本机验证当作其他 GPU 的兼容性保证。具体依赖和失败排查见 [docs/models.md](docs/models.md)。这些下载和编译步骤可能耗时较长。

### 2.3 配置 Astra 与本机路径

按 [Codex CLI 官方说明](https://developers.openai.com/codex/cli/) 安装并登录。当前代码明确调用 `gpt-6-astra`，需要账号具备相应模型访问能力；仅安装 CLI 不代表模型可用。Astra 会接收本次相关参考图、资产渲染和场景摘要，SAM3/SAM 3D 推理在本机进行。

第一次配置时复制模板：

```bash
cp .env.example .env.local
```

编辑 `.env.local`，填写本机真实绝对路径。下面是格式示例，**不要原样使用占位路径**：

```bash
R2S_SAM3_PYTHON="/absolute/path/to/sam3-env/bin/python"
R2S_SAM3_CHECKPOINT="/absolute/path/to/sam3.pt"
R2S_SAM3D_PYTHON="/absolute/path/to/Real2Sim-with-Astra/.venv-sam3d/bin/python"
R2S_SAM3D_REPO="/absolute/path/to/Real2Sim-with-Astra/vendor/sam-3d-objects"
R2S_SAM3D_CONFIG="/absolute/path/to/Real2Sim-with-Astra/vendor/sam-3d-objects/checkpoints/modelscope/pipeline.yaml"
# 如果 codex 不在 PATH 中，可选填：
# R2S_CODEX_BIN="/absolute/path/to/codex"
```

已有 `.env.local` 时直接编辑，不要用模板覆盖。无需激活核心或模型环境；`scripts/build.sh` 自动读取配置并启动各自的 Python。

### 2.4 短检查

```bash
.venv/bin/python -m pytest -q
bash scripts/build.sh --help
(
  set -a
  source .env.local
  set +a
  .venv/bin/real2sim doctor
)
```

`sam3.configured`、`sam3d.configured`、`sam3d.weights_present` 应为 `true`。`real_inference_verified` 只表示找到历史推理记录，新机器可能为 `false`；它不阻止首次尝试。`doctor` 不替代实际模型加载，也不验证 Astra 登录、网络或额度。首次安装后的完整推理仍需下面的实测。

## 3. 输入放在哪里

推荐把个人输入放在仓库内的 `data/`，它已经被 Git 忽略；也可以传入仓库外的绝对路径。仓库自带可复现示例在 `examples/rgbd_tabletop/`。

```text
Real2Sim-with-Astra/
├── data/                            # 个人输入，不进入 Git
│   ├── image/
│   │   ├── scene.jpg
│   │   └── selection.json           # 可选：按原图坐标选择对象
│   └── depth/
│       └── capture_001/             # RGB-D 采集目录
├── examples/rgbd_tabletop/
│   ├── capture/                     # 已提供的 RGB-D 示例
│   └── selection.json               # 只适用于该示例的原图
├── runs/                            # 默认输出，不进入 Git
└── .env.local                       # 本机配置，不进入 Git
```

### 路径 A：单张 RGB

输入为 PNG/JPEG 等可读取的 RGB 图片，建议先使用清晰、少遮挡的静态桌面图。放入 `data/image/scene.jpg` 后运行：

```bash
bash scripts/build.sh data/image/scene.jpg --data runs/single
```

可以先用仓库示例验证，不必准备新照片：

```bash
bash scripts/build.sh \
  examples/rgbd_tabletop/capture/image_external_rgb.jpg \
  --selection examples/rgbd_tabletop/selection.json \
  --data runs/single
```

传入的是**图片文件**，所以即使旁边存在深度文件，也不会读取深度。

### 路径 B：带标定的深度目录

当前支持仓库示例的 **D455 采集格式**，不是把任意 RGB 与 depth 图片放在一起就能使用。主 RGB 文件名目前固定为 `image_external_rgb.jpg`。

```text
capture_001/
├── image_external_rgb.jpg           # 必需：主 RGB
├── capture_metadata.json            # 必需：内外参、深度单位、裁剪等
├── d455_depth_raw.png               # 必需：uint16 原始深度，文件名由 metadata 指定
├── d455_depth_m.npy                 # 可选：米制深度复核，存在时必须与 PNG 一致
├── image_external_2_rgb.jpg          # 可选辅助 RGB；metadata 列出后必须存在
├── image_wrist_rgb.jpg              # 可选辅助 RGB；metadata 列出后必须存在
├── d455_calibration.json            # 示例保留的标定来源文件
└── calibration_bundle_20261016.json  # 示例历史标定；不作为已验证多相机外参
```

配准实际读取 `capture_metadata.json` 中的字段：

| 字段 | 内容 |
| --- | --- |
| `rgb.image_external.file` | 主 RGB 文件名，与入口固定文件保持一致 |
| `rgb.image_external.preprocessing.serial` | RGB 相机序列号，须与深度相机匹配 |
| `rgb.image_external.preprocessing.crop_roi_xyxy` | 原生 RGB 裁剪区域；未裁剪可为 `null` |
| `d455_depth.file` / `depth_scale_m` | 原始 uint16 深度文件及每单位对应的米数；零值为无效 |
| `d455_depth.intrinsics` | 深度相机宽高、`fx/fy/ppx/ppy`、畸变模型与系数 |
| `d455_depth.color_intrinsics_native` | 裁剪缩放前的 RGB 相机内参与畸变 |
| `d455_depth.color_from_depth` | 深度坐标到 RGB 坐标的 4×4 刚体变换，平移单位为米 |
| `receive_timestamp_span_s` | 本次各相机接收时间跨度记录，不能当作曝光同步证明 |

`rgb` 中其他条目作为辅助视角，程序会读取其 `file`。请使用真实采集标定，不能给新相机直接套用示例矩阵。不同镜头模型、预处理或采集格式需要适配代码；完整原始格式见 [示例 metadata](examples/rgbd_tabletop/capture/capture_metadata.json)。

先运行仓库内完整例子：

```bash
bash scripts/build.sh \
  examples/rgbd_tabletop/capture \
  --selection examples/rgbd_tabletop/selection.json \
  --data runs/depth
```

自己的数据按相同格式放好后：

```bash
bash scripts/build.sh data/depth/capture_001 --data runs/depth
```

额外相机 RGB 目前供 Astra 判断外观与遮挡结构，**没有做标定多视角三角化或点云融合**。两条路径依次运行，避免模型争用显存。

### 可选：只选择需要的物体

不传 `--selection` 时使用自动去重后的候选。自动扫描不保证找到所有类别；机器人排除也需结合结果检查。

可创建 `data/image/selection.json`，用主图中的像素点选择已有 SAM3 候选：

```json
{
  "objects": [
    {"name": "hammer", "point": [300, 319]},
    {"name": "green button box", "point": [234, 369]}
  ]
}
```

以上坐标只是示例，换照片必须修改。坐标原点在左上角，X 向右、Y 向下；RGB-D 使用**去畸变前的主 RGB**坐标。`name` 是输出名称，不会凭名称额外触发分割。点击处没有候选会报错，不会凭空生成该物体。

```bash
bash scripts/build.sh data/image/scene.jpg \
  --selection data/image/selection.json \
  --data runs/single
```

## 4. 输出在哪里、怎样看进度

`--data` 指定的是**输出项目根目录**。每次新运行创建随机项目 ID，不会直接把结果写在该根目录中，也不会自动覆盖上一次运行。

```text
PROJECT /your/checkout/runs/single/项目ID
```

记下打印的完整路径。默认不传 `--data` 时输出在 `runs/项目ID/`。`build.sh` 是批处理，**不会自动弹出网页或 MuJoCo 窗口**；模型详细日志写入文件，终端暂时安静不一定代表卡住。

在另一个终端进入仓库根目录，将下面变量替换成实际路径：

```bash
r2s_run="runs/single/替换成实际项目ID"
tail -f "$r2s_run/jobs/reconstruct/worker.log"
```

对应阶段创建文件后才能查看。按 `Ctrl+C` 退出日志查看，不会终止另一个终端的生成任务。

```text
runs/single/项目ID/
├── scene.json                       # 当前场景、来源和报告
├── input/                           # 复制的主图、辅助图与配准数据
├── jobs/
│   ├── segmentation/worker.log      # SAM3 分割日志和掩码
│   └── reconstruct/                 # SAM 3D 原始资产、纹理、凸碰撞体
│       ├── worker.log
│       ├── progress.json             # 模型阶段/物体序号
│       └── stages/asset_refinement/  # Astra 资产答案、候选、前后对照
│           ├── astra.log
│           ├── report.json
│           └── comparison.png
└── mujoco/                          # 最终导出，成功或检查失败都可能存在
    ├── scene.xml                    # MuJoCo 入口
    ├── assets/                      # 必须与 XML 一起保留
    ├── scene_record.json
    ├── validation.json              # 独立物理检查
    ├── workflow.json                # 全流程质量标记
    ├── depth_agreement.json         # 仅 RGB-D：误差及覆盖率
    ├── depth_quality.json           # 仅 RGB-D：待复核项
    └── evidence/
        ├── views/views.png          # 原图和原生多视角拼图
        └── astra/review.json         # 最终评价与未自动应用的建议
```

模型结束后还要生成纹理、碰撞体并进行 Astra 调用和物理检查。CPU 网格处理阶段 GPU 利用率可能降低。当前没有完整进度条或可靠剩余时间估计。

## 5. 查看场景与报告

仍在仓库根目录，将 `r2s_run` 设置为实际项目目录：

```bash
r2s_run="runs/single/替换成实际项目ID"
.venv/bin/real2sim view "$r2s_run/mujoco/scene.xml"
```

这时才会启动 MuJoCo 交互窗口。需要等待 XML 已生成；关闭查看器不会删除结果。无显示环境时先用图片查看器打开 `mujoco/evidence/views/views.png`，检查主视角和侧面。

完整执行结束会打印 `xml`、`physics_pass` 和 `mean_iou`。按下面顺序验收：

1. 看资产 `comparison.png`：物体是否选对，修正后是否丢失结构。
2. 看原生 `views.png`：有无漏物体、明显错误几何、穿插或摆放不对。
3. 看 `validation.json` 的 `physics_check_pass` 和每件物体的 `stable`、支撑、漂移、速度。
4. 看 `workflow.json`：是否仍有未修复资产、视觉偏差或深度偏差。
5. RGB-D 同时看 `depth_agreement.json` 的误差与覆盖率，低覆盖下的小误差不代表整体正确。

**出现 `Exported with failed physical validation` 时，场景可能已完整导出，但验收失败。** 可以打开检查；程序返回非零退出码，不要把这类结果当作合格训练资产。`physics_pass=true` 也可能伴随 `needs_review`，因为短时稳定不等于视觉或几何准确。

Astra 默认审查所有选中资产，但受当前执行器限制：锤子/按钮盒可参数化重建，其他物体可轻度保形清理或保留并标记待修。审查过不等于修好了。最终视觉建议只记录，不自动循环修改摆放。

## 6. 常用参数与故障处理

| 参数 / 现象 | 用法或含义 |
| --- | --- |
| `--seconds 10` | 独立物理检查时长，默认 10 秒，允许 0.1–30 秒 |
| `--refine-assets` | 与默认行为相同，兼容旧命令 |
| `--skip-asset-review` | 显式跳过资产和最终 Astra 调用，仅作未审查消融；仍需要 SAM3/SAM 3D |
| `CoACD … 12 convex hull limitation` | 凸碰撞近似未达到目标误差的警告；不等于程序崩溃，也不证明碰撞足够精确 |
| `socks://` 相关导入错误 | 当前模型子进程会规范化为 `socks5://`；旧环境依赖补装见 [模型说明](docs/models.md#终端-socks-代理兼容) |
| Astra 调用失败 | 查看 `jobs/reconstruct/stages/asset_refinement/astra.log` 或 `mujoco/evidence/astra/astra.log`；不会静默跳过 |

如果在**三维资产组装前**失败，且已经保留分割结果，可以续跑：

```bash
bash scripts/build.sh --resume "$r2s_run"
```

此时不再传输入文件、`--selection` 或 `--data`。旧日志保留，新推理在 `jobs/reconstruct_retry_<编号>/`。该功能不是任意阶段恢复；已经组装资产后发生的 Astra/物理失败不适用，不要重复用它覆盖后续成果。

反馈问题时保留 `PROJECT` 路径、命令、终端报错和相应日志末尾。不要为了通过检查而提高摩擦、改变质量或固定物体来掩盖重建问题。

## 7. 分享结果与可选 UI

分享时复制**整个 `mujoco/` 目录**，不要只复制 `scene.xml`。例如打包当前项目的仿真目录：

```bash
.venv/bin/python -m zipfile -c mujoco-scene.zip "$r2s_run/mujoco"
```

解压后加载其中 `mujoco/scene.xml`。若需要复现分割、模型答案和资产修正过程，则额外保留整个项目目录与原始输入/采集目录。CLI 不会自动生成 ZIP。

GUI 和 CLI 使用相同的重建与检查流程。先完成前面的环境配置，再运行：

```bash
bash scripts/prepare_demo_inputs.sh   # 可选：复制仓库示例到 data 的两个目录，不覆盖已有文件
bash scripts/run.sh
```

访问 [本地界面](http://127.0.0.1:7865/)。输入库固定为仓库下的 `data/image/` 和 `data/depth/`；高级用法可通过环境变量 `R2S_INPUT_DIR` 改变整个输入库的位置。`scripts/run.sh --data runs/gui` 只改变生成项目的输出根目录。

1. 在顶部选择 **单图** 或 **RGB-D 采集文件夹**，选择文件后点击 **导入并自动分割**。新增文件后点刷新。也可以用右上角 **上传单图**，图片会存入 `data/image/<项目ID>.png`。
2. 在原图上查看 SAM3 分割。在下面的物体卡片中删除多余项、选择类型或重量；漏掉的物体可以通过 **点击选物体**（正负点）或 **框选** 补充，预览后确认加入。机械臂自动排除，背景自动简化。
3. 点击 **确认并生成 MuJoCo** 或 **确认并生成 Isaac Sim**。后台依次执行真实 SAM 3D 资产重建、Astra 资产检查/可执行优化、布局与接触优化、10 秒物理检查、MuJoCo 多视角采集和 Astra 最终只读检查。页面显示当前阶段，期间不要同时修改该项目。
4. 底部 **Result** 显示物理检查、平均轮廓 IoU、穿插、Astra 检查状态；RGB-D 还显示深度复核状态。**已生成不代表全部检查通过**，有问题的结果保留并标为待复核。
5. 在 Result 点击 **打开 MuJoCo 预览** 或 **打开 Isaac Sim 预览**，启动这台电脑的原生窗口；也可以展开「输出文件与命令行」复制启动命令。浏览器的「场景预览」是可旋转几何预览，不是原生仿真窗口。也可下载包含模型资源的 ZIP。

GUI 每次确认生成都会保留独立任务目录：

```text
runs/<项目ID>/
├── input/                              # 本项目输入；RGB-D 包含配准数据
├── scene.json                          # 当前选中物体、参数和检查记录
└── jobs/<任务ID>/
    ├── job.json / progress.json         # 任务状态和当前阶段
    ├── worker.log                      # SAM 3D 运行日志
    ├── stages/asset_refinement/         # Astra 资产候选、回答和前后对照
    └── simulation/
        ├── scene.xml                   # Result 中显示的实际加载路径
        ├── validation.json
        ├── workflow.json
        └── evidence/                   # 多视角与 Astra 最终评估
```

旧任务输出不会被再次生成覆盖。修改选中物体或参数后，旧质量指标不再作为当前结果展示，需要重新生成。生成失败会显示错误与日志位置；刷新或重新打开项目能继续查看仍在当前服务中运行的任务，重启服务不会自动续跑模型。

同样输入仍可完全使用 CLI；它默认写入 `runs/<项目ID>/mujoco/`，不会弹出 GUI：

```bash
# 单图
bash scripts/build.sh data/image/tabletop.jpg \
  --selection examples/rgbd_tabletop/selection.json --data runs/single
# RGB-D
bash scripts/build.sh data/depth/tabletop \
  --selection examples/rgbd_tabletop/selection.json --data runs/depth
# 将路径换成实际生成的 XML
.venv/bin/python -m real2sim_workbench.cli view /absolute/path/to/scene.xml
```

`--selection` 示例仅对应这张桌面图；自己的图片应省略或提供对应选择点。API 也保留分割、确认选择、编辑属性和逐阶段操作，见 [API 文档](docs/api.md)。特殊结构当前仍是标注，按普通刚体处理。

### 更换图片尺寸、手动桌面与资产复检

横图、竖图使用原始图像坐标进行分割。原生参考渲染按长宽比适配缓冲区；长边超过 1600 像素时只降低评估渲染分辨率，原图、SAM3 掩码和相机标定保持不变。报告记录实际 `render_size`，轮廓质心仍以原图坐标表示。多视角拼图中的原图与参考渲染采用相同留边，不拉伸竖图。

桌面、支撑关系与摆放在生成中自动处理。以下命令只供需要精确手动设置的开发者使用，GUI 不再提供逐阶段操作。

```bash
bash scripts/recheck.sh runs/替换成项目ID \
  --table-width 1.2 --table-depth 0.8 --table-x 0 --table-y 0
```

不传桌面参数则保留当前值。复检成功导出仍可能物理不通过，Result 会列出问题；带层架、堆叠物体的单图还需要检查支撑关系与位置，尺寸适配不代表这些关系已经自动恢复。

## 8. 方法、验证与仓库约定

- [固定方法与验证约定](docs/workflow-v0.md)
- [模型环境与权重](docs/models.md)
- [RGB-D 示例及局限](examples/rgbd_tabletop/README.md)
- [API](docs/api.md) · [供 agent 使用的 skill](skills/real2sim/SKILL.md)
- [历史单图验证](docs/validation.json) · [历史 RGB-D 验证](docs/rgbd_validation.json)
- [新图完整流程实测（2026-10-24）](docs/new_image_validation.json)：1440×1080、两个按钮盒及浅盘，MuJoCo 物理与 Isaac 动态检查通过，最终视觉仍为部分匹配，用户已接受为 v0 的已知近似。Isaac 图形循环下的抬高/释放测试见 [下落记录](docs/isaac_drop_validation.json)。

本机短回归通过 64 项测试（2026-10-24）；单元测试不调用真实模型。模型环境实际运行过，但不是所有新场景都通过物理或视觉检查。历史成功记录不证明当前每次输出合格，也不证明抓取或任务成功。单图尺度、不可见几何、质量和摩擦需要区分估计与测量；RGB-D 桌面坐标不等于机器人基座标定。

`data/`、`runs/`、`outputs/`、`vendor/`、`.env.local` 和本地环境不进入 Git。随仓库提供的采集例子已获用户授权；新数据不默认公开。第三方组件见 [THIRD_PARTY.md](THIRD_PARTY.md)，自有代码许可尚未选择。


### 自动识别支撑关系并修正摆放

新生成会默认让 Astra 识别「杯子在架子某层上」等支撑关系及物体倾斜，最多两轮执行摆放、渲染和物理反馈。已有场景无需重新生成资产：完整生成会自动修正摆放；单独重检可在命令行运行：

```bash
bash scripts/recheck.sh runs/<项目ID> --auto-layout
```

结果仍在本次任务的 `simulation/scene.xml`，可从 Result 复制查看命令。`stages/relations/report.json` 保存是否采纳及每轮证据；只有物理通过并且轮廓改善的候选才自动采纳（已正确的场景允许轮廓基本不变）。未解决的网格结构和颜色问题会明确标记，不会要求你先手工填写明显的支撑关系。命令行不加 `--auto-layout` 时保留手动位置。


## 双模拟器生成与一键预览

当前 GUI 只保留增删物体、类型和质量设置。SAM 3D → Astra 资产审查/优化 → Astra 支撑关系与三维摆放迭代 → 仿真检查自动运行。选择目标后，在 Result 点击「打开 MuJoCo 预览」或「打开 Isaac Sim 预览」即可启动本机桌面窗口；无须复制命令。MuJoCo 预览初始暂停，空格开始/暂停；Isaac 使用窗口中的播放按钮。界面保留实际未通过项，不能保证单图几何完全准确。

Isaac Sim 配置与路线、测试边界见 [Isaac 使用说明](docs/isaac.md)。若同一项目已经完成精修和场景检查，生成 Isaac 会复用未改变的结果；修改物体选择、质量、类型或桌面后，会重新走完整生成。

```bash
# 从单图或 RGB-D 直接生成 Isaac（其余参数与 MuJoCo 相同）
bash scripts/build.sh data/image/tabletop.jpg --target isaac
bash scripts/build.sh data/depth/tabletop --target isaac
# 已有项目生成 Isaac；自动判断是否能复用
bash scripts/isaac.sh runs/<project-id>
# 原生预览，也可使用 GUI 的打开按钮
.venv/bin/python -m real2sim_workbench.cli view-isaac runs/<project-id>/jobs/<job-id>/isaac/scene.usdc
```
