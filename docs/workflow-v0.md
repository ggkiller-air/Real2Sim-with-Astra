# 固定工作流：rigid_v0.4

本文件定义 CLI `build` 与 GUI「确认并生成」共用的方法和测试约定。范围是单图或已标定 RGB-D 到 MuJoCo 普通刚体，以及同一已审查场景到 Isaac Sim 的直接 USD 导出。既有例子的物体质量、摩擦和几何准确性问题不因流程固定而被视为解决。

## 阶段与职责

| 阶段 | 输入 → 输出 | 实现与边界 |
| --- | --- | --- |
| 输入 | 单张 RGB 或采集文件夹 → 主图、标定与来源 | `pipeline.create_input`；目录走 `rgbd.register_capture`，当前目录格式见 RGB-D 例子 |
| 分割/选择 | RGB → 选中实例、支撑和排除掩码 | SAM3；排除机械臂/夹爪，支持选择文件中的原图点击坐标；保留用户选中的对象 |
| 初始资产 | 主图和掩码 → 彩色初始网格 | SAM **3D Objects**，与只分割的 SAM3 区分；单图用预测点图，RGB-D 用实测配准点图 |
| 初始坐标 | 网格/点图 → 共同支撑系和尺度 | 单图尺度为假设；RGB-D 是米制桌面局部坐标，不是机器人基座标定 |
| 全资产 Astra 审查 | 原图、资产渲染、辅助 RGB → 每个 ID 的决策与候选 | 默认处理所有选中资产，保留原始几何。一次模型调用覆盖本批资产，不按名称筛选目标 |
| 候选组装 | 保留/清理/参数化候选 → 场景资产 | 保留中心偏移、尺寸、坐标变换和来源；质量/摩擦不因修形而擅自修改。采纳用于后续评估，不等于质量认证 |
| 位姿与物理 | 初始场景 → 可检查的导出 | RGB-D 使用受深度约束的支撑姿态选择；两路径共用 Astra 支撑图与姿态修正、重力落稳和独立稳定性检查；RGB-D 候选另检查深度误差和覆盖率不明显退化 |
| 原生视角与复核 | 导出 → 轮廓、深度、诊断视角和 Astra 结论 | 最终 Astra 调用仅保存评价/建议，`suggestions_applied=false`；此为自动摆放循环后的独立检查；自动摆放最多两轮，不无限修改 |
| 交付 | MJCF、资产、报告 | `mujoco/scene.xml` 和同目录 `assets/`。`workflow.json` 汇总方法版本与待复核项，不能只看是否生成了 XML |

`assemble` 的初始坐标恢复早于资产审查，是为了正确处理 SAM 3D 的共同坐标和米制参考；它不等于资产阶段已经结束，也不意味着允许改变物理参数来遮掩错误。

## Astra 的可执行动作

Astra 不直接运行任意生成代码。`prompts/astra_asset_refinement.txt` 定义审查内容，`asset_refinement.py` 提供 JSON schema、参数边界、建模/清理执行器和前后渲染。因此实现是 **prompt + 受约束代码 + 检查报告**。

- `hammer` / `button_box`：支持的规则模板，按有界结构比例重建。
- `mesh_cleanup`：轻度保形清理。固定 4 次 Taubin 平滑，位移最多为包围盒对角线的 1%；不删除小组件，保留采样颜色，重新生成显示和碰撞资产。不修复缺失刷毛、孔洞、错误颜色或类别结构。
- `keep`：审查后保留。原始资产基本可接受记为 `review_status=acceptable`；现有执行器无法修复的结构错误记为 `needs_repair`，继续导出但标为待复核。不能把“没能力修”报告成“已经修好”。

每个选中 ID 恰好返回一次。原始网格、模型答案、候选和比较图保留。通用清理可能保留源网格的开放表面；凸碰撞体并不证明显示几何闭合、厚度真实或内腔正确。

最终 Astra 复核通过 `review.astra_review` 查看主视角和原生诊断视角；不修改资产或摆放。在此之前，`relations.optimize_relations` 默认执行最多两轮自动反馈：Astra 识别支撑父对象、实际承托层和直立需求，代码提取网格平面、摆正近似主轴、恢复子物体高度和图像对应位置，再运行原生渲染及物理检查。失败轮的场景、报告、渲染全部保留，并交给下一轮 Astra。候选只有物理通过且轮廓改善（或已完成场景无退化）才被采纳；失败不会偷偷改为固定关节或修改质量、摩擦。

## 输入路径

| 项目 | 单图 | RGB-D |
| --- | --- | --- |
| 入口 | 一个 RGB 文件 | 一个符合例子格式的目录 |
| 点图 | MoGe 预测 | 配准后的实测深度；缺测填补只作模型输入 |
| 尺度 | 假设/先验 | 传感器米制尺度 |
| 位置约束 | 预测共同坐标、掩码、支撑 | 有效实测表面、相机标定、掩码、支撑 |
| 额外视角 | 默认无 | Astra 外观参考；历史安装未核验，不做精确几何融合 |
| 未见表面、质量/摩擦 | 推断/假设 | 同样是推断/假设 |

原始例子与坐标约定见 [RGB-D 桌面例子](../examples/rgbd_tabletop/README.md)。深度能约束尺度和可见表面，不能自动消除资产结构错误。比较有无深度的收益需要相同主 RGB、选择、资产和物理设置的受控实验；当前两个历史结果不是这样的消融实验。

## 运行与测试

先按 [模型设置](models.md) 配置本机环境及 `.env.local`，并登录可调用 `gpt-6-astra` 的 Codex CLI。Astra 调用用于全资产审查、最多两轮支撑/姿态反馈和最终视角复核，发送相关图像；SAM3/SAM 3D 在本机运行。

```bash
# 在仓库根目录运行；无须启动 UI。
.venv/bin/python -m pytest -q
bash scripts/build.sh examples/rgbd_tabletop/capture/image_external_rgb.jpg \
  --selection examples/rgbd_tabletop/selection.json
bash scripts/build.sh examples/rgbd_tabletop/capture \
  --selection examples/rgbd_tabletop/selection.json
```

默认开启全部资产审查。旧参数 `--refine-assets` 仍兼容；仅做消融时用 `--skip-asset-review`，它同时跳过自动支撑推断和最终 Astra 调用，并明确标注未审查。模型调用/参数校验失败会报错，不会静默换成占位盒或跳过 Astra。默认物理检查为 10 秒。

若在三维资产组装前失败，可用 `bash scripts/build.sh --resume runs/<project-id>` 复用已选掩码；旧日志保留在原目录，新推理使用 `reconstruct_retry_<编号>`。这不是任意阶段断点续跑。具体中文步骤见 [运行指南](../README.md)。

程序打印独立项目路径，最终打开：

```bash
.venv/bin/real2sim view runs/<project-id>/mujoco/scene.xml
```

测试时分别检查：

1. 掩码是否包含目标、排除机械臂，选择是否正确。
2. `scene.reports.asset_refinement.path` 指向的 `report.json` 是否覆盖全部选中 ID；候选是否丢失结构。
3. `mujoco/evidence/views/`：原生主视角和其他视角；是否只在一个视角看起来正确。
4. `mujoco/validation.json`：支撑、穿插、漂移、速度和警告。初始 3 秒落稳后仍不稳定时最多加两轮各 8 秒落稳，保留失败记录，不放宽检查阈值。
5. RGB-D 的 `depth_agreement.json` 与 `depth_quality.json`：误差必须结合覆盖率。15 mm/80% 是人工复核启发阈值，不是测量精度保证。
6. `mujoco/evidence/astra/review.json` 与 `workflow.json`：哪些问题未解决，最终建议是否尚未应用。

退出成功表示执行完成且短时物理检查通过。`workflow.json` 仍可能是 `needs_review`；逐件资产保真没有因此自动认证。失败候选与报告应保留，不能通过固定物体、提高摩擦或修改质量来掩盖错误。

## 历史验证边界

历史实际 SAM3/SAM 3D/Astra 与 MuJoCo 运行见 [单图记录](validation.json) 和 [RGB-D 记录](rgbd_validation.json)。自动测试覆盖路由、真实清理/导出、来源保留和质量标记；GUI 已接入共享流程。后续新增支撑优化及双模拟器验证分别见 [关系修正记录](relations_validation.json) 和 [Isaac 记录](isaac_validation.json)。每份记录只证明所列输入、版本和实际执行阶段，不将复用资产的检查视为重新运行完整模型链路。

当前特殊结构只保存标签，尚不实现铰链、抽屉或软体；Isaac Lab 后续再接入。仓库原始例子由用户明确授权收入，其余新采集数据、权重、环境、凭据及生成运行目录不默认进入 Git。

## GUI 与命令行共用的边界

输入库使用 `data/image/` 和 `data/depth/`。`prepare_input` 完成项目导入；RGB-D 先配准并替换为校正 RGB，GUI 的 SAM3 交互全部在校正图像坐标内。CLI 的既有 selection manifest 仍以原图坐标选候选，然后变换掩码。两者随后共用 `generate_selected` 和 `finish_scene`。只重建保留的对象，质量与摩擦参数保留用户设置。

GUI 生成期间锁定项目修改。每次任务使用独立目录，旧证据不覆盖；完整流程自己保存中间及最终 scene，任务调度器不得再用旧 revision 覆盖。物理不通过是完成但待复核，模型/导出异常才是任务失败。Result 中的视觉、深度和物理指标分别呈现；最终 Astra 建议只读。CLI 仍可单独使用，不依赖 Web 服务。

## 自动摆放与复用已有资产（v0.4）

新生成默认执行自动关系修正。GUI 只提供最终生成按钮；支撑关系与姿态修正默认执行。已有资产的单独修正和手动桌面设置保留为 CLI/API 开发功能。

```bash
bash scripts/recheck.sh runs/<project-id> --auto-layout
```

接口 `POST /api/projects/{sid}/layout` 使用 `{revision, seconds:10}`。共享实现位于 `relations.py`，模型提示位于 `prompts/astra_relations.txt`。候选、模型原始答案、落稳前与落稳后物理报告保存在任务的 `stages/relations/round_N/`；`report.json` 记录是否采纳。未通过时 Result 显示自动摆放未解决，失败候选不会覆盖正式场景。

算法适用于普通刚体、近似水平承托面；直立修正限制 30°，XY 建议限制每轮 12 cm。杯子高度按中间承托面计算，不取架子最高点。它不修复源资产缺失结构或颜色，也不实现抽屉关节。深度门限保留已可靠观测物体的误差（最多增加 5 mm）和覆盖率（最多下降 5 个百分点）；该分支本次只做回归检查，不能将单图架子验证泛化成 RGB-D 全场景成功。

Isaac Sim 共享资产/摆放流程与原生验证见 [Isaac 说明](isaac.md)。GUI 已取消逐阶段操作，提供两种目标的生成与原生预览。
