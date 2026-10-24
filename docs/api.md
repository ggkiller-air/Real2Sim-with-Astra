# 本地 API

启动后访问 `/docs` 查看 OpenAPI。默认地址 http://127.0.0.1:7865。

| 方法 / 路径 | 用途 |
| --- | --- |
| GET `/api/status` | 配置情况、引擎和特殊结构状态 |
| GET / POST `/api/projects` | 列表；multipart file 上传图片 |
| GET `/api/projects/{id}` | 返回 scene 和当前 job |
| PUT `/api/projects/{id}` | 完整场景与当前 revision；编辑已有物体元数据 |
| POST `/api/projects/{id}/segment` | 自动扫描、文字候选、框/点实例修正 |
| POST `/api/projects/{id}/objects/{oid}/mesh` | multipart file 网格导入 |
| POST `/api/projects/{id}/actions/{action}` | 后台几何/布局/检查/导出 |
| GET `/api/jobs/{job_id}` | queued/running/done/failed；失败带 error |
| GET `/api/projects/{id}/download` | 当前导出 ZIP |

自动扫描：`{"mode":"auto"}`。内置桌面类别提示、机器人排除掩码和重复候选合并；已有物体（包括用户删除的物体）不会因再次扫描而重复加入。`scene.preprocessing` 保留背景/排除来源，不能通过 PUT 修改。检测到桌面时仅设置掩码边界近似，`calibration_source` 仍为 assumed。

文字分割：`{"mode":"text","labels":["paper cup","bowl"]}`。

实例分割：`{"mode":"instance","box":[x1,y1,x2,y2],"points":[[x,y,1],[x,y,0]],"label":"cup"}`。正点为 1、负点为 0；带 object_id 修正，不带则新增。

action 可选 proxy、initialize、optimize、settle、validate、export、reconstruct、capture、review。请求体 `{}` 或 `{"seconds":3}`。proxy 仅给无几何的选中物体生成占位盒，随后初始化布局，必须明确选择。

轮询任务至完成再重新获取场景。配置不足或模型失败不得报告成功。并发或过期编辑返回 409，数据错误返回 422。

## 真实重建与视觉反馈

点选预览：`POST /segment`，例如 `{"mode":"instance","points":[[230,345,1],[345,380,1],[320,300,0]],"preview":true}`。`scene.reports.selection_preview` 存放候选；`POST /selection/accept` 携带 `{"revision":当前修订}` 确认。与已有掩码重复时复用原 ID。`DELETE /selection/preview` 携带相同 revision 结构取消预览。

旧逐阶段 API 序列：`reconstruct → optimize → settle → validate → review`，仍保留用于调试，但不包含完整的全资产 Astra 优化。完整工作流请使用下方 `generate`。逐阶段调试时每步等待完成，再执行下一步。`reconstruct` 应用共同深度/桌面坐标系，不再调用旧 `initialize`。

新增 action：

- `settle`：重力落桌提出并保存初始姿态，随后仍需 validate。
- `capture`：原生 MuJoCo 多视角，保留现有验证和导出。
- `review`：采集视角、调用本机已登录的 Codex CLI / `gpt-6-astra`、约束建议、候选仿真和轮廓对比，采纳或拒绝后再次输出场景/物理报告/视角。调用失败如实失败，先前可用的场景不丢失。

`GET /api/jobs/{id}` 可附带 progress（模型加载、深度估计、物体序号）。三维结果含 `visual_texture`、`collision_meshes`、`reconstruction` 来源信息；这些字段不能经普通编辑接口更换。

## Astra 资产阶段（独立候选）

`POST /api/projects/{id}/actions/asset_refine`，请求 `{"object_ids":["o_…","o_…"]}`。省略 IDs 时处理当前选中的 SAM 3D 资产；空列表拒绝执行。界面“资产优化”可单独勾选目标。

Astra 实际查看参考图、裁剪和原始资产渲染，返回有界参数；当前支持橡胶锤/红绿按钮盒模板、`mesh_cleanup` 保形清理和 `keep`。每件资产均返回原始质量判断，无法修复的缺陷保留为 `needs_repair`。后端生成参数化网格、GLB、纹理和凸碰撞资产，输出同视角/同光照的前后对照。`scene.reports.asset_refinement` 保存候选清单与路径。

**此 action 不修改 active objects，不组装、不落桌、不运行物理、不覆盖已有导出和验证。** 候选采用已有 mesh / visual_texture / collision_meshes / size 格式，并记录归一化中心偏移及原始资产到规范坐标的变换，供后续采纳时正确转换。不能直接仅替换 mesh 路径忽略尺寸/中心。当前按钮不自动采纳候选；“生成场景”的既有流程保持原样。

## 固定无界面工作流

`bash scripts/build.sh INPUT` 支持单 RGB 文件和标定采集目录，默认审查全部选中资产并作最终 Astra 只读复核，见 [workflow-v0](workflow-v0.md)。GUI 的「确认并生成」已接入相同的完整流程；旧逐阶段 API 仍兼容。独立 `asset_refine` action 仍只保存候选，不自动采纳。

## 双路径 GUI 完整生成

- `GET /api/inputs`：返回输入库根目录和 `image`/`depth` 候选。默认仓库的 `data/`；输入与 runs 输出相互独立。
- `POST /api/inputs/import`：`{"kind":"image","name":"tabletop.jpg"}` 或 `{"kind":"depth","name":"tabletop"}`。创建项目并完成深度配准，不运行模型；接着对返回项目调用 `segment` 的 `auto` 模式。输入名相对于对应库目录，拒绝越界路径。
- `POST /api/projects`：上传单图仍可用，归档到 `data/image/<项目ID>.png`。
- `POST /api/projects/{sid}/generate`：`{"revision":当前修订号,"seconds":10}`。返回可轮询任务；只重建所选对象，默认全部经过 Astra 资产检查，之后布局、物理、多视角最终检查。生成期间拒绝同项目编辑或重复生成。
- `GET /api/jobs/{jid}`：`queued/running/done/failed`，可含 `progress.stage`。`done` 不等于物理检查通过。
- `GET /api/projects/{sid}/result`：输出 `project`、`xml`、`view_command` 和 `reports`。查看 `validation`、`silhouette`、`depth_quality`、`workflow_quality`；没有当前导出时 XML/命令为 null。

GUI 任务导出到 `jobs/<jid>/simulation/`；CLI 默认仍是项目下 `mujoco/`。重复生成不覆盖以前的 GUI 任务。项目导入后 GUI 的点/框坐标以当前 `scene.image` 为准；RGB-D 是校正后的图片，不能把原始图像坐标直接传给此接口。

## 桌面编辑与复检

`scene.table.center_xy` 是桌面相对场景原点的 XY 偏移（米），顶面仍为 Z=0。通过既有 PUT 项目接口编辑宽、深、偏移并设置 `calibration_source=user`，将使旧质量结果失效。`POST /api/projects/{sid}/table/fit` 接受 `{revision}`，按选中资产的旋转包围盒和 6 cm 余量铺桌；没有资产时使用默认桌面。

`POST /api/projects/{sid}/recheck` 接受 `{revision, seconds:10}`，复用所有选中已有资产，保持手动位置，重新导出、验证和最终只读评估，不重跑资产模型。返回普通任务，可用 `/api/jobs/{jid}` 轮询。CLI 使用 `scripts/recheck.sh PROJECT [--table-width W --table-depth D --table-x X --table-y Y]`。


`POST /api/projects/{sid}/layout`：`{revision, seconds:10}`，复用已有资产，调用 Astra 自动识别支撑关系、承托层及倾斜，再迭代最多两轮摆放和原生物理/轮廓检查。CLI 为 `scripts/recheck.sh PROJECT --auto-layout`。质量、摩擦和资产网格不变；仅通过检查的候选被采纳，失败候选及模型答案保留在 `stages/relations/`。生成入口默认执行这一阶段。

## 双模拟器与原生预览

`POST /api/projects/{sid}/generate` 增加 `target: "mujoco" | "isaac"`（默认 mujoco）。Isaac 在共享全流程之后导出 USD 并独立检查；已有未改变且完成审查的结果可直接复用。结果接口同时返回 `xml`、`usd` 和两份验证报告。

`POST /api/projects/{sid}/preview/{engine}`：`{revision}`，engine 仅允许 mujoco 或 isaac，启动本机固定预览程序，返回 starting/ready。`GET` 同一路径轮询状态。不会执行客户端提供的命令或任意文件，仍有同源、项目修订、计算互斥检查。原生窗口已创建才返回 ready；启动超时/退出可见，日志保存在 `jobs/preview_<engine>_<id>/`。
