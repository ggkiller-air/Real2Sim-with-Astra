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

真实几何完整序列：`reconstruct → optimize → settle → validate → review`。每步等待完成，再执行下一步。`reconstruct` 应用共同深度/桌面坐标系，不再调用旧 `initialize`。

新增 action：

- `settle`：重力落桌提出并保存初始姿态，随后仍需 validate。
- `capture`：原生 MuJoCo 多视角，保留现有验证和导出。
- `review`：采集视角、调用本机已登录的 Codex CLI / `gpt-6-astra`、约束建议、候选仿真和轮廓对比，采纳或拒绝后再次输出场景/物理报告/视角。调用失败如实失败，先前可用的场景不丢失。

`GET /api/jobs/{id}` 可附带 progress（模型加载、深度估计、物体序号）。三维结果含 `visual_texture`、`collision_meshes`、`reconstruction` 来源信息；这些字段不能经普通编辑接口更换。

## Astra 资产阶段（独立候选）

`POST /api/projects/{id}/actions/asset_refine`，请求 `{"object_ids":["o_…","o_…"]}`。省略 IDs 时处理当前选中的 SAM 3D 资产；空列表拒绝执行。界面“资产优化”可单独勾选目标。

Astra 实际查看参考图、裁剪和原始资产渲染，返回有界参数；当前模板为橡胶锤和红/绿按钮盒，其他类别保留原样。后端生成参数化网格、GLB、纹理和凸碰撞资产，输出同视角/同光照的前后对照。`scene.reports.asset_refinement` 保存候选清单与路径。

**此 action 不修改 active objects，不组装、不落桌、不运行物理、不覆盖已有导出和验证。** 候选采用已有 mesh / visual_texture / collision_meshes / size 格式，并记录归一化中心偏移及原始资产到规范坐标的变换，供后续采纳时正确转换。不能直接仅替换 mesh 路径忽略尺寸/中心。当前按钮不自动采纳候选；“生成场景”的既有流程保持原样。
