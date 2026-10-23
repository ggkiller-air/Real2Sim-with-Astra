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

action 可选 proxy、initialize、optimize、validate、export、reconstruct。请求体 `{}` 或 `{"seconds":3}`。proxy 仅给无几何的选中物体生成占位盒，随后初始化布局，必须明确选择。

轮询任务至完成再重新获取场景。配置不足或模型失败不得报告成功。并发或过期编辑返回 409，数据错误返回 422。
