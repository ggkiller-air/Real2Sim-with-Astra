# RGB-D 桌面例子

用户提供的 `camera_frame_20261023_193740` 原始采集文件，按用户要求收入仓库。`capture/` 保留 RGB、原始 uint16 深度、米制深度复核文件、当前标定和历史标定；不包含模型权重。机械臂、夹爪和远处设备不生成资产。

## 两条输入路径

在仓库根目录，模型环境与 `.env.local` 配好后：

```bash
# 单图：MoGe 预测共同深度，尺度仍依赖假设。
bash scripts/build.sh examples/rgbd_tabletop/capture/image_external_rgb.jpg \
  --selection examples/rgbd_tabletop/selection.json --refine-assets

# RGB-D：当前 D455 工厂标定 + 实测米制深度。
bash scripts/build.sh examples/rgbd_tabletop/capture \
  --selection examples/rgbd_tabletop/selection.json --refine-assets
```

输出打印 `PROJECT` 路径；最终结果为 `runs/<id>/mujoco/scene.xml`，带独立 `assets/`、原生视角、稳定性与深度一致性报告。查看：

```bash
.venv/bin/real2sim view runs/<id>/mujoco/scene.xml
```

`selection.json` 是原始主 RGB 图中的点击坐标和对象名称，选择六个 SAM3 实例：锤子、刷子、胡萝卜、苹果、绿按钮盒、红按钮盒。它不指定 3D 位置。省略选择文件会自动选择去重后的候选；自动识别不保证正确类别或完整覆盖。当前固定工作流默认让所有选中资产经过 Astra 审查：锤子/按钮盒可参数化重建，其他物体可做轻度保形清理或明确保留。旧 `--refine-assets` 参数仍兼容；显式跳过用 `--skip-asset-review`，结果会标为未审查。最终原生视角还会进入 Astra 只读复核。完整方法见 [workflow-v0](../../docs/workflow-v0.md)。

## 深度怎样使用

1. 校验 RGB/深度相机序列号、uint16 深度单位、NPY 米制复核和工厂外参。
2. 原始深度反投影到深度相机，以工厂外参变换到 RGB 相机。遵循当前镜头模型、裁剪及缩放，再配准到去畸变的主 RGB。
3. 将米制点图输入真实 SAM 3D 推理，代替 MoGe 预测。零值缺测只在模型条件输入中填补；桌面拟合和位置评估只用真实有效点。
4. 桌面实测点拟合支撑平面，建立米制局部世界。对每个物体的可见实测表面拟合位置/尺度；生成未见表面仍由模型估计。
5. 计算凸支撑稳定姿态假设，受限拟合实测表面；用原生轮廓与光轴深度误差排序。随后运行原有消穿插、重力落稳和独立 10 秒稳定性检查。保留落稳前后深度误差及覆盖率，防止将“稳定”误当作“还原正确”。
6. D435 和腕部 RGB 实际传给 Astra 作形状、外观参考。本版本没有验证历史相机安装外参，也没有本次腕部法兰世界位姿，因此没有进行三相机的标定点云融合或三角测量。

深度路径目前支持本例 `capture_metadata.json` 所描述的 D455 格式，不是任意 RGB + depth 文件的自动适配器。换采集设备需提供其真实内参、深度单位、镜头模型、RGB 预处理和深度到 RGB 外参。主 RGB 文件名为 `image_external_rgb.jpg`。

## 边界

- 桌面局部坐标有实测尺度，不等于机器人基座标定。
- 多相机只有主机接收时间，不能宣称曝光同步；适合当前静态桌面。
- 质量、摩擦和不可见几何仍是估计；全部为自由刚体，没有按钮关节、软体或真实材料辨识。
- 深度误差只统计可见重叠区域，必须同时看覆盖率；它不是独立测量精度认证。
- 首次落稳通过了物理检查，但视觉摆放变差，失败候选保留在运行记录中。最终结果须结合 `validation.json`、`depth_agreement.json` 和原生渲染查看。

投影约定参考 [RealSense SDK 源码](https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp)，相机导出参考 [MuJoCo camera 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)，并有原生渲染投影测试。

## 本次结果

验证索引见 [rgbd_validation.json](../../docs/rgbd_validation.json)。六个自由刚体通过 10 秒/5000 步稳定性检查，平均轮廓 IoU 0.758。最终锤子光轴深度误差中位数约 42 mm、胡萝卜可见深度重叠覆盖约 62%，`depth_quality.json` 明确标为需要复核；其余物体约 5–11 mm。稳定不等于高保真，不应把桌面平面 1.9 mm 拟合残差当作物体重建精度。
