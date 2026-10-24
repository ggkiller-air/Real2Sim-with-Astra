# Isaac Sim 输出与本机预览

GUI 选择输入、增删物体、设置类型与质量后，直接点击「确认并生成 Isaac Sim」。同一套 SAM 3D、Astra 资产精修、支撑关系与三维摆放优化在后台执行。Result 保留独立质量指标，并可一键打开原生 Isaac Sim 窗口。MuJoCo 有对应的生成、预览按钮。

## 环境

本机验证版本为 Isaac Sim 6.1.0。新机器需安装 Isaac Sim，在 `.env.local` 配置：

```bash
R2S_ISAAC_PYTHON=/absolute/path/to/isaacsim/python.sh
```

本工作区可自动发现相邻 `.tools/isaacsim-6.1.0/python_compat.sh`。安装和运行环境保持独立；不向 SAM3D 或项目主 Python 安装 USD/PhysX，也不依赖 DexBench。原生预览要求从本机图形桌面启动服务；远程/无显示服务仍可生成文件，用本机命令打开。

## 路线选择

采用 **共享场景直接导出 USD**：最终场景记录是共同来源，MuJoCo 输出 MJCF，Isaac 输出 USD。沿用相同显示网格、纹理、CoACD 凸碰撞分块、刚体位姿、质量及编译后的质心/主惯量/惯量轴；显式物理网格与显示网格分离。尺度仅在导出时烘焙一次，单位米、Z 向上。Isaac 使用 PhysX CPU TGS，500 Hz，RTX 渲染。

本机实际试过官方 MJCFImporter（内部 mujoco-usd-converter 0.5.0）。它成功生成 USD，但报告缺失 `physx.usda` 覆盖层，产物是多层、多物理变体，未完成本案例的 PhysX 验证。此次不将这份未经验证的转换产物作为默认交付，也不声称所有 MJCF 导入都失败。现有代码直接掌握网格、碰撞和惯量，因此直接 USD 导出更容易核对这些数据。试验日志保留在本机 `outputs/isaac_route_probe/`。

## 输出

GUI 每次任务使用独立目录：

```
runs/<project-id>/jobs/<job-id>/
  simulation/                    # 需要完整重建时生成的 MuJoCo 参考输出
  isaac/
    scene.usdc                   # 可编辑、可独立打开的 USD
    textures/                    # 相对路径纹理；分享时一起带上
    manifest.json                # 导出数据与场景指纹
    scene_record.json            # 来源场景
    mujoco_reference/            # 同一场景的质量/惯量参照
    validation.json              # 原生 PhysX 动态和接触检查
    preview.png                  # 原生渲染
    worker.log                   # 导出/仿真日志
```

第二个目标可复用已经完成 Astra 精修与场景检查、且指纹一致的结果。修改选择、质量、类型或摆放后不复用旧检查。点击预览不会重新运行模型；预览启动先显示“正在启动”，原生窗口创建成功后才显示已打开。重复点击复用同一进程；失败及超时会显示错误并保留日志。

CLI：

```bash
bash scripts/build.sh data/image/tabletop.jpg --target isaac
bash scripts/build.sh data/depth/tabletop --target isaac
bash scripts/isaac.sh runs/<project-id>
.venv/bin/python -m real2sim_workbench.cli view-isaac /absolute/path/to/scene.usdc
```

## 验证边界

架子/双杯/抽屉案例已在 PhysX 独立运行 10 秒、5000 步，每步读取接触。四个物体的位移、转角、速度和预期支撑接触通过既定阈值；没有增加固定关节或修改质量、摩擦掩盖问题。报告将 `dynamics_check_pass` 与 `physics_check_pass` 分开：目前没有独立完成 PhysX 的初始穿透资格检查，后者为 null，不能用 MuJoCo 的穿透结果冒充 PhysX 测量。

两种引擎接触模型不等价。USD 静/动摩擦映射滑动系数，MuJoCo 的滚动/扭转摩擦不能逐项等价映射。特殊类型仍只有标注，抽屉没有滑轨关节，软体没有柔性仿真。单图未见面、原始网格颜色和形状缺陷仍存在；Isaac 导出本身不会修复它们。当前测试是普通刚体生成/运行/预览，不是 Isaac Lab 任务、抓取或强化学习资格验证。
