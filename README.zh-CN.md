# Real2Sim 中文使用说明

完整安装、输入目录和测试步骤见 [README](README.md)。

1. 单图放 `data/image/`；RGB-D 采集文件夹放 `data/depth/`。
2. `bash scripts/run.sh`，打开 http://127.0.0.1:7865/。
3. 选择输入并导入，增加/删除物体，选择类型与质量。
4. 点击「确认并生成 MuJoCo」或「确认并生成 Isaac Sim」。重建、Astra 精修、支撑与摆放优化、验证全部自动完成。
5. 在 Result 查看指标，点击对应的「打开预览」，启动本机模拟器窗口。输出路径和命令在折叠区域内。

Isaac 配置和已验证范围见 [Isaac 使用说明](docs/isaac.md)。
