# 模型与运行环境

核心应用只依赖 `pyproject.toml` 中的轻量服务、几何处理与 MuJoCo 包。GPU 模型由独立 Python 子进程执行，避免 Torch / CUDA / 仿真依赖混用。

## SAM3

当前适配接口：官方 `build_sam3_image_model`、`Sam3Processor.set_text_prompt` 和 `predict_inst`。文字模式提出类别实例候选；框/正负点模式针对单个实例。每次请求启动独立进程并释放模型，第一版会有冷启动延迟。

安装与权重申请遵循 [官方仓库](https://github.com/facebookresearch/sam3)。本机接入环境已有 `sam3==0.1.4`、`torch==2.7.1+cu126`；项目不复制该环境。新机器应按官方说明安装，并通过本项目的真实分割请求验证兼容性。

在 `.env.local` 中配置 `R2S_SAM3_PYTHON` 和 `R2S_SAM3_CHECKPOINT`。不要把 token 写进仓库。

## SAM 3D Objects

执行 `scripts/fetch_models.sh` 获取固定版本官方源码。代码不会自动下载受限权重。

- 固定源码版本：`f91db411c50efee93d8db7aeb323885650f6f722`。
- 依照 [官方环境说明](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) 安装独立环境。
- 在 [官方权重页面](https://huggingface.co/facebook/sam-3d-objects) 申请访问权限，并在本机登录下载。
- 官方说明要求至少 32 GB 显存；当前主机为 24 GB，因此即便取得权重权限，仍需实际确认运行可行性，不保证当前硬件能跑通。
- 配置 `R2S_SAM3D_PYTHON`、`R2S_SAM3D_REPO`、`R2S_SAM3D_CONFIG` 后，界面才开放调用入口。

当前状态：下载权重返回 403，用户选择先完成其他模块。适配器基于官方 API 编写，尚未完成真实模型推理验证。它保存 GLB 以及原始模型姿态，导入几何供手工布局；模型坐标系到场景坐标系的转换尚未实模验证，不自动应用预测位姿。

## 第一版降级方式

导入现有 OBJ/GLB/STL/PLY 网格，或显式生成占位盒以检查交互、布局和导出流程。不会在 SAM 3D 失败时自动换成占位几何，也不会把占位几何标记为模型重建。
