# 模型与运行环境

核心应用只依赖 `pyproject.toml` 中的轻量服务、几何处理与 MuJoCo 包。GPU 模型由独立 Python 子进程执行，避免 Torch / CUDA / 仿真依赖混用。

## SAM3

当前适配接口：官方 `build_sam3_image_model`、`Sam3Processor.set_text_prompt` 和 `predict_inst`。文字模式提出类别实例候选；框/正负点模式针对单个实例。每次请求启动独立进程并释放模型，第一版会有冷启动延迟。

安装与权重申请遵循 [官方仓库](https://github.com/facebookresearch/sam3)。本机接入环境已有 `sam3==0.1.4`、`torch==2.7.1+cu126`；项目不复制该环境。新机器应按官方说明安装，并通过本项目的真实分割请求验证兼容性。

在 `.env.local` 中配置 `R2S_SAM3_PYTHON` 和 `R2S_SAM3_CHECKPOINT`。不要把 token 写进仓库。

## SAM 3D Objects

执行 `scripts/fetch_models.sh` 获取固定版本官方源码。权重可以从 Hugging Face 或用户指定的 [ModelScope 仓库](https://www.modelscope.cn/models/facebook/sam-3d-objects) 获取。

ModelScope 下载（无需改动 SAM3 或 MuJoCo 环境）：

```bash
.venv/bin/python scripts/download_sam3d_modelscope.py
```

默认保存到 `vendor/sam-3d-objects/checkpoints/modelscope/`。下载器固定每个文件的仓库修订，支持断点续传，逐文件核对 SHA-256，完成后生成 `download_verification.json`。当前非空权重、配置和说明约 13.1 GB；源仓库的空 `ss_encoder.safetensors` 不作为有效权重下载，提供的是非空 `.ckpt`。权重和下载记录在被 Git 忽略的目录内。

下载完成只解决主权重获取。还需要独立 Python/CUDA 推理环境；配置中的 MoGe 和代码中的 DINO 等辅助依赖需在首次实际推理时确认，不能把主权重下载完成说成模型已可用。

- 固定源码版本：`f91db411c50efee93d8db7aeb323885650f6f722`。
- Linux / Python 3.11 独立运行时安装：`bash scripts/setup_sam3d.sh`（需要 uv、conda；默认 RTX 4090 的 CUDA 架构 8.9，可通过 TORCH_CUDA_ARCH_LIST 修改）。
- 脚本固定 PyTorch 2.5.1 / CUDA 12.1、PyTorch3D 源码、Kaolin 0.17、MoGe/Utils3D 修订和运行时依赖。官方 Hydra 补丁仅应用在隔离环境内。
- 配置 `R2S_SAM3D_PYTHON`、`R2S_SAM3D_REPO`、`R2S_SAM3D_CONFIG` 后开放入口；`configured` 仅表示路径存在。
- 首次推理还会下载公开的 MoGe 与 DINO 编码器。无需把用户的 HF token 放入仓库。
- 2026-10-23 在本机 RTX 4090 24 GB 完成了真实锤子推理，输出彩色 GLB 和模型位姿；这取代此前“尚未安装/验证”的状态。不能从单次运行推断所有物体/复杂度都适配 24 GB。

ModelScope 主文件 21 个、共 13,105,863,596 字节，全部通过该来源 SHA-256；HF 公共元数据隐藏受限文件哈希，未独立确认与 HF 逐字节一致。安装脚本尚需在第二台干净机器独立重放，当前证据是本机实际安装及推理。

## Astra 视觉检查

使用本机已登录的 Codex CLI，模型明确指定 `gpt-6-astra`，图片通过 `--image` 传入，结果通过 JSON Schema 约束。参考实现遵循 [官方结构化检查循环示例](https://developers.openai.com/cookbook/examples/codex/build_iterative_repair_loops_with_codex)。`prompts/astra_scene_review.txt` 是提示词；`review.py` 实现调用、校验、候选仿真和采纳逻辑，所以不只有 prompt。

图片和几何生成在本机运行；启用 Astra 检查时，参考图、生成视角和场景摘要会通过已登录的 Codex 服务发送给所选模型。该调用需要可用的登录及模型额度。原始对话、凭据和无关文件不包含在构造的请求里。

## 第一版降级方式

导入现有 OBJ/GLB/STL/PLY 网格，或显式生成占位盒以检查交互、布局和导出流程。不会在 SAM 3D 失败时自动换成占位几何，也不会把占位几何标记为模型重建。

## 终端 SOCKS 代理兼容

模型子进程把桌面终端常见的 `socks://` 别名规范化为 `socks5://`，保留地址、认证和 `NO_PROXY`，不修改父终端或关闭代理。SAM 3D 隔离依赖锁定 `socksio==1.0.0`（HTTPX/Gradio）及 `PySocks==1.7.1`（Requests）。若已有旧环境缺少它们，仅补装：

```bash
uv pip install --python .venv-sam3d/bin/python socksio==1.0.0 PySocks==1.7.1
```

该规范化只兼容 SOCKS5 服务的别名写法；明确的 `socks4://`、`socks5://`、`socks5h://` 和 HTTP 代理保持原样。它不证明代理服务连通或模型账号可用。
