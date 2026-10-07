# 复现说明

## 环境
`````
系统：Windows 11 + WSL2 Ubuntu 24.04
GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8GB 显存
内存：31GB
评测主要受 CPU 仿真和软件渲染限制，GPU 显存占用不高，因为 TurboVLA 推理显存小于 1GB。
`````

## 源码
`````
仓库：https://github.com/H-EmbodVis/TurboVLA.git
分支：`main`
Commit：`6727c875666f8d5dda8d8cca0043da200738fe73`
本地路径：`~/workspace1/VLR_Learning/TurboVLA`
`````

## 外部资源

官方 release：`H-EmbodVis/TurboVLA`
统一 LIBERO checkpoint：`pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth`
BERT：`google-bert/bert-base-uncased`
DINOv3 ViT-B 官方仓库受限，因此生成了：
    `pretrained/dinov3-vitb16-standin`
    使用相同的 `DINOv3ViT` 架构和预处理配置
LIBERO 源码：`~/workspace1/LIBERO`
Hugging Face 下载使用镜像 `HF_ENDPOINT=https://hf-mirror.com`。

## 重要问题与处理

1. PyTorch `2.3.1+cu121` 不支持 RTX 5070 的 `sm_120` 架构，已升级到 `2.8.0+cu128`。
2. 一次中断的 pip 安装导致 `triton` 和部分 NVIDIA CUDA 动态库损坏，已从 PyTorch `cu128` 源重新安装。
3. MuJoCo 从 `3.14.0` 降级到 `2.3.7`，以匹配 `robosuite 1.4.1`。
4. 安装了系统包 `libosmesa6`，用于无界面 MuJoCo 渲染。
5. 必须设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，因为 LIBERO 初始状态文件使用旧版 pickle。
6. 官方统一 LIBERO checkpoint 只包含 `model_state_dict`，而评测入口要求 `ema_model_state_dict`。因此生成了 `turbovla_libero_eval.pth`，没有修改 TurboVLA 代码。

## 评测协议

- 四个套件：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`
- 每个套件 10 个任务
- 每个任务 50 次测试
- 每个套件 500 个 episode
- `seed=7`
- `num_open_loop_steps=12`
- `chunk_size=12`
- `precision=bf16`

## 结果

| 套件 | 成功率 |
| --- | ---: |
| libero_spatial | 0.932 |
| libero_object | 0.990 |
| libero_goal | 0.956 |
| libero_10 | 0.858 |
| 平均 | 0.934 |

当前四个套件的平均成功率为 93.4%，低于实验室 95% 的目标。差距主要来自 `libero_10`。

## libero_10 偏低分析

`libero_10` 为 429/500，成功率 85.8%，是四个套件中唯一明显低于 95% 的套件。只使用相同命令、相同 `seed=7` 重跑，预期不会明显提升；评测在固定种子和相同环境下基本可复现，随机波动不足以弥补约 9 个百分点的差距。

更可能的原因是当前 TurboVLA 主分支中的视觉编码器实现与官方 checkpoint 不完全匹配：

- 文件：`TurboVLA/turbovla/models/vision_encoder.py`
- 当前代码使用 `outputs.hidden_states[-1]`
- 这对应 DINOv3 最后一层 Transformer block 在最终 LayerNorm 之前的输出
- 官方 release checkpoint 更可能对应经过最终 LayerNorm 的 `outputs.last_hidden_state`

官方仓库 PR #6 已记录该问题。其对照实验中，`libero_10` 对应套件 Long 在修复前为 33.6%，修复后为 93.4%；四个套件平均从 60.2% 恢复到 97.1%。本次结果虽没有该 PR 的修复前结果那么低，但 `libero_10` 偏低的方向一致，可能还与 transformers、PyTorch、MuJoCo 渲染等环境版本差异叠加有关。

建议将第 110 行改为：

```python
tokens = outputs.last_hidden_state
```

修改后重新评测 `libero_10`，预期可恢复到约 93% 以上。
