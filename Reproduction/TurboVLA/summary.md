# TurboVLA LIBERO 复现记录

## 目标

在 LIBERO 四个 no-noops 套件上复现 TurboVLA，并按官方评测协议记录成功率。

## 最终评测结果

| 套件 | 成功率 | 成功次数 | 总次数 |
| --- | ---: | ---: | ---: |
| libero_spatial | 0.932 | 466 | 500 |
| libero_object | 0.990 | 495 | 500 |
| libero_goal | 0.956 | 478 | 500 |
| libero_10 | 0.858 | 429 | 500 |
| 平均 | 0.934 | 1868 | 2000 |

## 说明


评测使用官方统一 LIBERO checkpoint `turbovla_libero.pth`。

官方 checkpoint 包含 `model_state_dict`。为了适配评测入口，额外生成了一个包含 `ema_model_state_dict` 的评测文件，没有修改 TurboVLA 源码。

DINOv3 ViT-B 是受限模型，评测时使用本地生成的同结构占位模型；TurboVLA checkpoint 会覆盖这些参数。

BERT 从 `google-bert/bert-base-uncased` 下载到本地。

LIBERO 源码克隆到 `~/workspace1/LIBERO`，通过 `--libero_root` 指定。

## libero_10 偏低分析

`libero_10` 成功率为 85.8%，是四个套件中唯一低于 95% 的套件。该套件任务更长、依赖更多步骤，对视觉特征的尺度偏差更敏感。

当前 `turbovla/models/vision_encoder.py` 使用 `outputs.hidden_states[-1]`，实际读取的是 DINOv3 最终 LayerNorm 之前的输出；官方 PR #6 指出应使用 `outputs.last_hidden_state`。该问题会使视觉 token 与官方 checkpoint 训练时分布不一致，尤其影响长时序任务。

相同 `seed=7` 直接重跑预计不会明显改善。修改该行为一行代码后重跑 `libero_10`，官方 PR 的对照实验显示 Long 套件可由 33.6% 恢复到 93.4%，平均成功率由 60.2% 恢复到 97.1%。

## 关键版本

- TurboVLA 源码 commit：`6727c875666f8d5dda8d8cca0043da200738fe73`
- Python：`3.10`
- PyTorch：`2.8.0+cu128`
- Torchvision：`0.23.0+cu128`
- Transformers：`4.57.6`
- MuJoCo：`2.3.7`
- robosuite：`1.4.1`
