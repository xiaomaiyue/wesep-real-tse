# Text-guided TSE — 训练日志汇总（给导师汇报）

数据集：Libri2Mix（基于 LibriSpeech）train-100 训练 / dev 验证 / test 测试。
模型：BSRNN（feature_dim 128, num_repeat 6）+ 冻结 MiniLM-L6 文本编码器（384d）注入。
损失：SI-SDR loss（`loss = -SI-SDR`，**越负越好**）。验证/测试为 held-out（说话人与训练不相交）。
评测指标：**SI-SDRi**（相对混音的 SI-SDR 提升，dB，越高越好）；接受率 = SI-SDRi>1dB 的样本占比。

---

## 一、核心结论

| 现象 | 证据 |
|---|---|
| **机制正确**：模型确实学会"用文本选人+干净提取" | 训练集 swap test：oracle 文本 **+7.67 dB**、给错文本 **−16.09 dB**（25dB 翻转，87% 样本有效提取） |
| **但严重过拟合 / 泛化失败** | test 集 oracle SI-SDRi **≈ 0**（最佳 ckpt −0.03 / 接受率 3%）——在没见过的说话人上塌缩成 passthrough |
| **加数据、正则化、在线动态混音都没解决** | 三次实验 best val 均 ≈ 0，见下表 |

**关键推断**：在线混音提供了近 4 亿种不重复"混音组合"，仍过拟合 → 瓶颈不是混音多样性，而是 **(目标转写 → 目标音频) 的样本对/说话人多样性有限**（train-100 仅 ~251 说话人、27800 句固定配对），模型记住了这些固定配对，迁移不到新说话人/新转写。

---

## 二、逐 epoch train_loss / val_loss（过拟合一目了然）

> 规律一致：**train_loss 持续下降（拟合训练集），val_loss 很早触底后掉头上升（泛化变差）**。

### 实验①：固定 13900 混音，60 epoch
```
ep1  train= 0.106  val= 0.043
ep6  train= 0.021  val=-0.012   ← val 最佳
ep10 train=-0.245  val= 0.103
ep20 train=-2.800  val= 1.198
ep40 train≈-5.5    val≈ 2.6
ep60 train=-7.863  val= 3.623   ← val 一路恶化到 3.6
```

### 实验②：A 抗过拟合（降LR/加weight_decay/缩小模型/早停），20 epoch
```
ep1  train= 0.081  val=-0.012
ep4  train= 0.039  val= 0.277
ep9  train= 0.036  val=-0.003
ep12 train= 0.023  val=-0.011   ← val 最佳
ep20 train= 0.010  val= 0.002
```
→ 正则化**止住了 val 爆涨**（全程稳在 ~0），但 **val 也下不去**（≈passthrough，没学会泛化提取）。

### 实验③：B 在线动态混音（无限混音组合 + 随机SNR/增益），进行中
```
ep1  train= 0.012  val= 0.061
ep4  train=-0.584  val=-0.058   ← val 最佳
ep7  train=-2.174  val= 1.395
ep10 train=-3.454  val= 0.978
ep14 train=-4.880  val= 2.001   ← val 又一路上升
```
→ 即使无限混音，val 仍在 epoch 4 触底后上升 → **过拟合再现**。

---

## 三、评测 / Swap test（test 集，定量）

| 实验 | ckpt | TEST oracle SI-SDRi | TEST interferer | 接受率(>1dB) |
|---|---|---|---|---|
| ①固定13900 | ckpt5(best val) | **−0.03 dB** | −0.13 dB | 3.15% |
| ②A 正则化 | ckpt12(best val) | **−0.00 dB** | −0.00 dB | 0% |
| ③B 在线混音 | (待跑完评测) | 预计 ≈0 | — | — |

### Swap test 诊断（实验① ckpt60，证明机制成立）
| 条件 | SI-SDRi | 接受率 |
|---|---|---|
| **TRAIN-oracle**（训练集+正确文本） | **+7.67 dB** | 86.75% |
| **TRAIN-interferer**（训练集+错误文本） | **−16.09 dB** | 1.25% |
| **TEST-oracle**（测试集+正确文本） | **−3.90 dB** | 44.75% |

→ 训练集上 oracle vs interferer 摆动 **24 dB** = 模型真在用文本选人；但同一模型在 test 上塌掉 = 纯泛化问题。

---

## 四、下一步候选方向

1. **train-360**（~921 说话人、~10万句，约 3.6× 样本对）——直接补"(文本,音频)对 / 说话人"多样性这个真瓶颈。
2. **重审任务/架构**：文本→音频的"记忆捷径"太好走，或需对比学习等强制"用文本内容定位说话人"。

> 工程侧已全部打通：集群 2/4 卡 DDP 训练、在线动态混音 + 文本线索、自动评测 swap test。当前瓶颈是**研究层面的泛化问题**，非工程问题。
