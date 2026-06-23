# 文本线索目标说话人提取 —— 优化历程 (Optimization Journey)

> 目的：记录从「端到端文本条件化(全失败)」到「解耦:分离+文本路由(成功)」再到
> 「抗噪重训」的全过程，方便新窗口接上、知道每一步为什么这么做。
>
> 任务：从两人混音里，**用目标说话人「说的那句话的文字」当线索**，把目标单独提取出来。
> 数据：LibriMix / Libri2Mix, 16kHz, 2 人, min 模式。主干：BSRNN。
> 集群：`maiyue@10.26.1.75`，repo `/share/home/maiyue/tse1/wesep-real-tse`，conda env `wesep`。

---

## 0. 关键指标 & 名词
- **SI-SNRi**：相对原混音的提升(dB)。≈0 = 没分离;越高越好。**核心指标**。
- **swap test**：同一混音分别喂「对的文本(oracle)」和「错的文本(interferer)」，看输出变不变。
  gap≈0 = 忽略文本;gap 大 = 文本驱动。
- **passthrough / 对冲**：模型偷懒直接输出混音(对两个目标都不偏不倚)。是最大陷阱。
- **warm-start**：用官方 `spk_emb_100` checkpoint 的 BSRNN 主干权重做初始化。

---

## 1. 阶段一：端到端「文本直接指挥分离器」—— R2~R8 全部失败

把文本(MiniLM)经 cross-attn + FiLM 注入 BSRNN，端到端训。所有变体都卡在 SI-SNRi≈0、文本被忽略。

| 轮次 | 设置 | oracle | interferer | accept | 结论 |
|---|---|---|---|---|---|
| R2 | 从零 + FiLM + swap | -0.029 | -0.029 | 0% | 忽略文本 |
| R3 | 从零 + contrast + 强FiLM | -0.043 | -0.043 | 0% | 忽略文本 |
| R4 | warm + **冻结主干** | -0.463 | -0.480 | 17.7% | 会分离但乱猜(50/50) |
| R5 | warm + 解冻 + dual-target | -0.0002 | -0.0002 | 0% | 塌成 passthrough |
| R6 | warm + 冻结 + dual-target | -0.047 | -0.051 | 0.75% | cue 盲 |
| R7 | warm + 解冻 + 内容损失 w=1 | ≈0 | — | 0% | 内容损失太弱 |
| R8 | warm + 解冻 + 内容损失 w=10 | ≈0 | — | 0% | 对冲绕不过 |

### 根因(实测诊断，不是猜)
1. **对冲是 cue-盲最优解**：对「输出 vs target」的任何损失(SI-SDR / 内容MSE / CTC-to-target)，
   当模型还不会用文本时，输出混音最划算。判别测试:`content(混音,target)=0.08 < content(错人,target)=0.12`
   → 赌错比对冲更差 → 模型永远选对冲。**调权重没用(R7→R8 证明)**。
2. **机制本身没坏**(诊断 `diag_cue.py`)：cross-attn 对不同文本输出差 ~15%、FiLM gamma 差 ~15%，
   但**最终输出只差 ~1.5%** → 冻结主干把调制冲掉了;解冻又会塌 passthrough。
3. **文本可分性没问题**:oracle vs interferer 文本 embedding 余弦只有 0.12(高度可分)。
4. **更底层**:text cue 要做「内容对齐」(听懂每路说什么再跟文字比),比 speaker-emb 的「按音色选」难一个量级，
   SI-SDR 给不出能 bootstrap 的梯度。

---

## 2. 阶段二:路线 B —— 解耦(分离 + 文本路由)【成功】

**转念**:别逼文本去 steer 分离器(撞对冲)。拆成两步，每步都简单:
1. **分离器**:2 人 PIT 分离(不带 cue)。PIT 结构上**不能输出混音**→ 对冲陷阱消失。
   warm-start 官方 `spk_emb_100` 主干(迁移 464/528,输出头 nspk=1→2 重训)。
2. **路由器**:冻结 wav2vec2 ASR 转录两路 → 跟 cue 文本比 CER → 选最像的那路。**不用训**。

### 结果(全量 test 2000 对, sep ckpt_150)
- **ROUTED SI-SNRi = 17.54 dB, accept 99.2%, 路由准确 99.5%**
- interferer 文本消融 = **−37.9 dB**(换错文本就崩 → 文本驱动铁证)
- 对比 R2~R8 的 ~0 dB:**从撞墙到 17.5 dB**。
- 分离器 train SI-SNR:ep1 12.7 → ep150 平台 20.4 dB(warm-start 第 1 个 epoch 就 12.7)。

**关键认知**:路由 100% 准时,SI-SNRi 完全由分离器决定(路由器不产生 dB),瓶颈从"无解的对冲"变成"可控的分离器质量"。

---

## 3. 阶段三:换路由器(避免显式转文字 / 抗口音)

### CLAP(通用音频-文本模型) —— 失败
直接算音频↔文本相似度，不转文字。但 **CLAP 读不懂"说了什么词"**:两路相似度 3.87 vs 3.91(几乎一样)→ 选错(−36 dB)。
通用 CLAP 只适合"描述/音色"类线索，不适合"词内容"。

### 自训「语音内容↔文本」匹配器 —— 成功
双塔:音频塔(冻结 wav2vec2 → mean-pool → 投影);文本塔(字符级 emb → BiGRU → 投影，从零训，推理不依赖外部文本模型)。
InfoNCE 对比训练(audio_i ↔ 其转录)。`train_matcher.py`。
- 干净 test:ROUTED 17.9 dB, 路由 100%, 消融 −38 dB —— **和 ASR 打平,但全程不转文字**。
- (注:17.9 vs ASR 17.5 只是不同子集抖动;路由都≈100% 时 dB=分离器天花板,路由器不分高下。)

---

## 4. 阶段四:抗噪

### 先测:干净训的模型 **不抗噪**(合成白噪声)
| 条件 | 分离器天花板 | ASR 路由 | 匹配器 |
|---|---|---|---|
| clean | 18.16 | 18.16 / 路由100% | 18.16 / 100% |
| 5dB | 7.42 | 6.77 / 96.7% | 4.35 / 87.7% |
| 0dB | 5.33 | −0.64 / 70.7% | −0.56 / 70.3% |
结论:噪声主要把**分离器**搞垮(天花板 18→5);干净训的小匹配器还不如大 ASR。

### 再做:分离器 + 匹配器都**加噪重训**(`noise_prob=0.7`, SNR 0~20dB 随机)
- 分离器:从 ckpt_150 续训 30 epoch(加噪) → `SEP_NOISY/ckpt_180`
- 匹配器:从头加噪训 60 epoch → `MATCHER_NOISY/matcher_60`

| 条件 | 分离器天花板 | ASR 路由 SI-SNRi/路由 | 匹配器 SI-SNRi/路由 |
|---|---|---|---|
| clean | 18.21 | 18.21 / 100% | 18.01 / 99.7% |
| 5dB | 13.79 | 13.56 / 99% | 12.98 / 98% |
| **0dB** | **13.56** | **12.65 / 96.7%** | **12.05 / 95.7%** |

**结论**:
- 加噪重训决定性:**0dB 从 ~0 dB → ~12–13 dB**。最大杠杆是**分离器**(0dB 天花板 5.3→13.6)。
- 匹配器加噪后基本追平 ASR(0dB:12.05 vs 12.65,路由 95.7% vs 96.7%)。
- ASR 路由仍略强(960h 大模型),但匹配器轻量、自包含、不依赖 ASR。
- 备注:此处用**合成白噪声**;真实噪声(WHAM/babble)需用真噪声做同样增强(见下 TODO)。

---

## 5. 关键文件 / 怎么跑
- 分离器训练:`wesep/bin/train_sep.py`(PIT, warm-start, resume, 加噪 `--noise_prob/--snr_min/--snr_max`)
- 匹配器训练:`wesep/bin/train_matcher.py`(InfoNCE, 字符文本塔, 加噪)
- 端到端 eval:`wesep/bin/eval_routeB.py`(ASR 路由) / `eval_matcher.py`(匹配器) / `eval_compare.py`(并排+加噪)
- 路由可靠性:`wesep/bin/route_check.py`(干净源 100%)
- 失败诊断:`wesep/bin/diag_cue.py`, `content_check.py`
- 本地试听:`separate.py`(纯分离) / `extract_by_text.py`(按文本选人) / `extract_by_clap.py`(CLAP)
- warm-start 加载:`wesep/utils/checkpoint.py`(partial load,名字+shape 匹配)
- 模型:`models/route_b_separator_ckpt150.pt`(本地,88MB);集群 `exp/SEP_NOISY`, `exp/MATCHER_NOISY`
- 现成 ASR:torchaudio `WAV2VEC2_ASR_BASE_960H`(冻结);HF 下载走 `HF_ENDPOINT=https://hf-mirror.com`

整条链:`混音 ─[分离器]→ 两路 ─[路由器(ASR或匹配器)用文本选]→ 目标`

---

## 6. TODO / 下一步想法
- [ ] **用真实 WHAM 噪声**做增强训练(LibriMix 自带 WHAM;当前用的是合成白噪声)。需先把 WHAM wav 弄到集群。
- [ ] 更大数据(train-360)再提分离质量。
- [ ] checkpoint averaging。
- [ ] 关键词/描述线索:匹配器是字符级,天然支持;可专门测。
- [ ] 端到端蒸馏(把"分离+路由"蒸成单模型)。
- [ ] audio+text 联合 cue(官方 ECAPA 编码器权重也能一并 warm-start)。

---
*最后更新:2026-06-19。详尽中文版报告见 `文本TSE实验总结.pdf`。*
