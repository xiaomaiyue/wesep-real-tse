# 在 4090D 集群上训练 text-guided TSE — 操作手册

集群三类节点（职责严格分开）：

| 节点 | IP | SSH | 干什么 |
|------|------|-----|--------|
| **storge**（存储） | 10.26.1.74 | ✅ | **传文件 / 解压**（IO 都在这里做） |
| **login01**（登录） | 10.26.1.75 | ✅ | 建环境、提交/监控作业（**不传大文件、不跑计算**） |
| **gpu01-05**（计算） | 10.26.1.101-105 | ❌ | 只能通过 SLURM 进，跑训练 |

三类节点共享同一套 NFS `/share`，家目录 = `/share/home/<user>`（SSD）。数据传一次到 storge，GPU 作业即可读到。

---

## 步骤

### 0. 前提
- Mac 和集群都在校园网（校外先连 VPN）。
- 知道自己的集群用户名；首次登录建议 `yppasswd` 改密码、`conda init` 一次。

### 1. 传文件 —— 在 **Mac** 上运行（目标是 storge 10.26.1.74）
```bash
cd /Users/xiaomaiyue/Downloads/slr/tse1/wesep-real-tse/examples/audio/librimix/cluster
bash transfer_from_mac.sh <你的集群用户名>
```
传 ~3.1G 数据 + 代码到 `storge:~/tse1/`。**注意：传到 .74（storge），不是 .75（login01），更不是 GPU 节点。**

### 2. 建环境 + 修路径 —— 在 **login01** 上运行
```bash
ssh <你的集群用户名>@10.26.1.75
bash ~/tse1/wesep-real-tse/examples/audio/librimix/cluster/setup_cluster.sh
```
做三件事：把数据文件里写死的 Mac 绝对路径 `/Users/...` 改成集群的 `~/tse1`；建 `raw.list`；建 conda 环境 `wesep` + 装依赖（含 CUDA torch）。

### 3.（强烈建议）预检 —— 单卡 srun，几分钟
正式占 2 卡前，先确认管线在 CUDA 上跑得通、loss 会动：
```bash
srun --partition=gpu2node --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
# 进入 GPU 节点后：
conda activate wesep
cd ~/tse1/wesep-real-tse/examples/audio/librimix
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# ~50 条 mixture、常数 LR 的 overfit 探针：train SI-SDR 应能escape 0 dB
D=~/tse1/libri_data/Libri2Mix/wav16k/min
python local/train_local_text.py \
  --train_samples $D/train-100/samples.jsonl --train_cues $D/train-100/cues_text_oracle.yaml \
  --val_samples   $D/dev/samples.jsonl       --val_cues   $D/dev/cues_text_oracle.yaml \
  --exp_dir ./exp/preflight --device cuda --no_sched --epochs 60 --val_mixtures 50
exit   # 释放这张卡
```
看到 train/val SI-SDR 往正方向爬（本地探针曾 0 → +3.5 dB）就算通过。

### 4. 正式训练 —— 在 **login01** 上 sbatch（2×4090D）
```bash
cd ~/tse1/wesep-real-tse/examples/audio/librimix/cluster
sbatch train_text_tse.slurm
```
配方：`confs/tse_bsrnn_text.yaml`（BSRNN feature_dim128×6，MiniLM-384 文本 cue，fusion=multiply，batch15，`ExponentialDecrease`，150 ep）。**用 epoch 调度器，不依赖 val**——彻底避开本地那次 val-OOM→NaN→LR 衰减到 0 的坑。

### 5. 监控
```bash
slmwatch                 # 集群总览 + 你的作业
squeue -u $USER          # 作业队列
tail -f tse_text-<jobid>.out   # 训练日志（在你 sbatch 的目录下）
scancel <jobid>          # 需要时终止
```
checkpoint 在 `exp/BSRNN_TEXT/models/`。

### 6.（训练完）评测 / cue 消融
改 `--config` 不变、把 cue 的 `resource:` 换成 interferer / random 版本，跑 `wesep/bin/infer.py`（见 run.sh stage 5-6）。headline 诊断：oracle vs interferer 文本若结果翻转 = 模型真在用文本。

---

## 可能要调的地方
- **想更快/更大**：换 `gpu3node`（3 节点）或加卡 → 改 sbatch 的 `--gres=gpu:N`、`--partition`、`nproc_per_node`，并按比例调 conf 的 `batch_size` / `lr`。
- **scoring 依赖装不上**（pesq / s3prl / dnsmos 编译失败）：不影响训练，只影响 stage 6 打分，可单独再装。
- **torch 装成了 CPU 版**：`pip uninstall torch torchaudio` 后用 `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` 重装。
- **数据以后扩到 train-360**：体积变大，挪到 `/share/workspace`（HDD，几十 T），改路径前缀即可。
