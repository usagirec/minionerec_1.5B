# MiniOneRec 复现（Qwen2.5-1.5B · 单卡）

本仓库是对生成式推荐框架 **MiniOneRec**（语义 ID / Semantic ID + SFT + GRPO 强化学习）的一次复现，
在**Autodl租用单卡A800，80BG**环境（GPU训练约27h）下用 **Qwen2.5-1.5B 全参微调**跑通完整流水线：

```
文本嵌入 → SID 构建 → 数据转换 → SFT → RL(GRPO) → 评测
```

原仓库：<https://github.com/AkaliKong/MiniOneRec>　论文：arXiv:2510.24431

---

## 一、训练配置

### 1.1 SFT 阶段（约五小时）

| 参数 | 值 |
|------|-----|
| batch_size | 64 |
| micro_batch_size | 8 |
| gradient_accumulation_steps | 8 |
| learning_rate | 3e-4 |
| num_epochs | 5 |
| cutoff_len | 512 |
| optimizer | AdamW |
| precision | bf16 |
| scheduler | cosine with warmup |
| early_stopping_patience | 3 |
| freeze_LLM | False |
| GPU 数量 | 1 |

训练任务组成：
- SidSFTDataset: 用户历史 SID 序列 -> 下一个 SID（主任务）
- SidItemFeatDataset: 商品标题/描述 <-> SID 对齐（辅助任务）
- FusionSeqRecDataset: 混合文本+SID 序列推荐（辅助任务）
- 
### 1.2 RL 阶段（GRPO）（约20小时）

| 参数 | 值 |
|------|-----|
| train_batch_size | 64 |
| eval_batch_size | 64 |
| gradient_accumulation_steps | 2 |
| learning_rate | 1e-5 |
| num_train_epochs | 2 |
| num_generations | 16 |
| beta (KL penalty) | 0.01 |
| reward_type | ranking |
| beam_search | True |
| temperature | 1.0 |
| sync_ref_model | True |
| DeepSpeed | ZeRO Stage 2 |
| GPU 数量 | 1 |


## 二、环境

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r requirements.txt
```
| 项目 | 配置 |
|------|------|
| GPU | NVIDIA A800 80GB PCIe |
| 驱动 | NVIDIA-SMI 580.105.08 |
| CUDA | 13.0 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.57.1 |
| TRL | 0.24.0 |
| DeepSpeed | 0.18.0 |
| 平台 | AutoDL |

> 原始 Amazon 数据来自 UCSD；本仓库直接使用处理后的中间产物，不重新下载原始评论。

## 三、复现步骤

### Step 1 · 文本嵌入（可选，官方已提供 `.npy`）

用嵌入模型把 item 文本（标题 + 描述）转成向量 `*.emb-qwen-td.npy`。

```bash
cd rq/text2emb
# 修改 amazon_text2emb.sh 里的 --plm_checkpoint 为你的嵌入模型路径
bash amazon_text2emb.sh
```

### Step 2 · SID 构建（可选）


```bash
cd rq
bash rqvae.sh                 # 训练量化器
python generate_indices.py    # 产出 data/Amazon/index/*.index.json
```

### Step 3 · 数据转换

把交互序列 + SID 拼成 SFT/RL 所需的训练样本。

```bash
bash convert_dataset.sh
```

### Step 4 · SFT

单卡全参微调（由官方 8 卡脚本改造），带早停（EarlyStopping，patience=3，按 eval_loss 选最优）。

```bash
bash sft.sh
# 产物：output/sft/Industrial_and_Scientific/
```

### Step 5 · RL（GRPO）

在 SFT checkpoint 上做偏好优化：每个 prompt 采样 16 个候选，规则奖励 + 排序感知奖励，
组内归一得到优势并加 KL 约束防止偏离 SFT；采样/解码全程使用**约束束搜索**，保证候选均为合法 SID。

```bash
bash rl.sh
# 产物：output/rl/Industrial_and_Scientific/
```

### Step 6 · 评测

对某个模型做 beam search（约束解码），计算 HR@K / NDCG@K。

```bash
# 编辑 evaluate.sh：exp_name 填成待评模型路径（SFT 或 RL 的 checkpoint）
# 单卡需把脚本里两处 GPU 列表改为单卡：
#   split.py 的 --cuda_list "0,1,...,7" -> "0"
#   cudalist="0 1 ... 7"               -> cudalist="0"
bash evaluate.sh
# 指标由 calc.py 输出 NDCG / HR @ [1,3,5,10,20,50]
# 注意：日志中 CC 必须为 0（否则约束解码失效，模型生成了不存在的商品）
```

## 四、混合ID对比实验

### 混合语义 ID（文本嵌入 + 协同嵌入拼接）

**动机**：原始 SID 只由商品文本（标题 + 描述）编码而来，表达的是「这个商品是什么」，
缺少推荐系统最核心的**协同信号**——「这个商品和哪些商品一起被消费」。
因此语义相近但用户群体完全不同的商品，可能在 SID 空间里难以区分。

**做法**：用训练好的 SASRec 取出它的 item embedding（协同过滤表示），
与文本嵌入各自做 L2 归一化后**拼接**，一起送入量化器构建混合语义 ID；
数据转换与后续 SFT / RL 流程完全不变，从而保证与基线 SID **同协议可比**。

**结果**：混合 SID 在主观的「可分性」上确实更优，但**下游指标没有提升**：

| SID 方案 | HR@10 | NDCG@10 |
|---|---|---|
| 基线 SID + SFT+RL | **0.1398** | **0.0979** |
| 混合 SID + SFT+RL | 0.1381 | 0.0963 |

**结论**：语义 ID 的「可分性」是推荐效果的**必要而非充分条件**——
决定性因素是它是否编码了对用户偏好有用的结构。该负结果直接影响了后续优先级排序
（先补齐模型规模与偏好任务，再回头优化 SID）。

## 五、结论

数据集 **Industrial_and_Scientific**，每条样本单正例，故 HR@K = Recall@K。
下表把本复现（Qwen2.5-**1.5B**，单卡）插入论文 Table 1，对比全部 baseline，并标出与作者原版的差异。

| 类别 | 方法 | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| Traditional | GRU4Rec | 0.0638 | 0.0542 | 0.0774 | 0.0598 | 0.0999 | 0.0669 |
| | Caser | 0.0618 | 0.0514 | 0.0717 | 0.0555 | 0.0942 | 0.0628 |
| | SASRec | 0.0790 | 0.0700 | 0.0909 | 0.0748 | 0.1088 | 0.0806 |
| Generative | HSTU | 0.0927 | 0.0885 | 0.1037 | 0.0918 | 0.1163 | 0.0958 |
| | TIGER | 0.0852 | 0.0742 | 0.1010 | 0.0807 | 0.1321 | 0.0908 |
| | LCRec | 0.0915 | 0.0805 | 0.1057 | 0.0862 | 0.1332 | 0.0952 |
| LLM-based | BIGRec | 0.0931 | 0.0841 | 0.1092 | 0.0907 | 0.1370 | 0.0997 |
| | D3 | 0.1024 | 0.0991 | 0.1213 | 0.0989 | 0.1500 | 0.1082 |
| | S-DPO | 0.1032 | 0.0906 | 0.1238 | 0.0991 | 0.1524 | 0.1082 |
| **作者原版** | **MiniOneRec (7B)** | **0.1143** | **0.1011** | **0.1321** | **0.1084** | **0.1586** | **0.1167** |
| 本复现 (1.5B) | SFT | 0.0887 | 0.0770 | 0.1076 | 0.0845 | 0.1369 | 0.0941 |
| **本复现 (1.5B)** | **SFT+RL** | **0.0931** | **0.0811** | **0.1109** | **0.0886** | **0.1398** | **0.0979** |
| — | *Δ (RL vs 7B)* | -18.5% | -19.8% | -16.1% | -18.3% | **-11.9%** | -16.1% |
| — | *达成比例* | 81% | 80% | 84% | 82% | **88%** | 84% |

1. **复现成立** —— 1.5B 的 SFT+RL 在 HR@10 上超过传统强基线 SASRec（0.1398 vs 0.1088，**+28.5%**），
   并超过 HSTU / TIGER / LCRec / BIGRec，落在 D3 / S-DPO 与作者 7B 之间。
2. **RL 增益方向与论文一致** —— 6 个指标相对 SFT **全部上升**，且提升集中在 top 名次
   （NDCG@3 +5.3%、NDCG@5 +4.9%），与 rank-aware reward 的设计意图吻合。
3. **与作者 7B 差距约 12%–20%**（达成 80%–88%），且越往大 K 差距越小；
   差距主要来自**模型规模（1.5B vs 7B）**与**未做用户偏好/thinking 任务**（原始评论数据不可得）。
4. **混合语义 ID 的改进实验为负** —— 把 SASRec 协同嵌入与文本嵌入拼接后，
   下游指标与基线基本持平（HR@10 0.1381 对 0.1398），
   说明语义 ID 的可分性是**必要而非充分条件**。



## 七、后续计划（TODO）

- [ ] **P1 · 完成 7B 多卡复现** —— 直接闭合与作者 7B 的差距，pipeline 已就绪，仅差多卡资源。
- [ ] **P2 · 继续优化混合语义 ID** —— 当前版本把协同嵌入与文本嵌入等权拼接，协同信号可能被过度放大；
      下一步给协同分支加可调缩放系数做网格搜索，并配合碰撞去重后处理重新评估下游。
- [ ] **P3 · 在更多数据集上验证（Office_Products、Amazon23）。
- [ ] **P4 · 探索不同超参数。

## 八、致谢

复现基于原作者仓库 [AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)，
并参考论文 *MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation*。
