# 用 LoRA 对 MiMo-V2.6-Distill-Qwen-9B 做 RL 需要准备什么

> 依据：XiaomiMiMo/verl `mimo-oss` 分支中的 `docs/advance/ppo_lora.rst`、`examples/tuning/lora/run_qwen3_8b_fsdp.sh`、`verl/trainer/config/model/hf_model.yaml`，以及模型的 `config.json` 和权重索引。
> 下文的启动命令**没有实际跑过**，只是一份草稿。

LoRA RL 的训练流程和全参数 RL 完全一样：rollout → 打分 → 计算 GRPO 优势 → 更新参数。区别只在于**哪些参数参与训练**。所以要准备的东西和全参数 RL 基本相同，只是显存需求小得多。

## 1. 硬件

| 配置 | 能做什么 |
|---|---|
| 单卡 80GB（A100/H100） | 9B 模型 + 16K 上下文 + 每步 64 条轨迹，比较宽裕 |
| 单卡 40GB（A100-40） | 能跑，但要把 rollout 并发降到约 16 条，或把上下文降到 8K |
| 24GB 及以下 | 9B 模型的 bf16 权重就有 18.8GB，放不下。需要 QLoRA，或者换更小的模型 |
| CPU 内存 | 建议 ≥ 64GB（加载权重、Ray、数据处理） |
| 磁盘 | 模型 19GB + 环境 + checkpoint，建议 ≥ 100GB |

LoRA 部分的显存账（r=64，只加在语言模型的线性层上）：

| 模块 | 每层 LoRA 参数 | 层数 | 合计 |
|---|---|---|---|
| MLP：gate、up（4096→12288）、down（12288→4096） | 3.1M | 32 | 101M |
| 全注意力：q（4096→8192，含输出门）、k/v（→1024）、o | 2.0M | 8 | 16M |
| 线性注意力：in_proj_qkv（→8192）、in_proj_z（→4096）、out_proj | 1.8M | 24 | 44M |
| **合计** | | | **约 1.6 亿（占总参数 1.7%）** |

这部分参数加上它们的梯度和 Adam 状态，全部用 fp32 存，共 16 字节/参数 ≈ **2.6GB**。全参数训练对应的数字是 169GB。

## 2. 软件栈

- **训练框架**：verl。LoRA 走 FSDP/FSDP2 + HF PEFT 这条路径最成熟。Megatron 路径需要 Megatron-Bridge ≥ 0.5。
- **推理引擎**：vLLM 或 SGLang。
  - SGLang 目前要求 `lora.merge=True`：每步把 LoRA 合并进基座权重，再把完整的 18.8GB 权重同步给推理引擎。单卡上只需几秒。
  - vLLM 可以只同步 LoRA 增量，但它对 Qwen3.5 线性注意力层上的 LoRA 支持程度需要实测。稳妥的做法也是 `merge=True`。
- **模型依赖**：Qwen3.5 的混合线性注意力需要较新版本的 transformers、vLLM 或 SGLang，以及 flash-linear-attention 的 kernel。verl 里的 Qwen3.5 示例固定了 `vllm==0.18.0`。
- **其他**：Ray（verl 依赖它调度），以及 wandb 或 tensorboard 之类的监控工具。

## 3. 任务和奖励（最花功夫的部分）

至少需要以下三样：

1. **训练数据**：parquet 格式，每行包含 `prompt` 和打分所需的 `extra_info`。
2. **奖励函数**：输入模型输出，返回一个分数。
   - 最简单的是自定义 Python 函数，通过 `reward.custom_reward_function.path` 和 `.name` 指定。
   - 官方的五个任务都已经写好了，但其中四个依赖 K8s pod 或外部评委。在 Colab 上**只有 Music 任务（CPU 上用 `abc2midi` 按规则打分）可以直接用**。
3. **验证集**：用来观察奖励是否真的在上升。

选任务的经验法则：**SFT 模型在这个任务上的通过率最好在 20%–80% 之间**。如果一组 rollout 全对或全错，组内优势为 0，这一组就学不到任何东西。

## 4. 关键配置（单卡草稿，未测试）

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=$DATA/train.parquet data.val_files=$DATA/val.parquet \
  data.train_batch_size=8 data.max_prompt_length=4096 data.max_response_length=12288 \
  actor_rollout_ref.model.path=XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B \
  actor_rollout_ref.model.lora_rank=64 actor_rollout_ref.model.lora_alpha=64 \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_agg_mode=prompt-mean \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.rollout.name=sglang \
  +actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_training_steps=50 trainer.save_freq=5 trainer.test_freq=5
```

各项配置的理由：

- **`target_modules` 显式列出**：不用 `all-linear`。`all-linear` 会把 LoRA 也加到视觉塔上（`visual.blocks.*.attn.qkv`、`mlp.linear_fc*`）。纯文本任务用不到视觉塔，加上去既浪费，也会增加推理引擎出兼容问题的风险。上面列出的名字和视觉塔的模块名不重合。
- **学习率 1e-5**：官方全参数训练用 1e-6。LoRA 一般需要比全参数高约一个数量级的学习率（Thinking Machines 的《LoRA Without Regret》给出的经验约为 10 倍）。这只是起点，需要扫一下学习率。
- **`use_kl_loss=False` 和 `prompt-mean`**：与官方 9B RL 配置保持一致。顺带一提，LoRA 天然有一个零成本的参考模型：关掉 adapter 就是原始模型，所以如果以后要加 KL，也不需要额外的显存。
- **`entropy_from_logits_with_chunking`**：词表有 248K，16K 序列的完整 logits 用 fp32 存约 16GB，必须分块计算。
- **`gpu_memory_utilization=0.6`**：训练和推理共用一张卡。推理阶段 SGLang 占 60%，训练阶段它会把显存释放出来。

## 5. LoRA RL 特有的坑

1. **rank 不需要很大**：RL 每条轨迹提供的有效信息很少（本质上只有一个奖励标量），所以低 rank 通常就够了。r=16–64 都可以，r=64 主要是为了保险。
2. **lora_alpha 和学习率是耦合的**：缩放系数是 alpha/r，改 rank 时要一起调整学习率，或者固定 alpha/r 不变。
3. **合并误差**：用 `merge=True` 时，推理引擎拿到的是 bf16 合并后的权重，而训练端是"基座 + LoRA"分开计算的，两边算出的 logprob 会有细微差异（训推不一致）。需要监控 `rollout` 和 `actor` 两边 logprob 的差值，或者 importance ratio 的被裁剪比例。
4. **奖励全为 0 却看起来一切正常**：官方 Music 配置的注释特别提醒过，如果 `abc2midi` 没装，所有样本都会被跳过并记为 0 分，训练日志看上去完全正常。开训前要先冒烟测试奖励函数。
5. **Colab 会话会断**：`save_freq` 要设小，并确认 checkpoint 存到 Google Drive 之类的持久位置，否则断开后从头再来。

## 6. 最小起步路线

1. 在 Colab 上用 SGLang 加载 9B 模型，跑几条 Music 任务的 prompt，确认推理正常、输出格式正确。
2. 单独运行奖励函数，确认能打出非 0 分，并统计 SFT 模型的初始通过率和分数分布。
3. 用 batch 2 × 4、上下文 8K 跑 2–3 步，确认整个流程（rollout → 打分 → 更新 → 权重同步）能跑通，并记录峰值显存。
4. 放大到 batch 8 × 8、上下文 16K，跑 30–50 步，观察奖励曲线、熵、响应长度和 clip 比例。
