# %% [markdown]
# # WebDev RL · 阶段 1：LoRA GRPO 训练
#
# 在单张 A100 80GB 上，用 TRL 的 GRPO + LoRA 训练 **MiMo-V2.6-Distill-Qwen-9B** 做单轮网页生成。
# 奖励和评测 notebook 完全相同：截图 → 运行检查 → query 判官 + 组内挑选 → 官方 group_v1 公式。
#
# **和官方配置的对应**：
#
# | 项 | 官方 9B webdev | 这里 |
# |---|---|---|
# | 训练方式 | 全参数，64 卡 | LoRA r=64，单卡 |
# | 每步 | 32 题 × 8 份 | 8 题 × 8 份 |
# | 上下文 | 256K，多轮 agent | 16K，单轮 |
# | 学习率 | 1e-6 | 1e-5（LoRA 一般要高约 10 倍） |
# | KL / std 归一化 | 不用 / 不用 | 不用 / 不用 |
# | 采样 | T=0.6, top_p=0.95, top_k=20 | 相同 |
#
# **断点续训**：每 5 步存一次 checkpoint，重新运行本 notebook 会自动从最新的 checkpoint 接着训。
# 如果在浏览器里挂载了 Google Drive，checkpoint 会同时复制到 Drive。

# %%
import gc, json, os, shutil, subprocess, sys, time
from pathlib import Path

ON_VM = os.path.isdir("/content")
SELFTEST_ONLY = os.environ.get("SELFTEST_ONLY") == "1"   # 本地测试：只跑奖励函数自检


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-3000:])
    return r.returncode


if not SELFTEST_ONLY:
    # 用 nvidia-smi 查，不在 vLLM 启动前初始化 CUDA
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    assert _gpu, "没有 GPU：运行时请选 A100"
    print(_gpu)
    assert int(_gpu.split(",")[-1]) > 70000, "需要 80GB 显存：选 A100 并打开「高 RAM」"

# %% [markdown]
# ## 安装依赖（和评测 notebook 同一套版本）
# 如果 Drive 里有 `webdev_cache/pkgs.tar.zst`，直接解压，不用现场安装。

# %%
PINS = ["vllm==0.30.0", "transformers==5.18.0", "peft==0.21.2", "trl==1.14.2", "accelerate==1.15.0",
        "flash-linear-attention", "openai", "pandas", "pyarrow", "pillow", "playwright", "huggingface_hub", "datasets"]
PKGS = Path("/content/pkgs")                                        # 依赖装在这里，可整体打包
CACHE_TAR = Path("/content/drive/MyDrive/webdev_cache/pkgs.tar.zst")  # Drive 里的缓存


def use_pkgs():
    """把 /content/pkgs 放到最前面；写进环境变量，vLLM 的子进程也能找到。"""
    if str(PKGS) not in sys.path:
        sys.path.insert(0, str(PKGS))
    if str(PKGS) not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        os.environ["PYTHONPATH"] = str(PKGS) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(PKGS / "ms-playwright")


if ON_VM:
    t0 = time.time()
    if not PKGS.exists() and CACHE_TAR.exists():
        sh("which zstd >/dev/null || apt-get install -y -q zstd >/dev/null")
        sh(f"tar -I 'zstd -T0' -xf '{CACHE_TAR}' -C /content")
        print(f"从 Drive 缓存解压依赖，用时 {time.time() - t0:.0f}s")
    if not PKGS.exists():
        print("没有找到 Drive 缓存，现场安装（约 6 分钟）")
        sh(f"{sys.executable} -m pip install -q --target {PKGS} " + " ".join(f"'{p}'" for p in PINS))
        use_pkgs()
        sh(f"{sys.executable} -m playwright install chromium")
    use_pkgs()
    if not Path("/content/.pw_deps_ok").exists():
        sh(f"{sys.executable} -m playwright install-deps chromium >/dev/null 2>&1")
        Path("/content/.pw_deps_ok").touch()
    try:
        import vllm
        print(f"依赖就绪：vllm {vllm.__version__}，用时 {time.time() - t0:.0f}s")
    except Exception as e:  # CPU 机器上 vLLM 可能导入失败，不影响自检
        print("依赖就绪（vllm 未导入：", type(e).__name__, "）")

# %% [markdown]
# ## 配置

# %%
MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"

PROMPTS_PER_STEP = int(os.environ.get("PROMPTS_PER_STEP", 8))   # 每步几道题
NUM_GENERATIONS = 8           # 每题几份（GRPO 组大小）
MAX_STEPS = int(os.environ.get("MAX_STEPS", 60))
LEARNING_RATE = 1e-5
LORA_R, LORA_ALPHA = 64, 64
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",          # 8 层全注意力
                  "in_proj_qkv", "in_proj_z", "out_proj",          # 24 层线性注意力
                  "gate_proj", "up_proj", "down_proj"]             # MLP；不碰视觉塔
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", 12288))
VLLM_MAX_LEN = 16384
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)
SAVE_EVERY = 5
KEEP_SHOTS_EVERY = 10         # 每 10 步保留一次截图，其余删掉省磁盘

JUDGE = dict(judge_base_url=os.environ.get("JUDGE_BASE_URL", "https://openoneapi.com/v1"),
             judge_model=os.environ.get("JUDGE_MODEL", "deepseek-v4.1-flash"), judge_concurrency=8)

WORKDIR = Path(os.environ.get("WORKDIR") or ("/content/webdev_train" if ON_VM else "webdev_train_out"))
OUTPUT_DIR = WORKDIR / "checkpoints"
DRIVE_DIR = Path("/content/drive/MyDrive/webdev_lora")   # 挂载了 Drive 才会用
WORKDIR.mkdir(parents=True, exist_ok=True)

# %%
# __LIBRARY__

# %%
wc.configure(**JUDGE)
if not wc.CFG["mock_judge"]:
    assert wc.judge_api_key(), "找不到判官 key：在 Colab 密钥里添加 JUDGE_API_KEY"
VAL, TRAIN = wc.load_tasks()
print(f"训练集 {len(TRAIN)} 道（验证集 {len(VAL)} 道留给评测 notebook，不参与训练）")

# %% [markdown]
# ## 奖励函数
# TRL 每步把 8 题 × 8 份 = 64 份回答一起交给它。按题分组，渲染、判官打分，算出每份的奖励。
# 渲染或判官失败的样本按官方做法填组内均值（不贡献梯度）。

# %%
STEP_LOG = WORKDIR / "steps.jsonl"


def _completion_text(comp):
    """TRL 给的回答可能是纯文本、消息列表，或者把思考单独放在 reasoning 字段里。"""
    if isinstance(comp, str):
        return comp
    msg = comp[-1] if isinstance(comp, list) else comp
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or msg.get("thinking")
    return f"{reasoning}\n</think>\n{text}" if reasoning else text


def webdev_reward(prompts, completions, completion_ids=None, task_idx=None, query=None,
                  trainer_state=None, log_metric=None, **kwargs):
    step = trainer_state.global_step if trainer_state is not None else -1
    t0 = time.time()
    group_of, queries, records = {}, [], []
    for i, comp in enumerate(completions):
        g = group_of.get(task_idx[i])
        if g is None:
            g = group_of[task_idx[i]] = len(queries)
            queries.append(query[i])
        k = sum(r["task"] == g for r in records)
        records.append({"task": g, "k": k, "html": wc.extract_html(_completion_text(comp)),
                        "gen_tokens": len(completion_ids[i]) if completion_ids is not None else 0})
    sites = WORKDIR / "sites" / f"step_{step:05d}"
    shutil.rmtree(sites, ignore_errors=True)
    wc.render(records, sites)
    wc.grade(records, queries)
    rewards = wc.fill_invalid_with_group_mean(records)

    s = wc.summarize(records)
    n = max(1, len(records))
    metrics = {
        "webdev/reward_raw_mean": s["mean_reward"] or 0.0,
        "webdev/query_score": s["mean_query_score"] or 0.0,
        "webdev/group_std": s["mean_group_std"] or 0.0,
        "webdev/flat_group_frac": s["flat_groups"] / max(1, s["groups"]),
        "webdev/no_delivery_frac": s["status"].get("no_delivery", 0) / n,
        "webdev/runtime_fail_frac": s["status"].get("runtime_fail", 0) / n,
        "webdev/invalid_frac": (s["status"].get("render_fail", 0) + s["status"].get("judge_fail", 0)) / n,
        "webdev/reward_seconds": time.time() - t0,
    }
    if log_metric is not None:
        for name, v in metrics.items():
            log_metric(name, float(v))
    with open(STEP_LOG, "a") as f:
        f.write(json.dumps({"step": step, "time": time.time(), **metrics, "judge_usage": dict(wc.USAGE),
                            "rows": [{k: r.get(k) for k in ("task", "k", "status", "reward", "query_score", "pick_norm",
                                                             "gen_tokens", "runtime_fail")} | {"query": queries[r["task"]][:200]}
                                     for r in records]}, ensure_ascii=False) + "\n")
    if step % KEEP_SHOTS_EVERY != 0:
        shutil.rmtree(sites, ignore_errors=True)
    print(f"[step {step}] reward={metrics['webdev/reward_raw_mean']:.3f} query={metrics['webdev/query_score']:.2f} "
          f"group_std={metrics['webdev/group_std']:.3f} no_delivery={metrics['webdev/no_delivery_frac']:.0%} "
          f"({metrics['webdev/reward_seconds']:.0f}s)", flush=True)
    return rewards

# %% [markdown]
# ## 奖励函数自检（不需要 GPU，约 1 分钟）
# 一组 4 份手写回答：像样的页面、简陋页面、JS 报错页面、没交付。期望：前两份正常打分且第一份更高，报错页 0 分，没交付 0 分。

# %%
_SAMPLE_GOOD = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Study Bites</title>
<style>body{margin:0;font-family:Georgia,serif;background:#fffbe6;color:#222}header{padding:80px 40px;background:linear-gradient(120deg,#ff6b35,#f7c59f)}
h1{font-size:64px;margin:0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;padding:40px}
.card{background:#fff;border:4px solid #222;padding:24px;box-shadow:8px 8px 0 #222}</style></head>
<body><header><h1>Study Bites</h1><p>Group lessons served fresh.</p></header><section class="grid">
<div class="card"><h2>Math</h2><p>Algebra platter</p></div><div class="card"><h2>Physics</h2><p>Newton combo</p></div>
<div class="card"><h2>History</h2><p>Empire tasting</p></div></section></body></html>"""
_SAMPLE_PLAIN = "<!DOCTYPE html><html><body><h1>Website</h1><p>Some text.</p></body></html>"
_SAMPLE_BROKEN = "<!DOCTYPE html><html><body><h1>Study Bites</h1><script>window.onload=()=>{undefinedFunction()}</script></body></html>"
_q = "Build a brutalist-style static website for a tutoring cafe called Study Bites with subject menu cards for Math, Physics and History."
_selftest = webdev_reward(
    prompts=[None] * 4,
    completions=[[{"role": "assistant", "content": f"```html\n{_SAMPLE_GOOD}\n```"}],
                 [{"role": "assistant", "content": f"```html\n{_SAMPLE_PLAIN}\n```", "reasoning_content": "plan"}],
                 f"<think>x</think>```html\n{_SAMPLE_BROKEN}\n```",
                 [{"role": "assistant", "content": "I cannot build that."}]],
    task_idx=[-1] * 4, query=[_q] * 4, trainer_state=None)
print("自检奖励：", _selftest)
assert _selftest[2] == 0.0 and _selftest[3] == 0.0, "报错页和没交付应为 0"
assert _selftest[0] >= _selftest[1], "像样的页面应不低于简陋页面"
shutil.rmtree(WORKDIR / "sites", ignore_errors=True)
STEP_LOG.unlink(missing_ok=True)
if SELFTEST_ONLY:
    raise SystemExit(0)

# %% [markdown]
# ## 模型、LoRA、训练器
# `lora_sync.install()` 修补了 TRL 的权重同步：TRL 原本每步在 bf16 基座上原地合并再拆开 LoRA，
# 来回取整会让冻结的基座权重慢慢漂移（CPU 实测 100 次后 0.4% 的权重变了）。修补后在 fp32 里另算合并结果，基座保持不变。

# %%
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

lora_sync.install()

train_ds = Dataset.from_list([{"prompt": wc.build_messages(t["query"]), "task_idx": i, "query": t["query"]}
                              for i, t in enumerate(TRAIN)])

model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

args = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE, lr_scheduler_type="constant", warmup_steps=0,
    per_device_train_batch_size=1, gradient_accumulation_steps=PROMPTS_PER_STEP * NUM_GENERATIONS,
    num_generations=NUM_GENERATIONS, max_completion_length=MAX_COMPLETION, **SAMPLING,
    beta=0.0, epsilon=0.2, loss_type="dapo", scale_rewards="none", mask_truncated_completions=False,
    use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.55, vllm_enable_sleep_mode=True,
    vllm_max_model_length=VLLM_MAX_LEN,
    bf16=True, gradient_checkpointing=True,
    logging_steps=1, save_steps=SAVE_EVERY, save_total_limit=3, report_to="none", seed=0, log_completions=False,
)
peft_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=TARGET_MODULES,
                         task_type="CAUSAL_LM")


class CopyToDrive(TrainerCallback):
    """挂载了 Drive 时，每次存 checkpoint 后复制一份过去。"""

    def on_save(self, args, state, control, **kw):
        if DRIVE_DIR.parent.exists():
            src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if src.exists():
                shutil.copytree(src, DRIVE_DIR / src.name, dirs_exist_ok=True)
                shutil.copy(STEP_LOG, DRIVE_DIR / "steps.jsonl")


trainer = GRPOTrainer(model=model, reward_funcs=webdev_reward, args=args, train_dataset=train_ds,
                      processing_class=tokenizer, peft_config=peft_config, callbacks=[CopyToDrive()])
trainer.model.print_trainable_parameters()

# %% [markdown]
# ## 训练
# 每步大约 12–18 分钟（生成 + 截图 + 判官 + 训练）。日志里每步打印一行 `[step N] reward=…`。

# %%
last = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None
if last is None and (DRIVE_DIR / "steps.jsonl").exists():
    ckpts = sorted(DRIVE_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        last = str(shutil.copytree(ckpts[-1], OUTPUT_DIR / ckpts[-1].name, dirs_exist_ok=True))
print("从 checkpoint 继续：" if last else "从头开始训练", last or "")
trainer.train(resume_from_checkpoint=last)

final = WORKDIR / "final_adapter"
trainer.save_model(str(final))
print("LoRA adapter 已保存：", final)
if DRIVE_DIR.parent.exists():
    shutil.copytree(final, DRIVE_DIR / "final_adapter", dirs_exist_ok=True)

# %% [markdown]
# ## 训练曲线

# %%
import statistics as st

logs = [json.loads(l) for l in STEP_LOG.read_text().splitlines()]
print(f"{'step':>5} {'reward':>7} {'query':>6} {'std':>6} {'no_del':>7} {'rt_fail':>7}")
for l in logs:
    print(f"{l['step']:>5} {l['webdev/reward_raw_mean']:>7.3f} {l['webdev/query_score']:>6.2f} {l['webdev/group_std']:>6.3f} "
          f"{l['webdev/no_delivery_frac']:>7.0%} {l['webdev/runtime_fail_frac']:>7.0%}")
print("\n下一步：用评测 notebook，把 ADAPTER_DIR 设为", final, "，在验证集上对比训练前后。")
