# %% [markdown]
# # WebDev RL · 评测（阶段 0 基线 / 训练前后对比）
#
# 用 **MiMo-V2.6-Distill-Qwen-9B** 对官方 webdev 题目单轮生成网页，截图，用视觉判官按官方组内奖励打分。
#
# - **不填 `ADAPTER_DIR`**：阶段 0 基线。每题生成 8 份，看组内分数差异够不够 GRPO 学习。
# - **填了 `ADAPTER_DIR`**：训练前后对比。每题基座生成 4 份、训练后生成 4 份，放进同一组让判官挑，直接比较谁更好。
#
# 运行时：**A100 GPU + 高 RAM**。判官 key 放在 Colab 密钥 `JUDGE_API_KEY`（或 VM 上的 `/content/.judge_key`）。
# 判官提示词和奖励公式移植自 [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) `mimo-oss` 分支（Apache-2.0）。

# %%
import gc, json, os, shutil, subprocess, sys, time
from pathlib import Path

ON_VM = os.path.isdir("/content")
DRY_RUN_HTML_DIR = os.environ.get("DRY_RUN_HTML_DIR")  # 本地测试用：跳过 GPU 生成


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-3000:])
    return r.returncode


if not DRY_RUN_HTML_DIR:
    import torch
    assert torch.cuda.is_available(), "没有 GPU：运行时请选 A100"
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(torch.cuda.get_device_name(0), f"{mem:.0f} GiB")
    assert mem > 70, "需要 80GB 显存：选 A100 并打开「高 RAM」"

# %% [markdown]
# ## 安装依赖
# 版本都选的是发布超过两周、互相兼容的：vLLM 0.30.0 支持 Qwen3.5 架构，且在 TRL 1.14.2 支持的范围内（训练 notebook 用同一套）。

# %%
PINS = ["vllm==0.30.0", "transformers==5.18.0", "peft==0.21.2", "trl==1.14.2", "accelerate==1.15.0",
        "flash-linear-attention", "openai", "pandas", "pyarrow", "pillow", "playwright", "huggingface_hub", "datasets"]
if ON_VM and not DRY_RUN_HTML_DIR:
    try:
        import vllm
        assert vllm.__version__ == "0.30.0"
        print("依赖已安装")
    except Exception:
        sh(f"{sys.executable} -m pip install -q " + " ".join(f"'{p}'" for p in PINS))
        sh("playwright install --with-deps chromium")
        print("安装完成。如果后面 import 报错，重启会话后从下一格继续。")

# %% [markdown]
# ## 配置

# %%
MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
ADAPTER_DIR = os.environ.get("ADAPTER_DIR") or None   # 训练后的 LoRA 目录；留空 = 只评基座

N_TASKS = int(os.environ.get("N_TASKS", 16))   # 从留出的 100 道验证题里取前 N 道
N_SAMPLES = 8                                  # 每组 8 份；对比模式下基座 4 + 训练后 4
MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 12288
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)  # 官方 webdev 配置

JUDGE = dict(judge_base_url=os.environ.get("JUDGE_BASE_URL", "https://openoneapi.com/v1"),
             judge_model=os.environ.get("JUDGE_MODEL", "kimi-k3"), judge_concurrency=8)

WORKDIR = Path(os.environ.get("WORKDIR") or ("/content/webdev_eval" if ON_VM else "webdev_eval_out"))
WORKDIR.mkdir(parents=True, exist_ok=True)

# %%
# __LIBRARY__

# %%
wc.configure(**JUDGE)
if not wc.CFG["mock_judge"]:
    assert wc.judge_api_key(), "找不到判官 key：在 Colab 密钥里添加 JUDGE_API_KEY"
VAL, TRAIN = wc.load_tasks()
tasks = VAL[:N_TASKS]
print(f"验证集 {len(VAL)} 道，本次评测 {len(tasks)} 道；训练集 {len(TRAIN)} 道")

# %% [markdown]
# ## 生成

# %%
def generate(model_path, queries, n, seed=0):
    """vLLM 生成：每个 query n 份。返回 [(qi, text, n_tokens, finish_reason)]。"""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.85, seed=seed,
              limit_mm_per_prompt={"image": 0, "video": 0})
    t0 = time.time()
    outs = llm.chat([wc.build_messages(q) for q in queries], SamplingParams(n=n, max_tokens=MAX_NEW_TOKENS, **SAMPLING))
    print(f"{model_path}: 生成 {len(queries)}×{n} 用时 {time.time() - t0:.0f}s")
    rows = [(qi, c.text, len(c.token_ids), c.finish_reason) for qi, o in enumerate(outs) for c in o.outputs]
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    return rows


def merge_adapter(base_id, adapter_dir, out_dir):
    """把 LoRA 并进基座另存一份，供 vLLM 加载。增量在 fp32 里算，只在最后转一次 bf16。"""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForImageTextToText

    out_dir = Path(out_dir)
    if (out_dir / "model.safetensors.index.json").exists():
        return str(out_dir)
    snapshot_download(base_id, local_dir=out_dir, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"])
    base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_dir)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoraLayer):
                for a in m.active_adapters:
                    if a in m.lora_A:
                        w = m.base_layer.weight
                        delta = (m.lora_B[a].weight.float() @ m.lora_A[a].weight.float()) * m.scaling[a]
                        w.data = (w.data.float() + delta).to(w.dtype)
    model.unload().save_pretrained(out_dir, safe_serialization=True)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return str(out_dir)


queries = [t["query"] for t in tasks]
records = []
if DRY_RUN_HTML_DIR:
    fixtures = sorted(Path(DRY_RUN_HTML_DIR).glob("*.html"))
    for ti in range(len(tasks)):
        for k in range(N_SAMPLES):
            f = fixtures[(ti + k) % len(fixtures)]
            text = "```html\n" + f.read_text() + "\n```" if f.stem != "nodelivery" else "sorry"
            src = "adapter" if (ADAPTER_DIR and k >= N_SAMPLES // 2) else "base"
            records.append({"task": ti, "k": k, "source": src, "text": text, "gen_tokens": len(text) // 4, "finish": "stop"})
else:
    plan = [("base", MODEL, N_SAMPLES // 2 if ADAPTER_DIR else N_SAMPLES)]
    if ADAPTER_DIR:
        plan.append(("adapter", merge_adapter(MODEL, ADAPTER_DIR, WORKDIR / "merged"), N_SAMPLES // 2))
    offset = {}
    for src, path, n in plan:
        for qi, text, ntok, fin in generate(path, queries, n):
            k = offset.get(qi, 0)
            offset[qi] = k + 1
            records.append({"task": qi, "k": k, "source": src, "text": text, "gen_tokens": ntok, "finish": fin})

for r in records:
    r["html"] = wc.extract_html(r["text"])
print(f"{len(records)} 份：交付 {sum(bool(r['html']) for r in records)}，截断 {sum(r['finish'] == 'length' for r in records)}")

# %% [markdown]
# ## 截图 + 判官 + 奖励

# %%
sites = WORKDIR / "sites"
shutil.rmtree(sites, ignore_errors=True)
t0 = time.time()
wc.render(records, sites)
print(f"截图 {sum('shot' in r for r in records)}，运行检查失败 {sum('runtime_fail' in r for r in records)}，"
      f"渲染失败 {sum('render_fail' in r for r in records)}，用时 {time.time() - t0:.0f}s")
t0 = time.time()
wc.grade(records, queries)
print(f"判官用时 {time.time() - t0:.0f}s，调用 {wc.USAGE['calls']} 次，失败 {wc.USAGE['failures']} 次")

# %% [markdown]
# ## 报告
# 组内标准差是关键：GRPO 只从同组差异里学习，一组 8 份分数几乎相同（标准差 < 0.05）就学不到东西。
# 对比模式下看 `head_to_head`：同组里训练后的样本被判官选为更好（pick 为正）的程度。

# %%
import statistics as st

summary = wc.summarize(records)
summary.update(model=MODEL, adapter=ADAPTER_DIR, judge_model=wc.CFG["judge_model"], n_tasks=len(tasks))
if ADAPTER_DIR:
    for src in ("base", "adapter"):
        rs = [r for r in records if r["source"] == src]
        q = [r["query_score"] for r in rs if r.get("query_score") is not None]
        p = [r["pick_norm"] for r in rs if r.get("pick_norm") is not None]
        summary[f"{src}_query_score"] = round(st.mean(q), 4) if q else None
        summary[f"{src}_pick_norm"] = round(st.mean(p), 4) if p else None
        summary[f"{src}_no_delivery_or_runtime_fail"] = sum(r["status"] in ("no_delivery", "runtime_fail") for r in rs)
    summary["head_to_head"] = (None if summary["adapter_pick_norm"] is None or summary["base_pick_norm"] is None
                               else round(summary["adapter_pick_norm"] - summary["base_pick_norm"], 4))
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not ADAPTER_DIR:
    ok = 1 - summary["flat_groups"] / max(1, summary["groups"])
    print("\n结论：", "组内差异足够，可以开始训练。" if ok >= 0.5 else "超过一半的组几乎没有差异，先排查判官或采样温度。")

rows = [{k: v for k, v in r.items() if k not in ("text", "html")} | {"query": queries[r["task"]]} for r in records]
(WORKDIR / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
print("已保存", WORKDIR / "results.json")

# %% [markdown]
# ## 看图：每题最高分和最低分

# %%
def gallery(n_tasks=4):
    html = []
    for ti in range(min(n_tasks, len(tasks))):
        rows = sorted([r for r in records if r["task"] == ti and r["reward"] is not None and "shot" in r], key=lambda r: r["reward"])
        if not rows:
            continue
        cells = []
        for tag, r in (("最高", rows[-1]), ("最低", rows[0])):
            b = wc.img_b64(r["shot"], 360)
            cells.append(f"<div style='flex:1;min-width:260px'><b>{tag} reward={r['reward']:.3f}</b> [{r['source']}] "
                         f"query={r.get('query_score')} pick={r.get('pick_norm', 0):+.2f}<br>"
                         f"<img src='data:image/jpeg;base64,{b}' style='width:100%;max-height:520px;object-fit:cover;object-position:top;border:1px solid #ccc'></div>")
        html.append(f"<h4>题 {ti}：{queries[ti][:120]}…</h4><div style='display:flex;gap:16px;flex-wrap:wrap'>{''.join(cells)}</div>")
    return "".join(html)


(WORKDIR / "gallery.html").write_text(gallery())
if "ipykernel" in sys.modules:
    from IPython.display import HTML, display
    display(HTML(gallery()))
else:
    print("画廊已保存", WORKDIR / "gallery.html")
