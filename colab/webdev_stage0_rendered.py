# %% [markdown]
# # WebDev RL · 阶段 0：基线评测（不训练）
#
# 用 **MiMo-V2.6-Distill-Qwen-9B** 对官方 webdev 题目单轮生成网页，截图，用视觉判官打分，按官方组内奖励公式算 reward。
# 目的：确认奖励管线跑通，并看组内分数差异够不够 GRPO 学习。
#
# **运行前：**
# 1. 运行时选 **G4 GPU（96GB）** 或 **A100 GPU（80GB，开高 RAM）**。T4 / L4 放不下这个模型。
# 2. 在左侧「密钥」里添加 `JUDGE_API_KEY`（视觉判官的 API key），并在下面的配置格里填 `JUDGE_BASE_URL` 和 `JUDGE_MODEL`。任何兼容 OpenAI 接口、支持图片输入的模型都可以。
# 3. 第一次运行安装格后，如果提示需要重启会话，点「重启会话」再从配置格往下运行。
#
# 判官提示词和奖励公式移植自 [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) `mimo-oss` 分支的 `recipes/design/grader_service`（Apache-2.0）。和官方的区别：单轮生成一个 HTML 文件（官方是多轮 agent），在本机渲染截图（官方是独立评分服务）。

# %%
import os, sys, json, re, io, base64, time, random, shutil, subprocess, asyncio, math
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules


def sh(cmd):
    """Run a shell command and print its output."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    return r.returncode


if IN_COLAB:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh("free -h | head -2")
    sh("df -h /content | tail -1")

# %% [markdown]
# ## 安装依赖
# vLLM 的版本需要支持 Qwen3.5（混合线性注意力）。verl 官方的 Qwen3.5 示例固定的是 `vllm==0.18.0`；如果安装或加载报错，改 `VLLM_SPEC` 换一个更新的版本。

# %%
VLLM_SPEC = "vllm==0.18.0"
if IN_COLAB:
    sh(f"pip install -q '{VLLM_SPEC}' openai pandas pyarrow pillow playwright huggingface_hub")
    sh("playwright install --with-deps chromium")

# %% [markdown]
# ## 配置

# %%
MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"

N_TASKS = 16          # 评测多少道题（从留出的 100 道验证题里取前 N 道）
N_SAMPLES = 8         # 每道题生成几份，对应 GRPO 的组大小
MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 12288
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)  # 官方 webdev 配置实测更好的采样参数

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")      # 必填：判官模型名，例如某个支持图片的模型
JUDGE_CONCURRENCY = 8
JUDGE_TEMPERATURE = 1.0                              # 官方两个判官都用 1.0
PICK_ROUNDS = 8                                      # 组内挑选轮数（官方 8 轮，Williams 拉丁方）

SHOT_W, SHOT_H = 1280, 800      # 截图视口
SHOT_MAX_H = 12000              # 整页截图最大高度（官方 12000）
JUDGE_IMG_W = 720               # 送给判官前缩放到的宽度（官方 720）

WORKDIR = Path("/content/webdev_stage0" if IN_COLAB else os.environ.get("WORKDIR", "webdev_stage0_out"))
WORKDIR.mkdir(parents=True, exist_ok=True)
SITES = WORKDIR / "sites"

# 本地测试用的开关（Colab 上保持 False）：跳过 vLLM，用现成的 HTML；用假判官
DRY_RUN_HTML_DIR = os.environ.get("DRY_RUN_HTML_DIR")
MOCK_JUDGE = os.environ.get("MOCK_JUDGE") == "1"


def _secret(name):
    if IN_COLAB:
        try:
            from google.colab import userdata
            return userdata.get(name)
        except Exception:
            pass
    return os.environ.get(name)


JUDGE_API_KEY = _secret("JUDGE_API_KEY")
if not MOCK_JUDGE:
    assert JUDGE_MODEL, "请先填 JUDGE_MODEL"
    assert JUDGE_API_KEY, "请在 Colab 密钥里添加 JUDGE_API_KEY"

# %% [markdown]
# ## 读取题目
# 官方数据集共 2093 道建站需求。固定随机种子打乱后，前 100 道留作验证集（以后训练时不用），这里从中取前 `N_TASKS` 道。

# %%
import pandas as pd
from huggingface_hub import hf_hub_download

parquet = hf_hub_download("XiaomiMiMo/MiMo-V2.6-RL-oss", "webdev.parquet", repo_type="dataset")
df = pd.read_parquet(parquet)
all_tasks = []
for _, row in df.iterrows():
    inst = json.loads(row["extra_info"]["instance_json"])
    all_tasks.append({"id": inst["instance_id"], "query": inst["problem_statement"].strip()})
random.Random(0).shuffle(all_tasks)
VAL_TASKS = all_tasks[:100]
tasks = VAL_TASKS[:N_TASKS]
print(f"共 {len(all_tasks)} 道题，验证集 100 道，本次评测 {len(tasks)} 道 × {N_SAMPLES} 份")
print("示例：", tasks[0]["query"][:200])

# %% [markdown]
# ## 提示词
# 把官方 webdev agent 的系统提示改写成单轮版本：一次输出一个完整的 `index.html`。

# %%
SYSTEM_PROMPT = """You are an expert front-end engineer and web designer.

Build the website the user asks for and deliver it as ONE complete, self-contained HTML file.
For any request, always produce a working website; just build it, don't ask the user to confirm details.

Output rules:
- Reply with exactly one fenced code block: ```html ... ``` containing the whole file. No other files exist.
- Start the file with <!DOCTYPE html>.
- Styling: Tailwind CSS via CDN — <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script> — and/or an inline <style> tag.
- JavaScript, if any, goes in inline <script> tags. If the request asks for React, load React, ReactDOM and Babel standalone from a CDN inside the same file.
- Images may be referenced by absolute https URLs. Use mock content only; no backend.
- Write plain HTML/CSS/JavaScript with normal angle brackets, not HTML entities.

Make it complete, functional and visually polished."""


def build_messages(task):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Build a website for the following request:\n\n{task['query']}"},
    ]


def extract_html(text):
    """Pull the delivered HTML out of a model reply; None means no delivery."""
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = [m.group(2) for m in re.finditer(r"```([\w+-]*)[ \t]*\n(.*?)```", text, flags=re.S)
              if m.group(1).lower() in ("", "html")]
    blocks = [b for b in blocks if re.search(r"<(html|body|!doctype)", b, flags=re.I)]
    if blocks:
        return max(blocks, key=len).strip()
    m = re.search(r"(<!doctype html.*?</html>|<html.*?</html>)", text, flags=re.S | re.I)
    return m.group(1).strip() if m else None

# %% [markdown]
# ## 生成（vLLM）
# 每道题采样 `N_SAMPLES` 份。`finish_reason == "length"` 表示生成到上限被截断。

# %%
records = []  # one dict per (task, sample)
if SITES.exists():
    shutil.rmtree(SITES)

if DRY_RUN_HTML_DIR:
    fixtures = sorted(Path(DRY_RUN_HTML_DIR).glob("*.html"))
    for ti, task in enumerate(tasks):
        for k in range(N_SAMPLES):
            f = fixtures[(ti + k) % len(fixtures)]
            text = "```html\n" + f.read_text() + "\n```" if f.stem != "nodelivery" else "sorry, no code"
            records.append({"task": ti, "k": k, "text": text, "gen_tokens": len(text) // 4, "finish": "stop"})
else:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0, "video": 0},  # 纯文本任务，不给视觉塔预留显存
    )
    params = SamplingParams(n=N_SAMPLES, max_tokens=MAX_NEW_TOKENS, **SAMPLING)
    t0 = time.time()
    outs = llm.chat([build_messages(t) for t in tasks], params)
    print(f"生成用时 {time.time() - t0:.0f}s")
    for ti, out in enumerate(outs):
        for k, c in enumerate(out.outputs):
            records.append({"task": ti, "k": k, "text": c.text, "gen_tokens": len(c.token_ids), "finish": c.finish_reason})
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

for r in records:
    r["html"] = extract_html(r["text"])
    if r["html"]:
        d = SITES / f"{r['task']:03d}" / str(r["k"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(r["html"])

n_del = sum(r["html"] is not None for r in records)
n_len = sum(r["finish"] == "length" for r in records)
print(f"{len(records)} 份输出：交付 HTML {n_del}，被截断 {n_len}，平均生成 {sum(r['gen_tokens'] for r in records) / len(records):.0f} token")

# %% [markdown]
# ## 渲染截图 + 运行检查
# 通过本地 http 服务打开页面（`file://` 会让 React / ES module 页面白屏）。先滚动到底触发"滚动出现"的动画，再截整页。
# 运行检查（官方 runtime gate）：加载过程中有未捕获异常、内联脚本语法错误、或页面卡死，奖励直接为 0。

# %%
import socket
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


PORT = free_port()
server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
                          cwd=SITES, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)

ENV_NOISE = re.compile(r"Failed to fetch|NetworkError|net::|Load failed|ERR_|CORS", re.I)
SKIP_SCRIPT_TYPES = {"text/babel", "application/json", "application/ld+json", "importmap", "text/template",
                     "text/x-template", "text/plain", "text/markdown", "x-shader/x-vertex", "x-shader/x-fragment"}
NODE = shutil.which("node")


def syntax_errors(html):
    """`node --check` every inline classic/module script; returns the first failure or None."""
    if not NODE:
        return None
    for attrs, body in re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", html, flags=re.S | re.I):
        m = re.search(r"type\s*=\s*['\"]([^'\"]+)", attrs, flags=re.I)
        typ = (m.group(1).lower() if m else "")
        if typ in SKIP_SCRIPT_TYPES or not body.strip():
            continue
        suffix = ".mjs" if typ == "module" else ".js"
        p = WORKDIR / f"_check{suffix}"
        p.write_text(body)
        r = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            lines = r.stderr.strip().splitlines()
            hit = [l for l in lines if "Error" in l]
            return (hit or lines or ["syntax error"])[0][:200]
    return None


SCROLL_JS = """async () => {
  const h = Math.min(document.documentElement.scrollHeight, %d);
  for (let y = 0; y < h; y += 600) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); }
  window.scrollTo(0, 0);
}""" % SHOT_MAX_H


async def shoot_one(browser, r, sem):
    async with sem:
        url = f"http://127.0.0.1:{PORT}/{r['task']:03d}/{r['k']}/index.html"
        ctx = await browser.new_context(viewport={"width": SHOT_W, "height": SHOT_H})
        page = await ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            async def work():
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception:
                    pass  # 慢 CDN 不算页面的错，继续截图
                await page.wait_for_timeout(1200)
                await page.evaluate(SCROLL_JS)
                await page.wait_for_timeout(600)
                return await page.screenshot(full_page=True, type="jpeg", quality=80)
            jpg = await asyncio.wait_for(work(), timeout=75)
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
            if im.height > SHOT_MAX_H:
                im = im.crop((0, 0, im.width, SHOT_MAX_H))
            path = Path(url.replace(f"http://127.0.0.1:{PORT}", str(SITES))).with_name("shot.jpg")
            im.save(path, "JPEG", quality=80)
            r["shot"] = str(path)
            gray = im.convert("L").resize((64, max(1, im.height * 64 // im.width)))
            px = list(gray.tobytes()); mean = sum(px) / len(px)
            r["blank"] = (sum((p - mean) ** 2 for p in px) / len(px)) ** 0.5 < 3
            r["shot_h"] = im.height
        except asyncio.TimeoutError:
            r["runtime_fail"] = "hang"
        except Exception as e:
            r["render_fail"] = str(e)[:200]
        finally:
            real = [e for e in errors if not ENV_NOISE.search(e)]
            if real and "runtime_fail" not in r:
                r["runtime_fail"] = "pageerror: " + real[0][:200]
            await ctx.close()


async def shoot_all(rs):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        exe = os.environ.get("CHROMIUM_PATH")
        browser = await p.chromium.launch(**({"executable_path": exe} if exe else {}))
        sem = asyncio.Semaphore(6)
        await asyncio.gather(*(shoot_one(browser, r, sem) for r in rs))
        await browser.close()


todo = [r for r in records if r["html"]]
for r in todo:
    err = syntax_errors(r["html"])
    if err:
        r["runtime_fail"] = "syntax: " + err
t0 = time.time()
try:
    asyncio.get_running_loop()
    import nest_asyncio  # Jupyter 已经有事件循环
    nest_asyncio.apply()
except RuntimeError:
    pass
except ImportError:
    sh(f"{sys.executable} -m pip install -q nest_asyncio")
    import nest_asyncio
    nest_asyncio.apply()
asyncio.run(shoot_all(todo))
server.terminate()
n_shot = sum("shot" in r for r in records)
n_rt = sum("runtime_fail" in r for r in records)
n_rf = sum("render_fail" in r for r in records)
print(f"截图 {n_shot}/{len(todo)}，运行检查失败 {n_rt}，渲染失败 {n_rf}，用时 {time.time() - t0:.0f}s")

# %% [markdown]
# ## 判官
# 两个判官，提示词和官方一字不差：
# - **query**（每份单独）：需求实现度 0–1，只看做没做到，不看好不好看。
# - **pick**（每组一起）：把同一题的截图放在一起，挑出明显更好 / 更差的。8 轮，用 Williams 拉丁方安排顺序，抵消位置偏差。

# %%
QUERY_FIT_PROMPT = '# 维度：需求实现度 (query) —— 页面是否实现了 query 的要求\n\n> 对照用户 query 原文，看截图可见内容有没有把 query 要求的**主题、板块、要素、功能界面**做出来。\n> 只评"做了没做、做对没做对"，不评好不好看（好不好看由其它维度负责）。\n\n## 任务\n\n你是网页需求验收员。下面给你一段用户 query 和该 query 对应产出的网页全页截图，判断截图可见内容对 query 的实现程度，打一个 0 到 1 的连续分数。\n\n## 判分依据\n\n先把 query 拆成若干**可见的要求点**：主题/行业、指定的板块（如首页 hero、产品列表、定价表、FAQ、联系表单…）、指定的要素（如某个具体数据、图表、地图、时间轴…）、指定的功能界面（如搜索框、筛选器、购物车、登录框、日历…）、指定的风格或语言。然后逐点对照截图勾选。\n\n分数区间：\n- [0.0, 0.2)：截图可见内容基本没有实现 query，或主题完全错误\n- [0.2, 0.4)：只实现少量表面元素，多数核心要求缺失\n- [0.4, 0.6)：实现了主要方向，但多个关键板块、要素或功能界面明显缺失\n- [0.6, 0.8)：主体要求已实现，仅有少量次要缺失或偏差\n- [0.8, 1.0]：截图可见范围内完整、准确地实现了 query 的主题、板块、要素和功能界面\n\n先确定所在区间，再按缺失项的数量与重要性在区间内连续取值，不要只输出端点。\n\n## 注意\n\n- 只依据截图**可见**证据。截图看不到的交互（点击后才出现的内容）不算实现，也不算缺失，按可见部分判。\n- query 没提的东西做了不加分也不扣分。\n- 主题错了（做成了另一个行业/另一个产品）直接落到最低区间。\n- 页面空白、渲染失败、只有骨架 → 最低区间。\n- 不要因为页面漂亮就抬分，也不要因为页面丑就压分——这里只看需求覆盖。\n\n## 输出格式\n\n仅输出 JSON，不要 markdown：\n{"reason": "一句话：query 要求了哪几点，截图实现了哪些、缺了哪些", "query_score": 0.0}\n'

GROUP_PICK_PROMPT = '# 组内挑选（group pick）—— 挑出明显更好的与明显更差的，其余不排\n\n> 一次调用看同一 query 的全部 rollout 截图（编号随机）。任务不是排全序，而是：\n> 挑出**明显优于其他**的若干张（0 到多张）、**明显差于其他**的若干张（0 到多张）；\n> 其余视为"一般"。如果整组看不出明显差别，就明确说"无明显差别"，**不要硬排**。\n\n## 任务\n\n下面是 {n} 个针对同一个建站需求生成的网页全页截图，编号 0 到 {n_max}（顺序随机，与质量无关）。\n\n请你以**整体审美观感**为主要依据，回答三个问题：\n1. 哪几张**明显比组内其他的更好**？（可以是 0 张、1 张或多张）\n2. 哪几张**明显比组内其他的更差**？（可以是 0 张、1 张或多张）\n3. 如果全组水平接近、挑不出明显好坏，就直接判定"无明显差别"。\n\n建站需求（仅用于理解这是个什么页面、给视觉判断提供上下文，**不评需求实现度**）：\n{query}\n\n## 怎么判"明显"\n\n- "明显"的标准是：**一个普通人把这几张并排看一眼，不用细想就能指出来**。需要反复对比、放大细看才能分出高下的，不算明显。\n- 好坏都是**相对组内**而言，不看绝对水平：\n  - 即使这一组单独看都没有质量特别高的，但有一张明显比其他的要好一些，也可以把它选进 good。\n  - 反之，这一组质量都挺好，但有一两张稍逊一筹、且明显能看出来的，也可以把它们选进 bad。\n- 宁可少选、不要多选。选出的每一张都应该能用一句话说清"为什么它明显比其他的好 / 差"。\n- **不要为了凑数把中游的选进 good 或 bad**。\n\n## 整体审美观感看什么\n\n凭整体直觉：这个页面像不像一个成熟设计师做出来的成品？是否有统一的视觉语言、舒服的配色、清晰的层级、精致的素材、恰当的留白？看完第一反应是"这做得真好看"，还是"能用但普通"，还是"这很糟"。\n\n不要逐项打勾算分。但在组内比较时，下面这些**客观上公认的问题**是重要的参考：同组里带这些问题的页面，相对没有这些问题的页面更可能落到"明显差"。\n\n**渲染 / 结构类**\n- 白屏、纯色空白页、只有骨架 / 灰框 / loading 状态、内容没渲染出来\n- 页面大面积截断：图片、卡片、mockup、图表的边缘被切掉，尤其深色背景上的截断\n- 板块之间夹着大段无内容的纯色空带；某个区块一半有内容、一半空着\n- 元素互相重叠、文字压在图上看不清、内容溢出容器、明显错位\n\n**素材类**\n- 用简陋 / 画崩 / 比例失调的 SVG 涂鸦冒充图片（人物五官走形、物体变形、剪贴画感）\n- 占位符图片：灰框、破图、多张一模一样的占位 SVG 重复排列\n- 把通用图标（放大镜、星星、箭头）放得极大当主视觉，空洞廉价\n- 整页几乎没有任何视觉素材，只有文字 + 细线框，像文档站 / 线框稿\n\n**文字 / 配色类**\n- 文字不可读：对比度过低、字压在复杂背景上、字号失控\n- 配色刺眼冲突，或通篇单调死板（大片纯黑 / 纯白 / 纯灰，没有任何色彩层次）\n\n"明显好"则完全是主观感受：如果这一组里有哪一个页面相比其他的**明显让你眼前一亮**——特别有艺术感、设计感，或者特别酷炫——那它就是"明显好"。没有这种感觉，就不要选。\n\n## 注意\n\n- 编号是匿名候选，不要推测来源。\n- 不判断内容是否正确、功能是否可用、需求是否实现——那些由其他评审负责。\n- 多张页面如果几乎一样（同一模板、只是细节不同），它们应该同进同出，不要拆开。\n\n## 输出格式\n\n仅输出 JSON，不要 markdown：\n{\n  "reason": "先一句话说全组整体水平，再逐一说明每张被选进 good / bad 的理由（≤ 150 字）",\n  "good": [明显更好的编号列表，可为空],\n  "bad": [明显更差的编号列表，可为空],\n  "no_clear_difference": false\n}\n\n约束：good 与 bad 不能有交集；no_clear_difference 为 true 时 good 和 bad 必须都为空；不必覆盖所有编号，未列出的即为"一般"。\n'


def json_object(text):
    """First balanced JSON object in text (brace matching, like the official grader)."""
    s = text or ""
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
                continue
            if c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(s[start:i + 1])
                        return d if isinstance(d, dict) else None
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return None


def img_b64(path, width=JUDGE_IMG_W):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


_client = None


def chat(parts, retries=3):
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=180)
    content = [{"type": "text", "text": p} if isinstance(p, str) else p for p in parts]
    last = None
    for a in range(retries):
        try:
            resp = _client.chat.completions.create(model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE,
                                                   messages=[{"role": "user", "content": content}])
            return resp.choices[0].message.content
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"judge call failed: {last}")


def judge_query(query, shot_b64):
    if MOCK_JUDGE:
        return {"query_score": round(random.Random(hash(shot_b64)).uniform(0.1, 1.0), 2), "reason": "mock"}
    for _ in range(3):
        d = json_object(chat([f"【用户 query】\n{query}\n\n【该 query 的产出页面截图】：", image_part(shot_b64), QUERY_FIT_PROMPT]))
        if d is not None:
            try:
                s = float(d.get("query_score"))
                if 0.0 <= s <= 1.0:
                    return {"query_score": s, "reason": str(d.get("reason", ""))[:400]}
            except (TypeError, ValueError):
                pass
    return None


def williams_orders(n, seed=0):
    """Even n: n orders, each shot in each position once, each ordered adjacency once."""
    base, lo, hi = [0], 1, n - 1
    while len(base) < n:
        base.append(hi); hi -= 1
        if len(base) < n:
            base.append(lo); lo += 1
    perm = list(range(n)); random.Random(seed).shuffle(perm)
    return [[perm[(x + r) % n] for x in base] for r in range(n)]


def pick_orders(n, rounds, seed=0):
    if rounds == n and n % 2 == 0:
        return williams_orders(n, seed)
    rnd = random.Random(seed)
    return [rnd.sample(range(n), n) for _ in range(rounds)]


def pick_once(shots_b64, query, order):
    n = len(shots_b64)
    if MOCK_JUDGE:
        rnd = random.Random(sum(order) + len(shots_b64[order[0]]))
        good = [order[0]] if rnd.random() < 0.7 else []
        bad = [order[-1]] if rnd.random() < 0.7 and order[-1] not in good else []
        return {"good": good, "bad": bad, "no_clear_difference": not (good or bad), "order": order}
    rubric = GROUP_PICK_PROMPT.replace("{n}", str(n)).replace("{n_max}", str(n - 1)).replace("{query}", query)
    parts = []
    for k, idx in enumerate(order):
        parts += [f"【编号 {k}】", image_part(shots_b64[idx])]
    parts.append(rubric)
    try:
        d = json_object(chat(parts))
    except RuntimeError:
        d = None
    if d is None:
        return {"error": True, "order": order}

    def ints(lst):
        out = []
        for x in lst if isinstance(lst, list) else []:
            try: out.append(int(x))
            except (TypeError, ValueError): pass
        return out
    good = sorted({order[k] for k in ints(d.get("good")) if 0 <= k < n})
    bad = [i for i in sorted({order[k] for k in ints(d.get("bad")) if 0 <= k < n}) if i not in good]
    ncd = bool(d.get("no_clear_difference")) and not (good or bad)
    return {"good": good, "bad": bad, "no_clear_difference": ncd, "reason": str(d.get("reason", ""))[:400], "order": order}


PICK_MIN_OK, PICK_NCD_ZERO_MIN, PICK_NET_DEADZONE = 5, 2, 1
QUERY_DEDUCT = [(0.9, 0.0), (0.6, 0.2), (0.4, 0.4), (0.2, 0.8)]
RAW_LO, RAW_HI = -2.0, 1.0


def combine(votes, n):
    """Official group_v1 vote combination (see group_pick.py: combine)."""
    R = len(votes)
    ok = R >= PICK_MIN_OK
    if not R:
        return {"ok": ok, "pick_norm": [0.0] * n, "rounds": 0}
    n_ncd = sum(1 for v in votes if v[0] == "N")
    ncd = PICK_NCD_ZERO_MIN > 0 and n_ncd >= PICK_NCD_ZERO_MIN
    pick_sum = [sum(1 if v[i] == "G" else -1 if v[i] == "B" else 0 for v in votes) for i in range(n)]
    noise = [i for i in range(n) if pick_sum[i] != 0 and abs(pick_sum[i]) <= PICK_NET_DEADZONE]
    eff = [0 if (ncd or i in noise) else pick_sum[i] for i in range(n)]
    return {"ok": ok, "pick_norm": [s / R for s in eff] if ok else [0.0] * n, "rounds": R, "n_ncd": n_ncd}


def reward_one(pick_norm, query_score, runtime_ok):
    """Official reward_one: (pick_norm - deduct + 2) / 3, with query and runtime floors."""
    if query_score is None or pick_norm is None:
        return None
    if not runtime_ok:
        return 0.0
    for lo, d in QUERY_DEDUCT:
        if query_score >= lo:
            return round((pick_norm - d - RAW_LO) / (RAW_HI - RAW_LO), 4)
    return 0.0

# %%
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
shot_rows = [r for r in records if "shot" in r]
with ThreadPoolExecutor(JUDGE_CONCURRENCY) as pool:
    b64s = list(pool.map(lambda r: img_b64(r["shot"]), shot_rows))
    for r, b in zip(shot_rows, b64s):
        r["b64"] = b
    qs = list(pool.map(lambda r: judge_query(tasks[r["task"]]["query"], r["b64"]), shot_rows))
for r, q in zip(shot_rows, qs):
    r["query_score"] = q["query_score"] if q else None
    r["query_reason"] = q["reason"] if q else None
print(f"query 判官完成：{len(shot_rows)} 次，用时 {time.time() - t0:.0f}s")

t0 = time.time()
groups = {}
for r in records:
    groups.setdefault(r["task"], []).append(r)


def run_group(ti):
    rows = [r for r in groups[ti] if "shot" in r]
    if len(rows) < 2:
        return ti, rows, None
    orders = pick_orders(len(rows), PICK_ROUNDS, seed=ti)
    rounds = [pick_once([r["b64"] for r in rows], tasks[ti]["query"], o) for o in orders]
    votes = []
    for rd in rounds:
        if rd.get("error"):
            continue
        if rd["no_clear_difference"]:
            votes.append(["N"] * len(rows))
        else:
            votes.append(["G" if i in rd["good"] else "B" if i in rd["bad"] else "-" for i in range(len(rows))])
    return ti, rows, {"combine": combine(votes, len(rows)), "rounds": rounds}


with ThreadPoolExecutor(max(1, JUDGE_CONCURRENCY // 2)) as pool:
    picks = list(pool.map(run_group, sorted(groups)))
print(f"pick 判官完成：{len(picks)} 组 × {PICK_ROUNDS} 轮，用时 {time.time() - t0:.0f}s")

# %% [markdown]
# ## 算奖励
# 和官方一致：
# - 没交付 HTML → 硬 0（模型应该从这里学到教训）
# - 运行检查失败 → 0
# - 渲染失败 / 判官失败 → 记为无效，训练时用组均值替换（不贡献梯度）
# - 其余：`reward = (pick_norm − query_deduct + 2) / 3`，query 分低于 0.2 直接 0

# %%
for ti, rows, pk in picks:
    pn = pk["combine"]["pick_norm"] if pk else [0.0] * len(rows)
    for r, p in zip(rows, pn):
        r["pick_norm"] = p
for r in records:
    if r["html"] is None:
        r["reward"], r["status"] = 0.0, "no_delivery"
    elif r.get("render_fail"):
        r["reward"], r["status"] = None, "render_fail"
    elif r.get("runtime_fail") and "shot" not in r:
        r["reward"], r["status"] = 0.0, "runtime_fail"
    else:
        rw = reward_one(r.get("pick_norm"), r.get("query_score"), not r.get("runtime_fail"))
        r["reward"] = rw
        r["status"] = "ok" if rw is not None else "judge_fail"
        if r.get("runtime_fail") and rw is not None:
            r["status"] = "runtime_fail"

# %% [markdown]
# ## 报告
# 最关键的指标是**组内标准差**：GRPO 只从同组内的差异里学习。如果一组 8 份得分几乎一样（标准差 < 0.05），这组就学不到东西。

# %%
import statistics as st

valid = [r for r in records if r["reward"] is not None]
by_task = {}
for r in valid:
    by_task.setdefault(r["task"], []).append(r["reward"])
stds = {ti: (st.pstdev(v) if len(v) > 1 else 0.0) for ti, v in by_task.items()}
flat = [ti for ti, s in stds.items() if s < 0.05]
status_counts = {}
for r in records:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
qv = [r["query_score"] for r in records if r.get("query_score") is not None]

summary = {
    "n_tasks": len(tasks), "n_samples": N_SAMPLES, "judge_model": JUDGE_MODEL if not MOCK_JUDGE else "MOCK",
    "status": status_counts,
    "truncated": sum(r["finish"] == "length" for r in records),
    "mean_reward": round(st.mean(r["reward"] for r in valid), 4) if valid else None,
    "mean_query_score": round(st.mean(qv), 4) if qv else None,
    "mean_group_std": round(st.mean(stds.values()), 4) if stds else None,
    "flat_groups": f"{len(flat)}/{len(stds)}",
    "mean_gen_tokens": round(st.mean(r["gen_tokens"] for r in records)),
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
ok_ratio = 1 - len(flat) / max(1, len(stds))
print("\n结论：", "组内差异足够，可以进入阶段 1 训练。" if ok_ratio >= 0.5 else
      "超过一半的组几乎没有差异，先排查判官或采样温度，再开始训练。")

out = [{k: v for k, v in r.items() if k not in ("text", "html", "b64")} | {"query": tasks[r["task"]]["query"]} for r in records]
(WORKDIR / "results.json").write_text(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=1))
print("结果已保存：", WORKDIR / "results.json")

# %% [markdown]
# ## 看图：每题最高分和最低分

# %%
def gallery(n_tasks=4):
    html = []
    for ti in sorted(by_task)[:n_tasks]:
        rows = sorted([r for r in groups[ti] if r["reward"] is not None and "shot" in r], key=lambda r: r["reward"])
        if not rows:
            continue
        cells = []
        for tag, r in (("最高", rows[-1]), ("最低", rows[0])):
            b = img_b64(r["shot"], 360)
            cells.append(f"<div style='flex:1;min-width:260px'><b>{tag} reward={r['reward']:.3f}</b> · query={r.get('query_score')} · pick={r.get('pick_norm', 0):+.2f}"
                         f"<br><img src='data:image/jpeg;base64,{b}' style='width:100%;max-height:520px;object-fit:cover;object-position:top;border:1px solid #ccc'></div>")
        html.append(f"<h4>题 {ti}：{tasks[ti]['query'][:120]}…</h4><div style='display:flex;gap:16px;flex-wrap:wrap'>{''.join(cells)}</div>")
    return "".join(html)


if IN_COLAB or "ipykernel" in sys.modules:
    from IPython.display import HTML, display
    display(HTML(gallery()))
else:
    (WORKDIR / "gallery.html").write_text(gallery())
    print("画廊已保存：", WORKDIR / "gallery.html")
