"""WebDev RL reward pipeline shared by the eval (stage 0) and training (stage 1) notebooks.

Single-turn generation of one HTML file, rendered locally with Playwright over http, gated on
runtime errors, and scored with the official query judge + group pick, combined with the
official group_v1 reward. Judge prompts and reward math are ported from XiaomiMiMo/verl
(branch mimo-oss, recipes/design/grader_service, Apache-2.0).
"""

import asyncio
import base64
import io
import json
import os
import random
import re
import shutil
import socket
import statistics as st
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- settings
CFG = dict(
    judge_base_url="https://openoneapi.com/v1",
    judge_model="kimi-k3",
    judge_temperature=1.0,      # both official judges use 1.0
    judge_concurrency=8,
    pick_rounds=8,              # official: 8 rounds, Williams order
    shot_w=1280, shot_h=800,    # screenshot viewport
    shot_max_h=12000,           # full-page cap (official 12000)
    judge_img_w=720,            # images are scaled to this width for the judges (official 720)
    render_concurrency=6,
    chromium_path=os.environ.get("CHROMIUM_PATH"),
    mock_judge=os.environ.get("MOCK_JUDGE") == "1",
)
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failures": 0}
_usage_lock = threading.Lock()


def configure(**kw):
    unknown = set(kw) - set(CFG)
    if unknown:
        raise KeyError(f"unknown settings: {sorted(unknown)}")
    CFG.update(kw)


def judge_api_key():
    """Env var, then a key file (used when the VM is driven from the CLI), then Colab secrets."""
    if os.environ.get("JUDGE_API_KEY"):
        return os.environ["JUDGE_API_KEY"]
    for p in ("/content/.judge_key", os.path.expanduser("~/.judge_key")):
        if os.path.exists(p):
            return Path(p).read_text().strip()
    try:
        from google.colab import userdata
        return userdata.get("JUDGE_API_KEY")
    except Exception:
        return None


# ---------------------------------------------------------------- tasks & prompts
def load_tasks():
    """All 2093 official webdev tasks; the first 100 after a fixed shuffle are the held-out eval set."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download("XiaomiMiMo/MiMo-V2.6-RL-oss", "webdev.parquet", repo_type="dataset"))
    tasks = []
    for _, row in df.iterrows():
        inst = json.loads(row["extra_info"]["instance_json"])
        tasks.append({"id": inst["instance_id"], "query": inst["problem_statement"].strip()})
    random.Random(0).shuffle(tasks)
    return tasks[:100], tasks[100:]


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


def build_messages(query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Build a website for the following request:\n\n{query}"},
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


# ---------------------------------------------------------------- rendering + runtime gate
ENV_NOISE = re.compile(r"Failed to fetch|NetworkError|net::|Load failed|ERR_|CORS", re.I)
SKIP_SCRIPT_TYPES = {"text/babel", "application/json", "application/ld+json", "importmap", "text/template",
                     "text/x-template", "text/plain", "text/markdown", "x-shader/x-vertex", "x-shader/x-fragment"}
NODE = shutil.which("node") or ("/tools/node/bin/node" if os.path.exists("/tools/node/bin/node") else None)


def syntax_error(html, scratch):
    """`node --check` every inline script; the first failure message, or None."""
    if not NODE:
        return None
    for attrs, body in re.findall(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", html, flags=re.S | re.I):
        m = re.search(r"type\s*=\s*['\"]([^'\"]+)", attrs, flags=re.I)
        typ = m.group(1).lower() if m else ""
        if typ in SKIP_SCRIPT_TYPES or not body.strip():
            continue
        p = Path(scratch) / ("_check.mjs" if typ == "module" else "_check.js")
        p.write_text(body)
        r = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            lines = r.stderr.strip().splitlines()
            hit = [l for l in lines if "Error" in l]
            return (hit or lines or ["syntax error"])[0][:200]
    return None


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


SCROLL_JS = """async (maxH) => {
  const h = Math.min(document.documentElement.scrollHeight, maxH);
  for (let y = 0; y < h; y += 600) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); }
  window.scrollTo(0, 0);
}"""


async def _shoot_one(browser, url, out_path, sem):
    res = {}
    async with sem:
        ctx = await browser.new_context(viewport={"width": CFG["shot_w"], "height": CFG["shot_h"]})
        page = await ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            async def work():
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception:
                    pass  # a slow CDN is not the page's fault; shoot anyway
                await page.wait_for_timeout(1200)
                await page.evaluate(SCROLL_JS, CFG["shot_max_h"])
                await page.wait_for_timeout(600)
                return await page.screenshot(full_page=True, type="jpeg", quality=80)
            jpg = await asyncio.wait_for(work(), timeout=75)
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
            if im.height > CFG["shot_max_h"]:
                im = im.crop((0, 0, im.width, CFG["shot_max_h"]))
            im.save(out_path, "JPEG", quality=80)
            res["shot"] = str(out_path)
            res["shot_h"] = im.height
            g = list(im.convert("L").resize((64, max(1, im.height * 64 // im.width))).tobytes())
            mean = sum(g) / len(g)
            res["blank"] = (sum((p - mean) ** 2 for p in g) / len(g)) ** 0.5 < 3
        except asyncio.TimeoutError:
            res["runtime_fail"] = "hang"
        except Exception as e:
            res["render_fail"] = str(e)[:200]
        finally:
            real = [e for e in errors if not ENV_NOISE.search(e)]
            if real and "runtime_fail" not in res:
                res["runtime_fail"] = "pageerror: " + real[0][:200]
            await ctx.close()
    return res


async def _shoot_all(jobs):
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        exe = CFG["chromium_path"]
        browser = await p.chromium.launch(**({"executable_path": exe} if exe else {}))
        sem = asyncio.Semaphore(CFG["render_concurrency"])
        out = await asyncio.gather(*(_shoot_one(browser, u, o, sem) for u, o in jobs))
        await browser.close()
    return out


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", asyncio.run(coro)))
    t.start()
    t.join()
    return box["r"]


def render(records, sites_dir):
    """Write each record's HTML under sites_dir/<task>/<k>/, check scripts, screenshot over http."""
    sites_dir = Path(sites_dir)
    todo = []
    for r in records:
        if not r.get("html"):
            continue
        d = sites_dir / f"{r['task']:04d}" / str(r["k"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(r["html"])
        r["dir"] = str(d)
        err = syntax_error(r["html"], d)
        if err:
            r["runtime_fail"] = "syntax: " + err
        todo.append(r)
    if not todo:
        return
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                              cwd=sites_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        jobs = [(f"http://127.0.0.1:{port}/{Path(r['dir']).relative_to(sites_dir).as_posix()}/index.html",
                 Path(r["dir"]) / "shot.jpg") for r in todo]
        for r, res in zip(todo, _run_async(_shoot_all(jobs))):
            if "runtime_fail" in r:
                res.pop("runtime_fail", None)
            r.update(res)
    finally:
        server.terminate()


# ---------------------------------------------------------------- judges
QUERY_FIT_PROMPT = __QUERY_FIT__

GROUP_PICK_PROMPT = __GROUP_PICK__


def json_object(text):
    """First balanced JSON object in text (brace matching, like the official grader)."""
    s = text or ""
    start = s.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
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


def img_b64(path, width=None):
    width = width or CFG["judge_img_w"]
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _image_part(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


_client = None


def _chat(parts, retries=3):
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=CFG["judge_base_url"], api_key=judge_api_key(), timeout=240)
    content = [{"type": "text", "text": p} if isinstance(p, str) else p for p in parts]
    last = None
    for a in range(retries):
        try:
            resp = _client.chat.completions.create(model=CFG["judge_model"], temperature=CFG["judge_temperature"],
                                                   messages=[{"role": "user", "content": content}])
            with _usage_lock:
                USAGE["calls"] += 1
                if resp.usage:
                    USAGE["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    USAGE["completion_tokens"] += resp.usage.completion_tokens or 0
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    with _usage_lock:
        USAGE["failures"] += 1
    raise RuntimeError(f"judge call failed: {last}")


def judge_query(query, shot_b64):
    if CFG["mock_judge"]:
        return {"query_score": round(random.Random(hash(shot_b64)).uniform(0.1, 1.0), 2), "reason": "mock"}
    for _ in range(3):
        try:
            d = json_object(_chat([f"【用户 query】\n{query}\n\n【该 query 的产出页面截图】：", _image_part(shot_b64), QUERY_FIT_PROMPT]))
        except RuntimeError:
            return None
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
        base.append(hi)
        hi -= 1
        if len(base) < n:
            base.append(lo)
            lo += 1
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    return [[perm[(x + r) % n] for x in base] for r in range(n)]


def pick_orders(n, rounds, seed=0):
    if rounds == n and n % 2 == 0:
        return williams_orders(n, seed)
    rnd = random.Random(seed)
    return [rnd.sample(range(n), n) for _ in range(rounds)]


def pick_once(shots_b64, query, order):
    n = len(shots_b64)
    if CFG["mock_judge"]:
        rnd = random.Random(sum(order) + len(shots_b64[order[0]]))
        good = [order[0]] if rnd.random() < 0.7 else []
        bad = [order[-1]] if rnd.random() < 0.7 and order[-1] not in good else []
        return {"good": good, "bad": bad, "no_clear_difference": not (good or bad), "order": order}
    rubric = GROUP_PICK_PROMPT.replace("{n}", str(n)).replace("{n_max}", str(n - 1)).replace("{query}", query)
    parts = []
    for k, idx in enumerate(order):
        parts += [f"【编号 {k}】", _image_part(shots_b64[idx])]
    parts.append(rubric)
    try:
        d = json_object(_chat(parts))
    except RuntimeError:
        d = None
    if d is None:
        return {"error": True, "order": order}

    def ints(lst):
        out = []
        for x in lst if isinstance(lst, list) else []:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
        return out
    good = sorted({order[k] for k in ints(d.get("good")) if 0 <= k < n})
    bad = [i for i in sorted({order[k] for k in ints(d.get("bad")) if 0 <= k < n}) if i not in good]
    ncd = bool(d.get("no_clear_difference")) and not (good or bad)
    return {"good": good, "bad": bad, "no_clear_difference": ncd, "reason": str(d.get("reason", ""))[:400], "order": order}


# ---------------------------------------------------------------- reward (official group_v1)
PICK_MIN_OK, PICK_NCD_ZERO_MIN, PICK_NET_DEADZONE = 5, 2, 1
QUERY_DEDUCT = [(0.9, 0.0), (0.6, 0.2), (0.4, 0.4), (0.2, 0.8)]
RAW_LO, RAW_HI = -2.0, 1.0


def combine(votes, n):
    """Official vote combination (group_pick.py: combine)."""
    R = len(votes)
    ok = R >= PICK_MIN_OK
    if not R:
        return {"ok": ok, "pick_norm": [0.0] * n, "rounds": 0, "n_ncd": 0}
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


def grade(records, queries):
    """Fill query_score / pick_norm / reward / status on records.

    records: dicts with task (index into queries), k, html, and render output.
    Group = all records sharing a task index. Mirrors the official rules:
    no delivery -> 0, runtime failure -> 0, render/judge failure -> None (invalid).
    """
    shot_rows = [r for r in records if "shot" in r]
    with ThreadPoolExecutor(CFG["judge_concurrency"]) as pool:
        for r, b in zip(shot_rows, pool.map(lambda r: img_b64(r["shot"]), shot_rows)):
            r["b64"] = b
        for r, q in zip(shot_rows, pool.map(lambda r: judge_query(queries[r["task"]], r["b64"]), shot_rows)):
            r["query_score"] = q["query_score"] if q else None
            r["query_reason"] = q["reason"] if q else None

        groups = {}
        for r in shot_rows:
            groups.setdefault(r["task"], []).append(r)
        calls = []
        for ti, rows in groups.items():
            if len(rows) < 2:
                continue
            for o in pick_orders(len(rows), CFG["pick_rounds"], seed=ti):
                calls.append((ti, o))
        results = list(pool.map(lambda c: pick_once([r["b64"] for r in groups[c[0]]], queries[c[0]], c[1]), calls))

    rounds_by_task = {}
    for (ti, _), res in zip(calls, results):
        rounds_by_task.setdefault(ti, []).append(res)
    for ti, rows in groups.items():
        votes = []
        for rd in rounds_by_task.get(ti, []):
            if rd.get("error"):
                continue
            if rd["no_clear_difference"]:
                votes.append(["N"] * len(rows))
            else:
                votes.append(["G" if i in rd["good"] else "B" if i in rd["bad"] else "-" for i in range(len(rows))])
        pn = combine(votes, len(rows))["pick_norm"] if len(rows) >= 2 else [0.0] * len(rows)
        for r, p in zip(rows, pn):
            r["pick_norm"] = p

    for r in records:
        if not r.get("html"):
            r["reward"], r["status"] = 0.0, "no_delivery"
        elif r.get("render_fail"):
            r["reward"], r["status"] = None, "render_fail"
        elif r.get("runtime_fail") and "shot" not in r:
            r["reward"], r["status"] = 0.0, "runtime_fail"
        else:
            rw = reward_one(r.get("pick_norm"), r.get("query_score"), not r.get("runtime_fail"))
            r["reward"] = rw
            r["status"] = ("runtime_fail" if r.get("runtime_fail") else "ok") if rw is not None else "judge_fail"
        r.pop("b64", None)
    return records


def fill_invalid_with_group_mean(records):
    """Official convention: an invalid row gets its group's mean valid reward, so it adds no gradient."""
    by_task = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)
    out = []
    for r in records:
        if r["reward"] is not None:
            out.append(r["reward"])
        else:
            vals = [x["reward"] for x in by_task[r["task"]] if x["reward"] is not None]
            out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def summarize(records):
    valid = [r for r in records if r["reward"] is not None]
    by_task = {}
    for r in valid:
        by_task.setdefault(r["task"], []).append(r["reward"])
    stds = {t: (st.pstdev(v) if len(v) > 1 else 0.0) for t, v in by_task.items()}
    status = {}
    for r in records:
        status[r["status"]] = status.get(r["status"], 0) + 1
    qv = [r["query_score"] for r in records if r.get("query_score") is not None]
    return {
        "n": len(records),
        "status": status,
        "truncated": sum(r.get("finish") == "length" for r in records),
        "mean_reward": round(st.mean(r["reward"] for r in valid), 4) if valid else None,
        "mean_query_score": round(st.mean(qv), 4) if qv else None,
        "mean_group_std": round(st.mean(stds.values()), 4) if stds else None,
        "flat_groups": sum(s < 0.05 for s in stds.values()),
        "groups": len(stds),
        "mean_gen_tokens": round(st.mean(r.get("gen_tokens", 0) for r in records)) if records else 0,
        "judge_usage": dict(USAGE),
    }
