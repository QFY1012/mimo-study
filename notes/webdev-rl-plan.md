# Web Design（webdev）RL：官方怎么做，以及怎么缩到 Colab 上

> 依据：XiaomiMiMo/verl `mimo-oss` 分支中的 `recipes/design/`、`config/agent/design/webdev.yaml`，以及 HF 数据集 `webdev.parquet`。
> 第 3 节以后是我的设计方案，尚未实现，也没有跑过。

## 1. 官方的 webdev 任务

**数据**：`webdev.parquet` 共 **2093** 条，每条是一段英文建站需求。例如 "Build me a static React website for a tutoring cafe called Study Bites, brutalist style…"。题目里写明只要静态站，内容用 mock 数据，不需要后端。

**Agent（多轮）**：
- 工具：Bash / Read / Write / Edit / Grep / Glob。
- 最多 64 步；每次工具输出最多 16000 字符。
- 系统提示要求把成品放到 `/workspace/dist/index.html`，可以通过 CDN 引入 Tailwind。
- 每条轨迹单独起一个 K8s pod，镜像为 `webdev-rl-opensource:v2`。

**奖励（训练时）**：基于组内相对比较，由一个独立的评分服务计算：
1. 把 pod 里的工作目录打包，发给评分服务。服务重新构建站点，用 Chromium 截一张**整页截图**。
2. **runtime 闸门**（不调用模型）：内联 `<script>` 没通过 `node --check`、加载时抛出未捕获异常、或者渲染卡死，奖励直接为 0。
3. **query 判官**（视觉模型，每条单独打分）：评"需求实现度"，0–1 分；不评好不好看。
4. **组内 pick 判官**（视觉模型）：把同一题的 8 张截图放在一起，挑出"明显更好"和"明显更差"的几张，其余算一般，不要求排出全序。共做 8 轮，用 Williams 拉丁方安排图片顺序，把"排在第一张的图更容易被选中"这种偏差从 ±0.28 票降到 ±0.03 票。
5. 合成公式：
   - `raw = pick_norm − query_deduct`，其中 `pick_norm` ∈ [−1, 1]，表示这张图被选为好或差的平均票数；
   - `reward = (raw + 2) / 3`，取值范围 [0.067, 1]；
   - query_deduct 分档扣分：query 分 ≥0.9 不扣，≥0.6 扣 0.2，≥0.4 扣 0.4，≥0.2 扣 0.8，<0.2 奖励直接为 0。
6. 特殊情况：
   - 没有交付 `dist/`：奖励记为硬 0，模型要从这里学到教训；
   - 截图或渲染失败：把这条的奖励替换成组均值，等于不贡献梯度。

**评测时**：改用绝对分数。在 pod 里截图，一次视觉调用给出 7 个维度的分数：
- `score = (visual + query_fulfillment + premium_assets) / 3`
- `visual` = 布局完整性、字体层级、配色、留白、内容丰富度这 5 项的平均。
- `premium_assets` 衡量素材质量，例如是否用了真实图片，而不是简陋的 SVG 或占位图。

**训练超参**：
- n=8，batch 32，1 个 epoch，学习率 1e-6，上下文 256K，按 prompt 平均 loss（prompt-mean）。
- 采样参数 temperature 0.6 / top_p 0.95 / top_k 20。配置注释里说，这组参数让第 1 步的分数从 0.26–0.36 升到 0.56，组内方差没有变化。
- 两条必须注意的配置：
  - **截图必须放在共享盘上**，否则每组都会因为截图数量太少被跳过，奖励全为 0，而且不报任何错误；
  - **评审网页要通过 http 访问**，用 `file://` 打开时 React 页面是白屏（实测 182 张里有 44 张是空白）。

## 2. 和报告的对应
- 报告 §4.2.3 的"开放式设计 = pointwise rubric + 组内对比"就是这套实现；开源版的训练路径只保留了 query 判官和组内 pick。
- 报告 §7.2：9B 模型在 MiMo Visual Coding (mini) 上，SFT 后 64.0，RL 后 72.4。

## 3. Colab 缩小版的设计（我的方案）

| 官方组件 | Colab 替代 |
|---|---|
| K8s pod 沙箱 | Colab 本机为每条 rollout 建一个临时目录，用 subprocess 执行命令并设超时 |
| 多轮 agent（64 步，256K 上下文） | **阶段 0：单轮**，模型直接输出完整的 `index.html`；**阶段 1**：多轮，只给 Write/Edit/Bash 三个工具，最多 8–16 步，上下文 32K |
| 评分服务（重建、截图、判官） | 本机用 Playwright + Chromium 截图；通过 `python -m http.server` 用 http 访问页面 |
| 视觉判官 | 调用任意兼容 OpenAI 接口的视觉模型 API；如果 GPU 有空余，也可以自己部署小型 VLM |
| 奖励 | 照搬：runtime 闸门 → query 判官 → 组内 pick → `(raw+2)/3` |
| 训练 | verl FSDP2 + LoRA r=64，学习率约 1e-5，batch 8×8，单卡 80GB |

**为什么先做单轮**：省掉 agent loop、多条 chain 和工具出错这些问题，奖励管线可以单独调通。单轮输出一个 HTML 文件通常在 4K–12K token，16K 上下文就够。

**判官调用量估算**（每步 8 题 × 8 条）：
- query 判官：64 次单图调用；
- 组内 pick：8 组 × 8 轮 = 64 次多图调用，每次 8 张图。
- 合计每步约 **128 次视觉调用**，大约百万输入 token 的量级。
- 每步花费约 **$0.2–3**，取决于用哪个模型。跑 50 步约 $10–150。这是粗略估算，需要按实际图片 token 数核算。

**防奖励作弊**（报告 §4.2.6 的思路缩小版）：
- runtime 闸门和 query 下限要保留，否则模型会学会生成漂亮但不能用的页面，或者文不对题的页面；
- 判官只看截图，不看源码，避免模型在页面里写"这个设计非常优秀"之类的文字去影响判官；
- 定期抽查高分截图。

**评测**：留出 100 题作为验证集，用 7 维绝对评分，每 10 步评一次。训练用的组内相对分数不能拿来跨步比较。
