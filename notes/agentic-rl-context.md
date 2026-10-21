# MiMo 开源 RL 库中的 agentic RL 与上下文管理

> 依据：XiaomiMiMo/verl 的 `mimo-oss` 分支（commit 79090e5），以及它的两个子模块 `third_party/mimoagent-osr`（commit c3df846）和 `third_party/uni_agent`（commit c63e0b0）。

## 1. 是不是 agentic RL？是

- **多轮工具调用**：每条轨迹都在一个独立的 K8s pod 沙箱里运行。agent 在里面调用 bash、read、edit、grep 等工具，最多 500 步（`step_limit: 500`）。最后在同一个沙箱里跑测试或 verify 来计算奖励。
- **harness 层**：用的是 `mimoagent`，它是从 mini-swe-agent fork 出来的。Code 任务混用 4 种 mini-harness（`config/agent/code/`）：
  - `mini-bash`：只有 bash 一个工具
  - `mini-claude-code`：Bash/Read/Write/Edit/Grep/Glob
  - `mini-codex`：exec_command + apply_patch，走 responses 协议
  - `mini-mimocode`
- **接入训练的两种方式**：
  - Cyber、General、Visual：由 verl 的 AgentLoop 直接驱动 mimoagent（`recipes/{arvo,general,design}/agent_loop.py`）。
  - Code：经过 uni-agent 的模型网关。harness 像调用普通 OpenAI/Anthropic API 那样发请求，网关在中间把 token 记录下来（`recipes/code/mimoagent_runner.py`）。

## 2. 为什么上下文压缩在 RL 里是个难题

RL 训练需要的是**一条只追加、不改写的 token 序列**，外加一个 mask 标出哪些 token 是模型生成的（这些参与 loss）。上下文压缩会把历史替换成一段摘要，新上下文就不再是旧上下文的前缀，于是没法拼成一条序列训练。

还有一个相关的坑：**不能每轮把消息列表重新套一遍 chat template 来得到 token**。重新渲染得到的 token 和模型当时实际生成的 token 不一致：tool call 文本会被改写，thinking 会被删掉，解码再编码也不能保证还原。`recipes/general/token_trace.py` 的做法是：模型生成的部分直接保留原始 token id，只对环境新加进来的消息（工具结果等）做 tokenize。

## 3. 这个库怎么处理

### 3.1 开源的训练配置默认不开压缩，靠"避免溢出"
- `mini-codex.yaml` 里明确写了 `compaction_enabled: false`，其余几个 harness 也没有配置 Compact 工具。
- 取而代之的做法：
  - **上下文开到 256K**（`MAXLEN=262144`）。Qwen3.5 只有 8 层全注意力，每 token 的 KV 只有 32KB，长上下文的推理成本比较低。
  - **截断工具输出**：`max_observation_length` 为 48000 字符，约 12K token，从中间截断（mimoagent 的 `truncate_middle`）。Codex 那一路仿照 codex-rs 做了 4 层输出预算（`tools/codex/output_budget.py`）。
  - **prompt 太长时从中间截断任务文本**（`agent_loop.py` 中的 middle-truncate），而不是用 verl 默认的左截断。
  - **响应预算用完就结束**：抛出 `ResponseBudgetExhausted`，轨迹带着 `truncated` 标记正常收尾。

### 3.2 mimoagent 里实现了"由模型自己决定的压缩"（`src/mimoagent/compaction/`）
- **触发方式**：模型主动调用一个 Compact 工具。没有按 token 阈值自动触发，也不会在报错时被动触发。
- **重建后的历史**：`[system…, 压缩边界, 原始任务（逐字保留）, 摘要]`。原始任务逐字保留，防止任务描述在多次摘要后走样。
- **多次压缩**：只对上一个压缩边界之后的内容（即上一份摘要加上之后新做的工作）做摘要，不会把已经丢掉的历史再拿回来。
- 摘要提示词还要求模型："如果压缩前的细节丢了，先说明有缺口，再继续依赖它"。

### 3.3 训练端：一次 rollout 拆成多条 chain（uni-agent，`gateway/session/session.py`）
- 网关对每个请求的消息逐条计算前缀哈希：
  - 如果能接在某条已有 chain 的末尾，就追加到那条 chain 上；
  - 接不上（压缩之后、开子 agent、改写历史等情况），就**新开一条 chain**。
- 会话结束时，每条 chain 各产出一条训练序列。这就是报告里 Sample → Sequence → **Context** → Segment 层级中的 Context。
- **奖励怎么分**：只对最后一条 chain 打分，然后把同一个奖励**广播**给这次会话的所有 chain（`framework.py: _score_trajectories`）。
- 报告 §6.1 补充了一条规则：如果某个 Context 里没有剩下任何模型生成的 turn（例如被 mask 光了），这个 Context 就直接丢弃。

## 4. 对自己做缩小版 LoRA RL 的建议

1. **先不做压缩**：选短任务，上下文控制在 16K–32K；截断工具输出；设置步数上限。单卡 80GB 加 LoRA，训练时大约 32K 就是上限了（瓶颈是激活值）。
2. **预算用完的轨迹要单独处理**：如果直接给 0 分，模型会学到"越短越好"，这是一种奖励偏差。可以考虑把这类轨迹排除出 loss，或者单独统计它们占的比例。官方是怎么处理的，我没有查到明确结论。
3. **如果确实需要压缩**，照搬这里的思路：
   - 坚持 token-in/token-out，绝不重新渲染历史；
   - 压缩前后各算一条序列，共享同一个奖励；
   - 原始任务逐字保留；
   - 摘要那一轮也是模型生成的，所以也参与训练。这样模型能学会"写出好用的摘要"。
4. **多条 chain 带来的副作用**：同一条轨迹拆出的 chain 越多，它在 loss 里的权重越大；而且前面 chain 里的行为也拿到同一个奖励，信用分配变粗。官方用的 `loss_agg_mode=prompt-mean` 是按 prompt 求平均，按理能缓解前一个问题，但我没看到实现细节，这一点只是推测。
