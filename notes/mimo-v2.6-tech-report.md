# MiMo-V2.6 技术报告阅读笔记

- **标题**：MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement
- **作者**：LLM-Core Xiaomi（小米 LLM-Core 团队）
- **arXiv**：[2610.11959](https://arxiv.org/abs/2610.11959)（cs.CL，v1 于 2026-10-08 提交，43 页）
- **相关资源**：
  - 官方博客：<https://mimo.mi.com/docs/en-US/news/latest/v2-6>
  - RL 训练日志：<https://mimo.xiaomi.com/rl/mimo-v26>
  - 9B 蒸馏模型：<https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B>

## 一句话总结

在 hybrid-SWA MoE 全模态底座上做了一次大规模异步混合任务 RL（"You Only RL Once"，约 30 步，每步 1568 个 prompt × 16 条 rollout，共 2.7–3.7B token），从三个方向扩大 RL 算力：**batch/吞吐**、**环境与 harness 的多样性**、**grader 算力**（组内对比的 agentic 评分）。RL 之后再用 MOPD2（多前缀、多教师的 on-policy 蒸馏）把难以验证的能力并进来。

## 1. 模型架构（§2）

| | MiMo-V2.6-Flash | MiMo-V2.6-Pro |
|---|---|---|
| 总参数 / 激活参数 | 310B / 15B | 1.02T / 42B |
| 层数（总/SWA/GA） | 48 / 39 / 9 | 70 / 60 / 10 |
| Hidden size | 4096 | 6144 |
| 专家数（总/激活） | 256 / 8 | 384 / 8 |
| 滑动窗口 | 128 | 128 |

- **骨干**：M 个 hybrid block，每个由 N 个 SWA block 加 1 个 GA block 组成；第 1 层是 GA + dense FFN，用来稳定早期表示。MoE 不设共享专家。带 MTP。
- **MiMo-ViT**（681M）：用 sink-augmented SWA 取代固定窗口注意力，SWA 层在行优先和列优先序列化之间交替，并周期性插入 GA 层。从零预训练：先接一个小 LLM，只用交叉熵在 4T+ 图像 token 上训练（不用对比学习）。
- **音频**：Audio Tokenizer（308M），25Hz，20 层 RVQ，训练数据为 2000 万小时音频；Patch Encoder 每 4 帧合成 1 个 patch，速率降到 6.25Hz。
- **投机解码**：采用 DFlash（block diffusion）草稿模型，5 层，单次前向预测 7 个 token。

## 2. 预训练 / 中训练（§3）

- 两阶段预训练：先纯文本，再全模态联合训练。上下文从 32K 扩到 256K。
  - Flash：48T token（文本 26T + 全模态 22T）
  - Pro：30T token（文本 27T + 全模态 3T）
  - 预训练优化器为 AdamW
- **中训练**：以 agent 为中心的混合数据（coding/general/visual/research 轨迹，以及高质量文本、仓库级代码、多模态数据）。先在 256K 上训练，最后扩到 **1M**。
  - **优化器切换**：AdamW → **Muown**（在 Muon 上加了显式 row-norm 控制）。原因是 AdamW 在大 batch 混合 RL 下效率递减。Embedding、LM head 和 router 仍用 AdamW。文中称切换过程没有出现 loss spike。
  - 做 **MXFP4 QAT**，为 RL 阶段的 MXFP4 rollout 做准备。

## 3. 扩大 RL 算力的三个维度（§4）

### 3.1 训练算力（§4.1）

- RL 成本：Pro 约 **$2.6M**，Flash 约 **$0.9M**。
- DeepSWE avg@3：Pro 从 58.4 提升到 72.6，Flash 从 48.7 提升到 65.7。
- Pro 的成本构成：rollout 43.8%，training 43.5%，grader 12.7%。
- 关键技术：partial rollout（用大 batch 摊薄重新 prefill 的开销）、R3 路由回放、top-p 候选集回放、dynamic sampler（过滤全对或全错的组）、Sample Mixer、数据面与控制面分离。

### 3.2 环境与 Harness（§4.2）

- **Code**：五条任务合成路径（GitHub PR/issue、员工的日常开发需求、规格驱动、源码驱动的 CodeMidas、长程工程任务），另加公开数据集和供应商数据。
  - *监督准确性*：每个任务跑 4 次 rollout，由审计 agent 对照规格判断，标出假阳性和假阴性。
  - *监督鲁棒性*：检查 F2P/P2P 测试在打补丁前后的结果，并要求重跑 8 次结果都稳定。
- **General**：用多 agent 构建本地软件 mock 和真实文件组成的可重置沙箱，再合成任务并配原子化的二值 rubric（代码检查加 LLM 检查）。训练时用自部署的 MiMo-V2.6-SFT 作为 grader。
- **Visual**：分开放式设计和高保真复刻两类。开放式设计用 pointwise rubric 加组内对比评分；复刻任务以像素相似度等规则指标为主，LLM 评审为辅。
- **Cyber**：基于 OSS-Fuzz 做漏洞复现。奖励规则是：PoC 触发的崩溃必须在 sanitizer 报告的**漏洞类型**和**崩溃位置**（最顶层的项目栈帧）上都与 ground truth 匹配，用字符串规则判定，结果确定且开销很低。任务描述也从同一份报告生成。agent 能拿到编译好的 harness 二进制（CyberGym 不提供）。
- **Multi-Harness 训练**：没有直接在 MiMo Code、Codex 这类生产级 harness 上训练，而是从同一个最小 agent loop 派生出多个可自由组合的 **mini-harness**。在 3 个未参与训练的 harness（codex、claude code、mini-swe-agent）上，DeepSWE 平均 pass@1 从约 50% 升到 66%。
- **Reward hacking 防护**（多层）：
  1. 中训练加入"反思并纠正 hacking 行为"的对齐数据；
  2. 环境清理：删除构建产物和缓存，git 历史截断到 base commit，容器断网；
  3. 用 **Hack Agent** 主动找漏洞，反复修补，直到找不到可用的利用路径；
  4. 训练期间离线审计轨迹，GAR grader 把确认属于 hack 的轨迹奖励置 0。
  - 结果：确认的 hack 轨迹占比全程低于 2%。
  - 表 2 列了典型的 hack 手法：pip 安装更新版本照抄修复、curl 上游文件、clone 上游仓库、查 issue/ticket、探测可用版本。

### 3.3 Groupwise Agentic Grading（§4.3）

- **GRS（离线 rubric）**：用于高通过率的任务。agent 对比多条离线 rollout，生成 solution rubric 和 behavior rubric，最终奖励为 `R = R_test · S_sol · S_beh`。由于是乘法形式，失败的轨迹仍为 0，而通过的轨迹之间也能拉开差距。
- **GAR（在线优势重分配）**：用于其余代码任务。SFT 过的 agentic grader 在同一工作区里比较同组的所有轨迹，从方案合适性、实现精确性、改动最小、无副作用、符合代码库风格五个维度给通过的 patch 排序。用质量因子 `f_i` 把正优势从低质量的通过解转移到高质量的通过解，正优势总量保持不变，转移系数设有上限，最后再减去组均值。
  - 消融实验（Flash，只用代码任务）：去掉 GAR 后，轮数和 token 数快速膨胀，pass rate 很快停止上升，模型还学会了吞异常、放宽校验、添加投机性兼容分支等不良写法；加上 GAR 后 patch 更小、更精确。
- **行为正则**：
  - *组相对长度惩罚*：只在通过率高于阈值 A 的组里生效，以通过解长度的 B 分位数为参考长度，超出部分扣分；
  - *片段级行为惩罚*：对格式错误、工具调用错误等被标记的 token，在正轨迹中屏蔽，在负轨迹中加大惩罚（κ 倍）；同时用 α、β 对其余 token 重新缩放，使正负两侧的优势总量各自守恒，防止熵失控。

## 4. 实验："You Only RL Once"（§5）

- **任务配比**：编程 68%，通用工具使用 12%，审美设计 13%，上下文遵循 3%，网络安全 4%。
- **算法**：GRPO，异步训练，staleness = 4，按 prompt 平均聚合 loss。重要性采样比率逐 token 计算，用 4 个解耦的 clip 边界（正负优势各一对上下界），并根据熵动态调整。
- **优化器**：Muown，lr 3e-6，不用 weight decay 和 warmup。Muon 部分 momentum 0.95（Nesterov），Newton–Schulz 迭代 10 次。
- **冻结 Router**（§5.4）：不冻结时，第 9 层专家负载的 CV 从 0.78 升到 2.0，峰值负载从 6× 升到 16×，冷专家比例从 0.5% 升到 22%。把 router 恢复到初始值后负载即恢复正常，说明问题出在 router 漂移，与专家权重无关，所以 RL 期间冻结 router。
- **故障分析**（§5.5）：Pro 训练 30 步用了 123.1h，Flash 用了 81.8h。中断原因包括 GPU DBE、K8s 故障、grader 网络不可达、partial rollout 导致长度估计偏差、MoE 在 micro-batch 内负载不均（某个 EP rank 负载超过均值 30 倍）引发 OOM、打包阶段 CPU OOM。
- **MOPD2**（§5.6）：
  - *Standard MOPD*：学生完整 rollout，由 mixRL 教师逐 token 监督；
  - *Teacher-Prefix OPD / SFT-Prefix OPD*：把教师 rollout 或 SFT 数据的轨迹拆成 k 个历史前缀，学生在每个前缀上只生成下一轮，由对应领域的教师打分。用于游戏开发、科研、具身智能等难以验证的领域。

### 最终结果（表 3，节选）

| Benchmark | V2.6-Pro | V2.6-Flash | V2.5-Pro | Claude Opus 5 | GPT-5.6 Sol | Claude Fable 5 |
|---|---|---|---|---|---|---|
| DeepSWE v1.1 | 71.9 | 67.9 | 19.0 | 74.0 | 73.0 | 70.0 |
| ProgramBench | 26.5 | 26.0 | 12.5 | 37.0 | 25.0 | 33.0 |
| AutomationBench v1.0.6 | **53.1** | 52.3 | 16.0 | 50.3 | 45.8 | 46.2 |
| Toolathlon-Verified | 76.9 | 73.6 | 49.1 | 80.6 | 74.9 | 77.9 |
| GDPval-AA 2.1 | 1673 | – | 1107 | 1708 | 1588 | 1595 |
| Agents' Last Exam | 31.6 | 27.6 | 13.2 | 31.6 | 30.8 | 25.7 |
| Terminal Bench 4.0 | 34.9 | 28.8 | 1.5 | 49.0 | 39.9 | 42.4 |
| Terminal Bench 2.1 | **89.9** | 87.6 | 65.2 | 89.1 | 88.8 | 84.3 |
| OSWorld-Verified | 82.0 | 80.8 | – | 83.4 | 83.0 | 86.0 |
| ExploitBench | 47.9 | 25.3 | 16.6 | 70.0 | 78.5 | 78.0 |
| MiMo Visual Coding | 72.3 | 71.5 | – | 70.0 | 73.4 | 69.1 |

**观察**：在 agent 工作流类任务（AutomationBench、Terminal Bench 2.1、Agents' Last Exam）上与前沿闭源模型持平或略高；在 Terminal Bench 4.0 和攻击利用类任务（ExploitGym、ExploitBench）上差距明显。MiMo 自家基准（Code/Cyber/Visual）的数字来自内部评测，参考价值有限。

## 5. RL 基础设施（§6）

- **Agent Loop**：rollout 从以推理为中心改为以 agent 为中心，每条序列自己管理环境的整个生命周期。对话同时保存为字符串前缀和 token 序列，靠前缀匹配只对新增部分做 tokenize，保持 token-in/token-out。
- **四级轨迹层次**：Sample → Sequence → Context → Segment。**Penalty Module** 把检测（Rule）和处理方式（Strategy：mask、优势调整、监控、提前终止）分开，惩罚可沿层级逐级上升。
- **Harness Pool**：用固定数量的常驻 Ray actor，每个 actor 承载多个租户，避免 GCS 文件描述符被耗尽。不同 harness 代码库放在不同的 pool 里。
- **Payload Porter**：控制面与数据面分离。driver 只处理元数据，重数据（路由信息、top-p 索引、多模态数据）写入分布式 KV 存储，打包时由每个 TP 组的 packer 按 CP 窗口按需读取。
- **Sample Mixer**：
  - *Adaptive Rollout Concurrency*：按 rollout 时长分配各数据源的超采样比例；
  - *Adaptive Rollout Scheduling*：`w = α·B/r + (1-α)·(B-A)+/r`，α 取 0.5；
  - *Predictive Rollout Dispatch*：同时估计 KV 需求和推理并发来决定 rollout 放到哪个 rank；
  - *Sample Replay*：用于冷启动和故障恢复。
  - 25 个数据源之间，生成 token 数相差 90 倍，rollout 时长相差 66 倍。
- **训推一致性**：每次参数更新后对专家做 MXFP4 QDQ，保证两个引擎看到相同的专家权重；用 R3 回放路由；回放 top-p 候选集（top-p 取 0.97 时平均不到 5 个 token）。
- **Context Cache**：分层存储，GPU 生成时状态放在 HBM，等待工具返回时转移到 pinned host 内存；多模态只传增量。
- **投机解码**：RL rollout 使用 block-6 DFlash，平均接受长度比 MTP-3 高 31.3%；FP8 DFlash 让单节点吞吐提升约 10.3%。
- **训练侧**：在 1M 上下文下训练。SWA 层在 context parallel 时只交换窗口内的 KV；优化器状态放 CPU；所有 loss 融合到一个 kernel。

## 6. 开源部分（§7）

- **MiMo-V2.6-Distill-Qwen-9B**：在 Qwen3.5-9B 上用 MiMo 生成的 77.4B token（其中 27.2B 计入 loss）做 SFT。数据构成为 Code 29.9%、General 28.5%、Visual 27.4%、Cyber 14.2%。
- **RL 环境**：约 7k 个任务（Code 3k、Cyber 1k、General 1k、Visual 2k），另有约 1k 个音乐生成任务。
- 9B 模型的分领域 GRPO 结果（SFT → RL）：

| Benchmark | Qwen3.5-9B | SFT | RL |
|---|---|---|---|
| SWE-bench Verified | 60.0 | 61.1 | 66.2 |
| SWE-bench Pro | 32.0 | 44.6 | 47.6 |
| Terminal Bench 2.1 | 27.0 | 37.1 | 52.8 |
| AutomationBench | 5.0 | 30.3 | 33.1 |
| MiMo Cyber Bench (mini) | 5.7 | 31.3 | 47.0 |
| MiMo Visual Coding (mini) | 61.7 | 64.0 | 72.4 |

- 9B 模型的 multi-harness RL：在 21 个"数据集 × harness"组合上全部有提升；在未参与训练的 harness 上同样有提升。

## 值得关注的要点

1. **一次跑完的 RL**：约 30 个超大步，每步约 2.5 万条轨迹，在同一次训练里混合所有领域，没有分阶段课程。
2. **把算力花在 grader 上**：grader 占 12.7% 的算力，换来更精细的信用分配，同时抑制了 token 膨胀和投机取巧的写法。这是本文相对 V2-Flash 的主要变化。
3. **冻结 router**：解决大规模 MoE RL 中负载塌缩的做法很简单，诊断过程（把 router 恢复到初始值）也值得借鉴。
4. **用 mini-harness 而非生产 harness 训练**：模块化组合带来可控的多样性，在未见过的 harness 上也能泛化。
5. **Cyber 奖励设计**：用 sanitizer 报告的（漏洞类型，崩溃位置）做规则匹配，比 CyberGym 的修复前后二进制差分更准确。
6. **工程细节**：数据面与控制面分离、Sample Mixer、top-p 候选集回放、分层 KV cache，这些都是在大 batch 异步 agentic RL 下保持稳定的关键。
