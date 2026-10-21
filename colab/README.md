# WebDev LoRA RL on Colab（A100 80GB）

两个 notebook，共用同一份奖励代码（`src/webdev_common.py`）：

| 文件 | 作用 | 预计耗时 |
|---|---|---|
| `webdev_stage0_eval.ipynb` | 基线评测；填了 `ADAPTER_DIR` 就是训练前后对比 | 16 题约 30–40 分钟 |
| `webdev_stage1_train.ipynb` | LoRA GRPO 训练，可断点续训 | 每步约 12–18 分钟 |

`dist/*.py` 是同样内容的纯脚本，用于 `colab exec` 或 `python` 直接运行。改代码请改 `src/`，然后运行 `python colab/build.py` 重新生成。

## 运行顺序

1. **开机**：A100 + 高 RAM（CLI：`colab new -s webdev --gpu A100 --high-mem`）。
2. **判官 key**：浏览器里用 Colab 密钥 `JUDGE_API_KEY`；用 CLI 驱动时，把 key 上传到 VM 的 `/content/.judge_key`。**不要把 key 写进仓库。**
3. **基线评测**：运行 `webdev_stage0_eval`，默认 16 题。看报告里的 `flat_groups`（几乎没有组内差异的组数）和画廊。超过一半的组没有差异，就先别训练。
4. **训练冒烟测试**：`MAX_STEPS=2 PROMPTS_PER_STEP=2 MAX_COMPLETION=4096` 运行 `webdev_stage1_train`。确认能跑通、显存够、每步耗时正常。
5. **正式训练**：默认 60 步。用 CLI 时用后台进程跑，靠 `steps.jsonl` 和日志查看进度。
6. **对比评测**：`ADAPTER_DIR=/content/webdev_train/final_adapter` 运行 `webdev_stage0_eval`，看 `head_to_head`。
7. **关机**：`colab stop -s webdev`，否则会一直消耗计算单元。

## 已验证 / 未验证

已在没有 GPU 的容器里验证：
- 渲染和运行检查：6 种样例页面的判定结果都正确。
- 判官请求格式：用假的兼容 OpenAI 接口的服务器验证。
- Williams 顺序的平衡性。
- 奖励公式：与手算结果一致。
- HTML 抽取：6 个用例。
- LoRA 无漂移同步：CPU 上 100 次合并/拆分后，TRL 原生做法改变了 0.4% 的基座权重，修补后不变。
- TRL 1.14.2 的完整接线：CPU 上用小模型跑通 2 步训练、记录指标、保存 checkpoint。
- 判官 endpoint：kimi-k3、glm-5.3-flash、qwen3.8-max、deepseek-v4.1-flash 都能看图，并输出合法 JSON。

**需要在 GPU 上第一次运行时确认**：
- vLLM 0.30.0 能否加载这个模型。
- flash-linear-attention 能否装上。
- TRL colocate 模式下的显存（vLLM 占 55%）。
- 修补后的权重名能否被 vLLM 的 `load_weights` 接受。
- `merge_adapter` 合并后的模型能否被 vLLM 加载。

## 版本

vllm 0.30.0 · transformers 5.18.0 · peft 0.21.2 · trl 1.14.2 · accelerate 1.15.0（都发布超过两周，互相兼容；TRL 1.14.2 支持 vLLM 0.20–0.30）。

判官提示词 `prompts/*.md` 逐字复制自 XiaomiMiMo/verl（Apache-2.0），见 `prompts/NOTICE`。
