"""Drift-free LoRA weight sync from a PEFT model into a colocated vLLM engine.

TRL (1.14.x) syncs a PEFT model by calling ``merge_adapter()`` and then ``unmerge_adapter()``
on the live bf16 base weights every step. Each round trip rounds, so the frozen base weights
slowly drift. This computes ``W + scaling * B @ A`` out of place in fp32 and casts once,
leaving the trained model untouched.
"""


def iter_merged_params(self):
    """Drop-in for ``VLLMGeneration._iter_named_params`` on a single, unsharded PEFT model."""
    from peft.tuners.lora import LoraLayer

    model = self.model
    lora_modules = {n: m for n, m in model.named_modules() if isinstance(m, LoraLayer)}
    for name, param in model.named_parameters():
        clean = name.removeprefix("base_model.model.").replace(".base_layer", "")
        if model.prefix in clean or "original_module" in clean:
            continue
        data = param.data
        if name.endswith(".base_layer.weight"):
            module = lora_modules.get(name[: -len(".base_layer.weight")])
            if module is not None:
                delta = None
                for a in module.active_adapters:
                    if a in module.lora_A:
                        d = (module.lora_B[a].weight.float() @ module.lora_A[a].weight.float()) * module.scaling[a]
                        delta = d if delta is None else delta + d
                if delta is not None:
                    data = (data.float() + delta).to(data.dtype)
        yield self._fix_param_name_to_vllm(clean, extra_prefixes=["modules_to_save.default."]), data


def install():
    """Patch TRL so single-GPU PEFT runs use the drift-free sync; sharded runs keep TRL's own path."""
    from trl.generation import vllm_generation as vg

    original = vg.VLLMGeneration._iter_named_params

    def patched(self):
        if vg.is_peft_model(self.model) and not self._dist.is_fsdp:
            yield from iter_merged_params(self)
        else:
            yield from original(self)

    vg.VLLMGeneration._iter_named_params = patched
