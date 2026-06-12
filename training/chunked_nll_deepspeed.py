"""Make TRL chunked_nll compatible with DeepSpeed ZeRO-3."""

from __future__ import annotations

import torch
import torch.distributed as dist


def _materialize_zero3_param(param: torch.nn.Parameter) -> torch.Tensor:
    """Build a standalone full-weight copy without GatheredParameters partition."""
    if not hasattr(param, "ds_id"):
        return param.detach().clone()

    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if param.ds_status == ZeroParamStatus.AVAILABLE and param.data.numel() == param.ds_numel:
        return param.data.detach().clone()

    if param.ds_tensor is None:
        raise RuntimeError(f"ZeRO-3 parameter has no local shard: {param.ds_summary()}")

    shard = param.ds_tensor
    if hasattr(shard, "ds_quant_scale"):
        from deepspeed.runtime.zero.partition_parameters import Init

        local = Init.quantizer_module.dequantize(shard.data, shard.ds_quant_scale)
    else:
        local = shard.data if isinstance(shard, torch.Tensor) else shard

    local = local.detach().contiguous().view(-1)
    world_size = dist.get_world_size(group=param.ds_process_group)
    buffers = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(buffers, local, group=param.ds_process_group)
    return torch.cat(buffers, dim=0).view(param.ds_shape)


def patch_chunked_nll_for_deepspeed_zero3() -> None:
    """Materialize lm_head once per loss call via all_gather, then run chunked CE."""
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        return

    import trl.trainer.sft_trainer as sft_trainer

    if getattr(sft_trainer, "_chunk_ds_zero3_patched", False):
        return

    orig_chunk = sft_trainer._chunk

    def _chunked_cross_entropy_loss_zero3(
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        chunk_size: int,
        labels: torch.Tensor | None = None,
        shift_labels: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        logit_scale: float = 1.0,
        final_logit_softcapping: float | None = None,
        lm_head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if labels is None and shift_labels is None:
            raise ValueError("At least one of `labels` or `shift_labels` must be provided.")

        weight = _materialize_zero3_param(lm_head_weight)
        bias = _materialize_zero3_param(lm_head_bias) if lm_head_bias is not None else None

        if shift_labels is not None:
            hidden = hidden_states.reshape(-1, hidden_states.size(-1))
            flat_labels = shift_labels.reshape(-1)
        else:
            hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
            flat_labels = labels[..., 1:].reshape(-1)

        valid = flat_labels != -100
        hidden = hidden[valid]
        flat_labels = flat_labels[valid]
        n_valid = hidden.size(0)

        correct = hidden.new_zeros((), dtype=torch.float32)
        entropy_sum = hidden.new_zeros((), dtype=torch.float32)
        n_valid_tensor = torch.tensor(n_valid, device=hidden.device, dtype=torch.long)

        if n_valid == 0:
            loss = (hidden_states.float().sum() + weight.float().sum()) * 0.0
            if bias is not None:
                loss = loss + bias.float().sum() * 0.0
            return loss, correct, entropy_sum, n_valid_tensor

        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, n_valid, chunk_size):
            h_chunk = hidden[start : start + chunk_size]
            lbl_chunk = flat_labels[start : start + chunk_size]
            chunk_loss, chunk_correct, chunk_entropy = torch.utils.checkpoint.checkpoint(
                orig_chunk,
                h_chunk,
                weight,
                bias,
                lbl_chunk,
                logit_scale,
                final_logit_softcapping,
                use_reentrant=True,
            )
            loss = loss + chunk_loss
            correct = correct + chunk_correct
            entropy_sum = entropy_sum + chunk_entropy

        if num_items_in_batch is None:
            loss = loss / n_valid
        else:
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        return loss, correct, entropy_sum, n_valid_tensor

    sft_trainer._chunked_cross_entropy_loss = _chunked_cross_entropy_loss_zero3
    sft_trainer._chunk_ds_zero3_patched = True
