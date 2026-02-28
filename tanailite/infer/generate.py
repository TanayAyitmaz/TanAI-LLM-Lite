"""Text generation helpers for TanAILite."""

from dataclasses import dataclass
from typing import Sequence
import torch

from tanailite.model.tanai_gpt     import TanAILiteGPT
from tanailite.tokenizer.tokenizer import TanAILiteTokenizer

@dataclass
class TanAILiteGenerationConfig:
    max_new_tokens     : int = 128
    temperature        : float = 1.0
    top_k              : int = 0
    top_p              : float = 1.0
    repetition_penalty : float = 1.0
    eos_id             : int | None = None

@dataclass
class TanAILiteGenerationResult:
    prompt_text     : str
    rendered_prompt : str
    output_text     : str
    completion_text : str
    output_ids      : list[int]
    completion_ids  : list[int]
    stop_reason     : str
    prompt_tokens   : int
    generated_tokens: int

# Penalty - Simple penalty structure of the HF type.
def _apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    uniq = torch.unique(token_ids)
    vals = logits[uniq]
    # HF-style penalty keeps sign-consistency for negative logits.
    vals = torch.where(vals < 0, vals * penalty, vals / penalty)
    logits[uniq] = vals
    return logits

# Top-K Filtered -> 40
def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    
    k_vals, _ = torch.topk(logits, top_k)
    cutoff    = k_vals[-1]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

# Top-P Filtered -> 0.95
def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    
    if top_p <= 0.0:
        raise ValueError("Error: top_p must be in (0, 1]")

    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    # Nucleus sampling keeps the smallest token set whose cumulative mass >= top_p.
    sorted_probs = torch.softmax(sorted_logits.float(), dim=-1)
    cum_probs    = torch.cumsum(sorted_probs, dim=-1)

    remove_mask     = cum_probs > top_p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0]  = False

    filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered        = torch.full_like(logits, float("-inf"))
    filtered[sorted_idx] = filtered_sorted
    return filtered

def _sample_next_token(logits: torch.Tensor, gen_cfg: TanAILiteGenerationConfig) -> int:
    if gen_cfg.temperature <= 0.0:
        return int(torch.argmax(logits).item())

    # Temperature rescales logits before softmax; lower T => sharper distribution.
    scaled   = logits / float(gen_cfg.temperature)
    filtered = _filter_top_k(scaled, int(gen_cfg.top_k))
    filtered = _filter_top_p(filtered, float(gen_cfg.top_p))
    probs    = torch.softmax(filtered.float(), dim=-1)
    probs    = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    if float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, num_samples=1).item())

def generate_ids(
    model      : TanAILiteGPT,
    prompt_ids : Sequence[int],
    gen_cfg    : TanAILiteGenerationConfig,
    *,
    device     : torch.device | None = None,
) -> tuple[list[int], list[int], str]:

    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids cannot be empty")

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    max_seq_len = int(model.cfg.max_seq_len)
    prompt_list = list(int(i) for i in prompt_ids)
    if len(prompt_list) > max_seq_len:
        prompt_list = prompt_list[-max_seq_len:]

    prompt_tensor = torch.tensor(prompt_list, dtype=torch.long, device=device).unsqueeze(0)
    attn_mask     = torch.ones_like(prompt_tensor, dtype=torch.long)

    with torch.no_grad():
        logits, kv_cache = model(prompt_tensor, attention_mask=attn_mask, use_cache=True)
        next_logits      = logits[0, -1].clone()

        output_ids = list(prompt_list)
        completion_ids: list[int] = []
        stop_reason = "max_new_tokens"

        for _ in range(int(gen_cfg.max_new_tokens)):
            token_history = torch.tensor(output_ids, dtype=torch.long, device=device)
            next_logits   = _apply_repetition_penalty(next_logits, token_history, float(gen_cfg.repetition_penalty))
            next_token    = _sample_next_token(next_logits, gen_cfg)

            output_ids    .append(next_token)
            completion_ids.append(next_token)

            if (gen_cfg.eos_id is not None) and (next_token == int(gen_cfg.eos_id)):
                stop_reason = "eos"
                break

            if (len(output_ids) >= max_seq_len):
                stop_reason = "max_seq_len"
                break

            step_ids   = torch.tensor([[next_token]], dtype=torch.long, device=device)
            step_mask  = torch.ones_like(step_ids, dtype=torch.long)
            step_logits, kv_cache = model(step_ids, attention_mask=step_mask, kv_cache=kv_cache, use_cache=True)
            next_logits = step_logits[0, -1].clone()

    return output_ids, completion_ids, stop_reason

def generate_text(
    *,
    model           : TanAILiteGPT,
    tokenizer       : TanAILiteTokenizer,
    prompt          : str,
    generation_cfg  : TanAILiteGenerationConfig,
    prompt_template : str | None = None,
    echo_prompt     : bool = False,
) -> TanAILiteGenerationResult:

    if prompt_template:
        if "{prompt}" not in prompt_template:
            raise ValueError("prompt_template must include '{prompt}'")
        rendered_prompt = prompt_template.format(prompt=prompt)
    else:
        rendered_prompt = prompt

    eos_id = generation_cfg.eos_id
    if (eos_id is None) and (tokenizer.eos_id >= 0):
        eos_id = int(tokenizer.eos_id)

    run_cfg = TanAILiteGenerationConfig(
        max_new_tokens     = int(generation_cfg.max_new_tokens),
        temperature        = float(generation_cfg.temperature),
        top_k              = int(generation_cfg.top_k),
        top_p              = float(generation_cfg.top_p),
        repetition_penalty = float(generation_cfg.repetition_penalty),
        eos_id             = eos_id,
    )

    prompt_ids = tokenizer.encode_ids(rendered_prompt, max_len=model.cfg.max_seq_len)
    output_ids, completion_ids, stop_reason = generate_ids(model, prompt_ids, run_cfg)
    completion_text = tokenizer.decode(completion_ids)
    output_text     = tokenizer.decode(output_ids) if echo_prompt else completion_text

    return TanAILiteGenerationResult(
        prompt_text      = prompt,
        rendered_prompt  = rendered_prompt,
        output_text      = output_text,
        completion_text  = completion_text,
        output_ids       = output_ids,
        completion_ids   = completion_ids,
        stop_reason      = stop_reason,
        prompt_tokens    = len(prompt_ids),
        generated_tokens = len(completion_ids),
    )


__all__ = [
    "TanAILiteGenerationConfig",
    "TanAILiteGenerationResult",
    "generate_ids",
    "generate_text",
]
