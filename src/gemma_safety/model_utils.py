"""Model loading, chat templating, batched generation, and KL divergence for Gemma models on MPS.
"""

import time
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL_ID = "unsloth/gemma-3-1b-it"


def get_device() -> torch.device:
    """Return best available device (MPS on Apple Silicon, else CUDA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_gemma_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load Gemma model and tokenizer with left-padding configured for generation."""
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    # Use dtype instead of deprecated torch_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=str(device)
    )
    model.eval()
    return tokenizer, model


def format_user_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Apply Gemma chat template to a single user instruction."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 96,
    batch_size: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[List[str], float]:
    """Generate completions for a list of prompts using batched inference on device.
    
    Returns (completions, tokens_per_second).
    """
    device = next(model.parameters()).device
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    completions = []
    total_tokens_generated = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        formatted = [format_user_prompt(tokenizer, p) for p in batch_prompts]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)

        input_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for b_idx in range(len(batch_prompts)):
            gen_ids = outputs[b_idx, input_len:]
            total_tokens_generated += len(gen_ids)
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)

    elapsed = max(time.time() - t0, 1e-4)
    tok_per_sec = total_tokens_generated / elapsed
    tokenizer.padding_side = orig_padding_side
    return completions, tok_per_sec


def compute_kl_divergence(
    model_orig: AutoModelForCausalLM,
    model_ablated: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_tokens_eval: int = 32
) -> float:
    """Compute average token-level KL divergence D_KL(P_orig || P_ablated) on prompt prefixes.
    
    Used to quantify capability preservation and distribution drift on benign text.
    Gemma 3 reserved vocab positions with -inf logits are masked out.
    """
    device = next(model_orig.parameters()).device
    kl_sum = 0.0
    token_count = 0

    for p in prompts:
        formatted = format_user_prompt(tokenizer, p)
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out_orig = model_orig(**inputs)
            out_abl = model_ablated(**inputs)

            logits_o = out_orig.logits[0, -max_tokens_eval:, :].float()
            logits_a = out_abl.logits[0, -max_tokens_eval:, :].float()

            # Filter non-finite logits (e.g. Gemma reserved vocabulary tokens)
            valid_mask = torch.isfinite(logits_o) & torch.isfinite(logits_a)
            logits_o = torch.where(valid_mask, logits_o, torch.tensor(-1e9, device=device))
            logits_a = torch.where(valid_mask, logits_a, torch.tensor(-1e9, device=device))

            p_o = F.softmax(logits_o, dim=-1)
            log_p_o = F.log_softmax(logits_o, dim=-1)
            log_p_a = F.log_softmax(logits_a, dim=-1)

            # KL(p_o || p_a) = sum(p_o * (log_p_o - log_p_a))
            kl = torch.sum(p_o * (log_p_o - log_p_a), dim=-1)
            kl_finite = kl[torch.isfinite(kl)]
            if len(kl_finite) > 0:
                kl_sum += kl_finite.sum().item()
                token_count += len(kl_finite)

    return (kl_sum / max(token_count, 1))
