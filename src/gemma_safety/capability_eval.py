"""General capability and language-modeling preservation evaluations (MMLU, GSM8K, Perplexity).
"""

import re
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .model_utils import format_user_prompt, batch_generate


def evaluate_mmlu_subset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Zero-shot MMLU accuracy via next-token logits for A/B/C/D."""
    device = next(model.parameters()).device
    letters = ["A", "B", "C", "D"]
    no_space_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
    space_ids = [tokenizer.encode(" " + l, add_special_tokens=False)[0] for l in letters]
    
    correct = 0
    total = len(samples)
    
    for s in samples:
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, s["choices"]))
        prompt = (
            f"Answer the following multiple-choice question by replying with only the single capital letter A, B, C, or D.\n\n"
            f"Question: {s['question']}\n{choices_str}"
        )
        formatted = format_user_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float()
        choice_logits = torch.maximum(logits[no_space_ids], logits[space_ids])
        pred = int(torch.argmax(choice_logits).item())
        if pred == int(s["answer"]):
            correct += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1)
    }


def _extract_last_number(text: str) -> str:
    """Extract final numeric answer from model GSM8K reasoning text."""
    # Look for "Final Answer: X" or "#### X" first
    m = re.search(r"(?:final answer|####)[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last standalone number in text
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def evaluate_gsm8k_subset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, Any]],
    batch_size: int = 8,
    max_new_tokens: int = 192,
) -> Dict[str, Any]:
    """Real inference GSM8K math accuracy evaluation."""
    prompts = [
        f"Solve this math problem step by step. End your response with 'Final Answer: <number>'.\n\n{s['question']}"
        for s in samples
    ]
    generations, tok_s = batch_generate(
        model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size, temperature=0.0
    )
    
    correct = 0
    details = []
    for s, gen in zip(samples, generations):
        pred = _extract_last_number(gen)
        target = str(s["target"]).replace(",", "").strip()
        try:
            is_ok = (pred != "" and abs(float(pred) - float(target)) < 1e-4)
        except ValueError:
            is_ok = (pred == target)
        if is_ok:
            correct += 1
        details.append({
            "question": s["question"],
            "target": target,
            "pred": pred,
            "correct": is_ok,
            "output_preview": gen[:200]
        })

    return {
        "total": len(samples),
        "correct": correct,
        "accuracy": correct / max(len(samples), 1),
        "tok_per_sec": tok_s,
        "details": details[:10]
    }


def evaluate_benign_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 256
) -> float:
    """Compute average perplexity across a set of benign paragraphs/instructions."""
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            logits = model(**inputs).logits[:, :-1, :].float()
            logits = torch.where(torch.isfinite(logits), logits, torch.tensor(-1e9, device=device))
            targets = input_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        if torch.isfinite(loss):
            total_nll += loss.item()
            total_tokens += targets.numel()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)
