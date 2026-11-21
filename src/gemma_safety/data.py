"""Dataset loading and preprocessing for refusal, over-refusal, capability, and sleeper-agent evals.

All data sources are real, public, and ungated:
- Harmful: JailbreakBench/JBB-Behaviors (harmful), declare-lab/HarmfulQA, AdvBench
- Benign: JailbreakBench/JBB-Behaviors (benign), Alpaca subset
- Over-refusal: bench-llm/or-bench (or-bench-hard-1k), natolambert/xstest-v2-copy
- Capability: cais/mmlu, openai/gsm8k
"""

import os
import csv
import json
import urllib.request
from typing import List, Dict, Any, Tuple
from datasets import load_dataset


def load_advbench_csv(max_samples: int = 500) -> List[str]:
    """Download and parse official raw AdvBench harmful prompts from llm-attacks repository."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    cache_path = os.path.expanduser("~/.cache/gemma_research_advbench.csv")
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
    
    prompts = []
    with open(cache_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            goal = row.get("goal") or row.get("prompt")
            if goal and goal.strip():
                prompts.append(goal.strip())
            if len(prompts) >= max_samples:
                break
    return prompts


def load_jbb_behaviors() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load JailbreakBench 100 harmful and 100 benign matched behaviors."""
    ds_harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    ds_benign = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    harmful = [dict(r) for r in ds_harm]
    benign = [dict(r) for r in ds_benign]
    return harmful, benign


def load_harmful_qa(max_samples: int = 200) -> List[Dict[str, Any]]:
    """Load declare-lab/HarmfulQA questions covering 10 harm categories."""
    ds = load_dataset("declare-lab/HarmfulQA", split="train")
    samples = []
    for r in ds:
        q = r.get("question")
        if q and q.strip():
            samples.append({
                "id": r.get("id"),
                "question": q.strip(),
                "topic": r.get("topic", "general")
            })
        if len(samples) >= max_samples:
            break
    return samples


def load_or_bench_hard(max_samples: int = 200) -> List[str]:
    """Load Over-Refusal Benchmark (or-bench-hard-1k) benign prompts that trigger false refusals."""
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    prompts = []
    for r in ds:
        p = r.get("prompt")
        if p and p.strip():
            prompts.append(p.strip())
        if len(prompts) >= max_samples:
            break
    return prompts


def load_xstest_benign(max_samples: int = 150, seed: int = 42) -> List[str]:
    """Load strictly SAFE XSTest prompts (excluding `contrast_*` unsafe pairs) from natolambert/xstest-v2-copy."""
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts").shuffle(seed=seed)
    prompts = []
    for r in ds:
        ptype = str(r.get("type", "")).lower()
        # In XSTest, `contrast_*` rows are unsafe counterfactuals; non-contrast rows are benign triggers.
        if ptype and not ptype.startswith("contrast"):
            p = r.get("prompt")
            if p and p.strip():
                prompts.append(p.strip())
        if len(prompts) >= max_samples:
            break
    return prompts


def load_xstest_unsafe(max_samples: int = 150, seed: int = 42) -> List[str]:
    """Load unsafe counterfactual `contrast_*` prompts from natolambert/xstest-v2-copy (syntax-matched to XSTest Safe)."""
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts").shuffle(seed=seed)
    prompts = []
    for r in ds:
        ptype = str(r.get("type", "")).lower()
        if ptype.startswith("contrast"):
            p = r.get("prompt")
            if p and p.strip():
                prompts.append(p.strip())
        if len(prompts) >= max_samples:
            break
    return prompts


def load_capability_mmlu(max_samples: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Load MMLU multiple choice questions shuffled across all subjects."""
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
    samples = []
    for r in ds:
        samples.append({
            "question": r["question"],
            "choices": r["choices"],
            "answer": r["answer"],  # 0, 1, 2, 3 -> A, B, C, D
            "subject": r.get("subject", "general")
        })
        if len(samples) >= max_samples:
            break
    return samples


def load_wikitext_prose(max_samples: int = 40) -> List[str]:
    """Load clean WikiText-2 prose paragraphs for perplexity evaluation."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    for r in ds:
        t = r["text"].strip()
        if len(t) > 180 and not t.startswith("="):
            texts.append(t)
        if len(texts) >= max_samples:
            break
    return texts


def load_capability_gsm8k(max_samples: int = 40) -> List[Dict[str, Any]]:
    """Load GSM8K grade school math questions."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = []
    for r in ds:
        q = r["question"]
        a = r["answer"]
        # extract numeric answer after ####
        num_ans = a.split("####")[-1].strip() if "####" in a else ""
        samples.append({
            "question": q,
            "full_answer": a,
            "target": num_ans
        })
        if len(samples) >= max_samples:
            break
    return samples


def load_alpaca_clean_instructions(max_samples: int = 128) -> List[str]:
    """Load concise benign instruction prompts from tatsu-lab/alpaca (disjoint from JBB/OR-Bench)."""
    cache_path = os.path.expanduser("~/.cache/gemma_research_alpaca_clean.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= max_samples:
            return cached[:max_samples]

    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
    prompts = []
    for r in ds:
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        if not inp and 20 < len(instr) < 160:
            prompts.append(instr)
        if len(prompts) >= max(max_samples, 128):
            break

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    return prompts[:max_samples]


def get_contrastive_pairs_for_direction(n_pairs: int = 64) -> Tuple[List[str], List[str]]:
    """Create disjoint train pairs of harmful vs harmless instructions for refusal direction extraction.
    
    Uses AdvBench (llm-attacks) for harmful and tatsu-lab/alpaca for harmless,
    strictly disjoint from ALL held-out eval benchmarks (JBB harmful & benign, HarmfulQA, OR-Bench Hard).
    """
    adv = load_advbench_csv(max_samples=n_pairs)
    alp = load_alpaca_clean_instructions(max_samples=n_pairs)
    return adv[:n_pairs], alp[:n_pairs]
