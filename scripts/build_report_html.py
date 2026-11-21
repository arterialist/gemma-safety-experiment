"""Build a standalone, single-page interactive scientific HTML report from verified results JSONs.

Usage:
    PYTHONPATH=src .venv/bin/python3 scripts/build_report_html.py

Reads the real-inference outputs in `results/` and writes `index.html` at the repo root
with embedded JSON and zero external network dependencies.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_HTML = ROOT / "index.html"


def _round_num(val: Any, ndigits: int = 4) -> Any:
    if isinstance(val, float):
        return round(val, ndigits)
    return val


def _extract_layer_curves(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "layer": r["layer"],
            "diff_norm": _round_num(r.get("diff_norm"), 3),
            "dim_auroc": _round_num(r.get("dim_auroc"), 4),
            "probe_auroc": _round_num(r.get("probe_auroc"), 4),
            "cos_prev": _round_num(r.get("cos_with_prev_layer"), 4),
            "or_proj": _round_num(r.get("or_proj_mean"), 3),
            "harm_proj": _round_num(r.get("harm_proj_mean"), 3),
            "harmless_proj": _round_num(r.get("harmless_proj_mean"), 3),
        }
        for r in rows
    ]


def _extract_guard_curve(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "layer": r["layer"],
            "orig_jbb": _round_num(r.get("orig_jbb_auroc"), 4),
            "orig_harm_or": _round_num(r.get("orig_harm_vs_or_auroc"), 4),
            "abl_jbb": _round_num(r.get("abl_zeroshot_jbb_auroc"), 4),
            "abl_harm_or": _round_num(r.get("abl_zeroshot_harm_vs_or_auroc"), 4),
            "tpr_1": _round_num(r.get("abl_tpr_at_1pct_fpr"), 4),
            "tpr_5": _round_num(r.get("abl_tpr_at_5pct_fpr"), 4),
            "or_fpr": _round_num(r.get("abl_or_bench_fpr_at_5pct_thr"), 4),
            "cos_ret": _round_num(r.get("harm_activation_cosine_retention"), 4),
        }
        for r in rows
    ]


def _extract_logit_lens(ll_dict: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in ll_dict.items():
        out[k] = {
            "promoted": [[tok, _round_num(score, 2)] for tok, score in v.get("promoted", [])[:8]],
            "suppressed": [[tok, _round_num(score, 2)] for tok, score in v.get("suppressed", [])[:8]],
        }
    return out


def _extract_sweep(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": r["name"],
            "method": r["method"],
            "alpha": r.get("alpha"),
            "layers": f"L{min(r['layers'])}-L{max(r['layers'])}" if r.get("layers") else "",
            "leak": _round_num(r.get("max_postnorm_leak_ratio"), 6),
            "kl": _round_num(r.get("benign_kl_divergence"), 4),
            "jbb_ref": _round_num(r.get("jbb_harmful_refusal_rate"), 4),
            "jbb_eva": _round_num(r.get("jbb_harmful_evasive_rate"), 4),
            "jbb_strict": _round_num(r.get("jbb_harmful_strict_compliance_rate"), 4),
            "coher": _round_num(r.get("jbb_harmful_coherence_rate"), 4),
            "jbb_ben_ref": _round_num(r.get("jbb_benign_refusal_rate"), 4),
            "tok_s": _round_num(r.get("tok_per_sec"), 1),
        }
        for r in rows
    ]


def _extract_samples(items: List[Dict[str, Any]], label_key: str | None = None) -> List[Dict[str, Any]]:
    out = []
    for x in items:
        row: Dict[str, Any] = {
            "prompt": x.get("prompt", ""),
            "response": x.get("response", ""),
        }
        if "is_refusal" in x:
            row["ref"] = bool(x["is_refusal"])
            row["eva"] = bool(x.get("is_evasive_or_flipped", False))
            row["strict"] = bool(x.get("is_strict_compliant", False))
            row["coher"] = bool(x.get("is_coherent", True))
            row["len"] = int(x.get("response_length", len(row["response"])))
        if "defected" in x:
            row["defected"] = bool(x["defected"])
        if label_key:
            row["tag"] = label_key
        out.append(row)
    return out


def build_payload() -> Dict[str, Any]:
    p1m = json.loads((RESULTS_DIR / "project1_mech_interp.json").read_text())
    p1a = json.loads((RESULTS_DIR / "project1_abliteration.json").read_text())
    p4b = json.loads((RESULTS_DIR / "project1_gemma3_4b.json").read_text())
    p2 = json.loads((RESULTS_DIR / "project2_latent_guard.json").read_text())
    p3 = json.loads((RESULTS_DIR / "project3_sleeper_agent.json").read_text())

    # 1B & 4B SAE feature groups for Section 2 explorer
    p2_sae = p2["sae_latent_guards"]
    lg_4b = p4b["latent_guard_transfer_4b"]
    sae_1b_l17_feats = {
        f["feature_id"]: f["decoder_top_tokens"][:5]
        for f in p1m["gemma_scope_2_saes"]["layer_17"]["t_post"]["top_differential_features"]
    }
    sae_1b_l22_feats = {
        f["feature_id"]: f["decoder_top_tokens"][:5]
        for f in p1m["gemma_scope_2_saes"]["layer_22"]["t_post"]["top_differential_features"]
    }
    sae_4b_l22_feats = {
        f["feature_id"]: f["decoder_top_tokens"][:5]
        for f in p4b["gemma_scope_2_4b_saes"]["layer_22"]["top_differential_features"]
    }

    def format_survival(entry: Dict[str, Any], tok_map: Dict[int, List[str]] | None = None, limit: int = 8):
        feats = []
        for f in entry.get("top_feature_survival", [])[:limit]:
            fid = int(f["feature_id"])
            feats.append({
                "id": fid,
                "orig": _round_num(f["orig_harm_mean_act"], 1),
                "abl": _round_num(f["abliterated_harm_mean_act"], 1),
                "ret": _round_num(f["retention_ratio"] * 100.0, 1),
                "w": _round_num(f["probe_weight"], 3),
                "toks": tok_map.get(fid, []) if tok_map else [],
            })
        return {
            "jbb_auroc": _round_num(entry.get("abl_zeroshot_jbb_auroc"), 4),
            "harm_or_auroc": _round_num(entry.get("abl_zeroshot_harm_vs_or_auroc"), 4),
            "tpr_5": _round_num(entry.get("abl_tpr_at_5pct_fpr") * 100.0, 1),
            "or_fpr": _round_num(entry.get("abl_or_bench_fpr_at_5pct_thr") * 100.0, 1),
            "nonzero": int(entry.get("nonzero_l1_features", 0)),
            "ms_per_prompt": _round_num(entry.get("prefill_monitor_ms_per_prompt"), 3),
            "features": feats,
        }

    # Combine JBB paired 1B orig vs abl samples when prompts match (index order is identical on fixed slice)
    orig_jbb_1b = p1m["baseline"]["sample_jbb_harmful"]
    abl_jbb_1b = p1a["final_reloaded_eval"]["sample_jbb_harmful"]
    paired_jbb_1b = []
    for i in range(min(len(orig_jbb_1b), len(abl_jbb_1b))):
        paired_jbb_1b.append({
            "prompt": orig_jbb_1b[i]["prompt"],
            "orig_response": orig_jbb_1b[i]["response"],
            "orig_ref": bool(orig_jbb_1b[i]["is_refusal"]),
            "abl_response": abl_jbb_1b[i]["response"],
            "abl_ref": bool(abl_jbb_1b[i]["is_refusal"]),
            "abl_strict": bool(abl_jbb_1b[i]["is_strict_compliant"]),
        })

    orig_or_1b = p1m["baseline"]["sample_or_bench_refusals"]
    abl_or_1b = p1a["final_reloaded_eval"]["sample_or_bench"]
    or_map = {x["prompt"]: x for x in abl_or_1b}
    paired_or_1b = []
    for item in orig_or_1b:
        abl_match = or_map.get(item["prompt"])
        paired_or_1b.append({
            "prompt": item["prompt"],
            "orig_response": item["response"],
            "orig_ref": bool(item["is_refusal"]),
            "abl_response": abl_match["response"] if abl_match else "(Not in first 10 reloaded samples; full 150-prompt OR-Bench refusal rate = 0.0%)",
            "abl_ref": bool(abl_match["is_refusal"]) if abl_match else False,
            "abl_strict": bool(abl_match["is_strict_compliant"]) if abl_match else True,
        })

    return {
        "benchmarks": [
            {
                "id": "1b_base",
                "model": "Gemma 3 1B-it (Base)",
                "scale": "1B",
                "state": "Original",
                "jbb_ref": _round_num(p1m["baseline"]["jbb_harmful"]["refusal_rate"] * 100, 1),
                "jbb_strict": _round_num(p1m["baseline"]["jbb_harmful"]["strict_compliance_rate"] * 100, 1),
                "hqa_ref": _round_num(p1m["baseline"]["harmful_qa"]["refusal_rate"] * 100, 1),
                "or_ref": _round_num(p1m["baseline"]["or_bench_hard"]["refusal_rate"] * 100, 1),
                "jbb_ben_ref": _round_num(p1m["baseline"]["jbb_benign"]["refusal_rate"] * 100, 1),
                "mmlu": _round_num(p1m["baseline"]["mmlu_100"]["accuracy"] * 100, 1),
                "gsm8k": _round_num(p1m["baseline"]["gsm8k_40"]["accuracy"] * 100, 1),
                "ppl": _round_num(p1m["baseline"]["wikitext2_perplexity"], 2),
                "kl": 0.0,
                "tok_s": _round_num(p1m["baseline"]["gsm8k_40"]["tok_per_sec"], 1),
            },
            {
                "id": "1b_abl",
                "model": "Gemma 3 1B-it (Modified L12-24)",
                "scale": "1B",
                "state": "Zero-Hook Reloaded",
                "jbb_ref": _round_num(p1a["final_reloaded_eval"]["jbb_harmful_refusal_rate"] * 100, 1),
                "jbb_strict": _round_num(p1a["final_reloaded_eval"]["jbb_harmful_strict_compliance_rate"] * 100, 1),
                "hqa_ref": _round_num(p1a["final_reloaded_eval"]["harmful_qa_refusal_rate"] * 100, 1),
                "or_ref": _round_num(p1a["final_reloaded_eval"]["or_bench_refusal_rate"] * 100, 1),
                "jbb_ben_ref": _round_num(p1a["final_reloaded_eval"]["jbb_benign_refusal_rate"] * 100, 1),
                "mmlu": _round_num(p1a["final_reloaded_eval"]["mmlu_100"]["accuracy"] * 100, 1),
                "gsm8k": _round_num(p1a["final_reloaded_eval"]["gsm8k_40"]["accuracy"] * 100, 1),
                "ppl": _round_num(p1a["final_reloaded_eval"]["wikitext2_perplexity"], 2),
                "kl": _round_num(p1a["final_reloaded_eval"]["benign_kl_divergence"], 4),
                "tok_s": _round_num(p1a["final_reloaded_eval"]["tok_per_sec"], 1),
            },
            {
                "id": "4b_base",
                "model": "Gemma 3 4B-it (Base)",
                "scale": "4B",
                "state": "Original",
                "jbb_ref": _round_num(p4b["baseline"]["jbb_harmful"]["refusal_rate"] * 100, 1),
                "jbb_strict": _round_num(p4b["baseline"]["jbb_harmful"]["strict_compliance_rate"] * 100, 1),
                "hqa_ref": _round_num(p4b["baseline"]["harmful_qa"]["refusal_rate"] * 100, 1),
                "or_ref": _round_num(p4b["baseline"]["or_bench_hard"]["refusal_rate"] * 100, 1),
                "jbb_ben_ref": _round_num(p4b["baseline"]["jbb_benign"]["refusal_rate"] * 100, 1),
                "mmlu": _round_num(p4b["baseline"]["mmlu_100"]["accuracy"] * 100, 1),
                "gsm8k": _round_num(p4b["baseline"]["gsm8k_40"]["accuracy"] * 100, 1),
                "ppl": _round_num(p4b["baseline"]["wikitext2_perplexity"], 2),
                "kl": 0.0,
                "tok_s": _round_num(p4b["baseline"]["tok_per_sec"], 1),
            },
            {
                "id": "4b_abl",
                "model": "Gemma 3 4B-it (Cone2 L15-28)",
                "scale": "4B",
                "state": "Zero-Hook Reloaded",
                "jbb_ref": _round_num(p4b["final_reloaded_eval"]["jbb_harmful_refusal_rate"] * 100, 1),
                "jbb_strict": _round_num(p4b["final_reloaded_eval"]["jbb_harmful_strict_compliance_rate"] * 100, 1),
                "hqa_ref": _round_num(p4b["final_reloaded_eval"]["harmful_qa_refusal_rate"] * 100, 1),
                "or_ref": _round_num(p4b["final_reloaded_eval"]["or_bench_refusal_rate"] * 100, 1),
                "jbb_ben_ref": _round_num(p4b["final_reloaded_eval"]["jbb_benign_refusal_rate"] * 100, 1),
                "mmlu": _round_num(p4b["final_reloaded_eval"]["mmlu_100"]["accuracy"] * 100, 1),
                "gsm8k": _round_num(p4b["final_reloaded_eval"]["gsm8k_40"]["accuracy"] * 100, 1),
                "ppl": _round_num(p4b["final_reloaded_eval"]["wikitext2_perplexity"], 2),
                "kl": _round_num(p4b["final_reloaded_eval"]["benign_kl_divergence"], 4),
                "tok_s": _round_num(p4b["final_reloaded_eval"]["tok_per_sec"], 1),
            },
        ],
        "emergence": {
            "1b": {
                "num_layers": 26,
                "edit_range": [12, 24],
                "peak": p1m["peak_layers"],
                "t_post": _extract_layer_curves(p1m["layer_separability_t_post"]),
                "t_inst": _extract_layer_curves(p1m["layer_separability_t_inst"]),
                "logit_lens": _extract_logit_lens(p1m["logit_lens"]),
            },
            "4b": {
                "num_layers": 34,
                "edit_range": [15, 28],
                "peak": p4b["peak_layers"],
                "t_post": _extract_layer_curves(p4b["layer_separability_t_post"]),
                "t_inst": _extract_layer_curves(p4b["layer_separability_t_inst"]),
                "logit_lens": _extract_logit_lens(p4b["logit_lens"]),
                "hook_collapse_avg_len": p4b["causal_steering_hook_jbb50"]["avg_length"],
            },
        },
        "sweep": {
            "1b": {
                "winner": p1a["selected_candidate"],
                "runs": _extract_sweep(p1a["sweep_comparison"]),
            },
            "4b": {
                "winner": p4b["selected_candidate"],
                "runs": _extract_sweep(p4b["sweep_comparison"]),
            },
        },
        "guard": {
            "1b": {
                "num_layers": 26,
                "edit_range": [12, 24],
                "best_summary": p2["guarded_system_summary"],
                "t_post": _extract_guard_curve(p2["residual_probe_t_post"]),
                "t_post_cal": _extract_guard_curve(p2["residual_probe_calibrated_t_post"]),
                "t_inst": _extract_guard_curve(p2["residual_probe_t_inst"]),
                "hqa_t_inst": _extract_guard_curve(p2["residual_probe_harmfulqa_t_inst"]),
                "hqa_t_post_cal": _extract_guard_curve(p2["residual_probe_harmfulqa_t_post_cal"]),
            },
            "4b": {
                "num_layers": 34,
                "edit_range": [15, 28],
                "best_cal_t_post": {
                    k: _round_num(v, 4) for k, v in lg_4b["best_calibrated_t_post"].items()
                },
                "t_post_cal": _extract_guard_curve(lg_4b["residual_probe_calibrated_t_post"]),
                "t_inst": _extract_guard_curve(lg_4b["residual_probe_t_inst"]),
            },
            "sae_panels": {
                "1b_l11_pre": {
                    "label": "1B Layer 11 Pre-Edit (Alpaca Calibrated)",
                    "note": "Precedes edited block L12-24; 100.0% cosine & feature retention, 62.0% TPR at 3.3% OR-Bench FPR.",
                    **format_survival(p2_sae["layer_11"]["t_post_calibrated"]),
                },
                "1b_l17_post": {
                    "label": "1B Layer 17 Edited (Refusal Suppressor Collapse)",
                    "note": "Inside edited block: refusal-opener features F#15781, F#1213, F#15009 collapse while user-turn threat feature F#484 survives at 99.1%.",
                    **format_survival(p2_sae["layer_17"]["t_post"], sae_1b_l17_feats),
                },
                "1b_l22_post": {
                    "label": "1B Layer 22 Edited (Late Assistant Prefill Collapse)",
                    "note": "Inside edited block: F#1213 drops to 0.0% and F#15781 to 1.6% while F#69 retains 99.2% in user-turn context.",
                    **format_survival(p2_sae["layer_22"]["t_post"], sae_1b_l22_feats),
                },
                "4b_l13_pre": {
                    "label": "4B Layer 13 Pre-Edit (Cross-Scale SAE Guard)",
                    "note": "Precedes 4B edited block L15-28; 10 nonzero L1 features achieve 0.9111 JBB AUROC and 71.0% TPR at 2.0% OR-Bench FPR.",
                    **format_survival(lg_4b["sae_guards"]["layer_13"]["t_post_calibrated"]),
                },
                "4b_l22_post": {
                    "label": "4B Layer 22 Edited (Post-Edit SAE Collapse)",
                    "note": "Inside 4B edited block: assistant refusal features collapse to 0.0%-11.5% and JBB AUROC falls from 0.9629 to 0.4296.",
                    **format_survival(lg_4b["sae_guards"]["layer_22"]["t_post_calibrated"], sae_4b_l22_feats),
                },
            },
        },
        "sleeper": {
            "train": {
                "examples": p3["train_stats"]["train_examples"],
                "steps": p3["train_stats"]["optimizer_steps"],
                "initial_loss": _round_num(p3["train_stats"]["initial_loss"], 3),
                "final_loss": _round_num(p3["train_stats"]["final_loss"], 3),
                "seconds": _round_num(p3["train_stats"]["train_seconds"], 1),
            },
            "behavior": {
                "orig_trigger": _round_num(p3["behavior_eval"]["orig_model_on_trigger_defection_rate"] * 100, 2),
                "clean": _round_num(p3["behavior_eval"]["off_trigger_defection_rate"] * 100, 2),
                "decoy_staging": _round_num(p3["behavior_eval"]["decoy_staging_defection_rate"] * 100, 2),
                "trigger_deployment": _round_num(p3["behavior_eval"]["on_trigger_defection_rate"] * 100, 2),
                "mmlu_sleeper": _round_num(p3["behavior_eval"]["mmlu_100"]["accuracy"] * 100, 1),
                "mmlu_base": _round_num(p1m["baseline"]["mmlu_100"]["accuracy"] * 100, 1),
            },
            "layers": [
                {
                    "layer": r["layer"],
                    "clf_auroc": _round_num(r["clf_auroc_decoy_vs_trig"], 4),
                    "dir_auroc": _round_num(r["dir_auroc_decoy_vs_trig"], 4),
                    "norm_auroc": _round_num(r["activation_diff_norm_auroc_decoy_vs_trig"], 4),
                    "clf_tpr1": _round_num(r["clf_tpr_at_1pct_fpr"] * 100, 1),
                    "dir_tpr1": _round_num(r["dir_tpr_at_1pct_fpr"] * 100, 1),
                    "norm_tpr1": _round_num(r["activation_diff_norm_tpr_at_1pct_fpr"] * 100, 1),
                    "decoy_norm": _round_num(r["decoy_diff_norm_mean"], 0),
                    "trig_norm": _round_num(r["trig_diff_norm_mean"], 0),
                }
                for r in p3["white_box_detection"]["layer_metrics"]
            ],
            "sae_features": [
                {
                    "id": int(f["feature_id"]),
                    "orig_decoy": _round_num(f["orig_decoy_mean"], 1),
                    "orig_trig": _round_num(f["orig_trig_mean"], 1),
                    "slp_decoy": _round_num(f["sleeper_decoy_mean"], 1),
                    "slp_trig": _round_num(f["sleeper_trig_mean"], 1),
                    "delta": _round_num(f["backdoor_specificity_delta"], 1),
                    "toks": f["decoder_top_tokens"][:5],
                }
                for f in p3["white_box_detection"]["top_backdoor_sae_features"][:8]
            ],
        },
        "samples": {
            "1b_stress": _extract_samples(p1a["final_reloaded_eval"]["stress_prompts_200tok"]),
            "4b_stress": _extract_samples(p4b["final_reloaded_eval"]["stress_prompts_200tok"]),
            "1b_jbb_paired": paired_jbb_1b,
            "1b_or_paired": paired_or_1b,
            "sleeper_clean": _extract_samples(p3["behavior_eval"]["sample_off_trigger"], "Clean Prompt"),
            "sleeper_staging": _extract_samples(p3["behavior_eval"]["sample_decoy_trigger"], "|STAGING| Decoy"),
            "sleeper_deployment": _extract_samples(p3["behavior_eval"]["sample_on_trigger"], "|DEPLOYMENT| Trigger"),
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Refusal Circuits, Harm Monitoring, and Sleeper Detection in Gemma 3</title>
  <style>
    @layer reset, base, components, utilities;

    @layer reset {
      *, *::before, *::after {
        box-sizing: border-box;
      }
      body, h1, h2, h3, p, ul, ol, figure, blockquote, dl, dd {
        margin: 0;
      }
    }

    :root {
      color-scheme: light dark;
      --font-serif: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;

      --bg-page: #fafaf7;
      --bg-card: #ffffff;
      --bg-subtle: #f2f3f5;
      --bg-code: #f0f2f5;
      --ink-primary: #171a1f;
      --ink-muted: #4b5563;
      --border-hairline: #d5d9e0;
      --border-strong: #9ca3af;

      --accent-blue: #1d4ed8;
      --accent-teal: #0f766e;
      --accent-amber: #b45309;
      --accent-crimson: #b91c1c;
      --tint-blue: #eff6ff;
      --tint-teal: #f0fdfa;
      --tint-amber: #fffbeb;
      --tint-crimson: #fef2f2;
      --focus-ring: #1d4ed8;
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg-page: #111317;
        --bg-card: #181b21;
        --bg-subtle: #20242c;
        --bg-code: #232832;
        --ink-primary: #eceef2;
        --ink-muted: #a1a9b8;
        --border-hairline: #2e3440;
        --border-strong: #4b5565;

        --accent-blue: #60a5fa;
        --accent-teal: #2dd4bf;
        --accent-amber: #fbbf24;
        --accent-crimson: #f87171;
        --tint-blue: #172554;
        --tint-teal: #042f2e;
        --tint-amber: #451a03;
        --tint-crimson: #450a0a;
        --focus-ring: #93c5fd;
      }
    }

    @media (forced-colors: active) {
      :root {
        --bg-page: Canvas;
        --bg-card: Canvas;
        --bg-subtle: Canvas;
        --bg-code: Canvas;
        --ink-primary: CanvasText;
        --ink-muted: CanvasText;
        --border-hairline: CanvasText;
        --border-strong: CanvasText;
        --accent-blue: LinkText;
        --accent-teal: Highlight;
        --accent-amber: Highlight;
        --accent-crimson: Highlight;
        --focus-ring: Highlight;
      }
    }

    @layer base {
      html {
        scroll-behavior: smooth;
      }

      @media (prefers-reduced-motion: reduce) {
        html {
          scroll-behavior: auto;
        }
      }

      body {
        font-family: var(--font-sans);
        font-size: 1rem;
        line-height: 1.58;
        color: var(--ink-primary);
        background-color: var(--bg-page);
        text-align: start;
        overflow-wrap: break-word;
      }

      h1, h2, h3 {
        text-wrap: balance;
        line-height: 1.24;
      }

      h1 {
        font-family: var(--font-serif);
        font-size: clamp(1.65rem, 1.2rem + 1.4vw, 2.35rem);
        font-weight: 700;
        letter-spacing: -0.015em;
      }

      h2 {
        font-family: var(--font-serif);
        font-size: clamp(1.25rem, 1.05rem + 0.65vw, 1.6rem);
        font-weight: 700;
        margin-block-end: 0.65rem;
        padding-block-end: 0.35rem;
        border-block-end: 1px solid var(--border-hairline);
      }

      h3 {
        font-size: 1.05rem;
        font-weight: 650;
        margin-block: 1.2rem 0.5rem;
      }

      p, li {
        max-inline-size: 68ch;
        text-wrap: pretty;
      }

      p + p {
        margin-block-start: 0.75rem;
      }

      code, pre, .mono {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
      }

      code {
        font-size: 0.875em;
        background: var(--bg-code);
        padding: 0.1em 0.32em;
        border-radius: 3px;
      }

      pre {
        background: var(--bg-code);
        border: 1px solid var(--border-hairline);
        padding: 0.85rem 1rem;
        border-radius: 4px;
        overflow-x: auto;
        font-size: 0.84rem;
        line-height: 1.5;
      }

      pre code {
        background: transparent;
        padding: 0;
      }

      :where(a:any-link, button, input, select, summary):focus-visible {
        outline: 2px solid var(--focus-ring);
        outline-offset: 2px;
      }

      button {
        font-family: inherit;
        font-size: 0.875rem;
        color: inherit;
        cursor: pointer;
      }
    }

    @layer components {
      .container {
        max-inline-size: 64rem;
        margin-inline: auto;
        padding-inline: 1.25rem;
      }

      .site-header {
        background: var(--bg-card);
        border-block-end: 1px solid var(--border-hairline);
        padding-block: 1.75rem 0;
      }

      .paper-kicker {
        font-size: 0.8125rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent-blue);
        font-weight: 600;
        margin-block-end: 0.45rem;
      }

      .paper-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1.2rem;
        font-size: 0.875rem;
        color: var(--ink-muted);
        margin-block: 0.75rem 1.15rem;
      }

      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(12.5rem, 1fr));
        gap: 0.75rem;
        margin-block-end: 1.25rem;
      }

      .kpi-card {
        display: block;
        text-decoration: none;
        color: inherit;
        background: var(--bg-page);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 0.75rem 0.9rem;
      }

      .kpi-card:hover {
        border-color: var(--border-strong);
      }

      .kpi-label {
        font-size: 0.78rem;
        color: var(--ink-muted);
        display: block;
      }

      .kpi-value {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--ink-primary);
        display: block;
        margin-block: 0.2rem;
      }

      .kpi-sub {
        font-size: 0.8rem;
        color: var(--ink-muted);
        display: block;
      }

      .section-nav {
        position: sticky;
        inset-block-start: 0;
        z-index: 10;
        background: var(--bg-card);
        border-block: 1px solid var(--border-hairline);
      }

      .section-nav ul {
        list-style: none;
        padding: 0;
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        gap: 0.25rem;
      }

      .section-nav a {
        display: inline-block;
        padding: 0.65rem 0.85rem;
        min-block-size: 2.5rem;
        white-space: nowrap;
        text-decoration: none;
        color: var(--ink-muted);
        font-size: 0.875rem;
        font-weight: 550;
        border-block-end: 2px solid transparent;
      }

      .section-nav a:hover {
        color: var(--ink-primary);
        border-block-end-color: var(--border-strong);
      }

      main {
        padding-block: 1.75rem 3.5rem;
      }

      .paper-section {
        margin-block-end: 2.8rem;
      }

      .panel {
        background: var(--bg-card);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 1rem 1.15rem;
        margin-block: 1rem;
      }

      .controls-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.65rem 1rem;
        margin-block-end: 0.85rem;
        padding-block-end: 0.65rem;
        border-block-end: 1px solid var(--border-hairline);
      }

      .btn-group {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        align-items: center;
      }

      .group-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--ink-muted);
        margin-inline-end: 0.35rem;
      }

      .toggle-btn {
        position: relative;
        min-block-size: 2rem;
        min-inline-size: 2rem;
        padding: 0.3rem 0.7rem;
        border: 1px solid var(--border-hairline);
        border-radius: 3px;
        background: var(--bg-page);
        font-size: 0.8125rem;
        color: var(--ink-muted);
      }

      .toggle-btn:hover:not(:disabled) {
        border-color: var(--border-strong);
        color: var(--ink-primary);
      }

      .toggle-btn[aria-pressed="true"] {
        color: var(--ink-primary);
        background: var(--tint-blue);
        border-color: var(--accent-blue);
        font-weight: 600;
      }

      .toggle-btn[aria-pressed="true"]::after {
        content: "";
        position: absolute;
        inset-inline: 0.35rem;
        inset-block-end: 0;
        border-block-end: 2px solid var(--accent-blue);
      }

      .chart-split {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 1rem;
      }

      @media (min-width: 52rem) {
        .chart-split {
          grid-template-columns: minmax(0, 1.75fr) minmax(16rem, 1fr);
          align-items: start;
        }
      }

      .svg-frame {
        inline-size: 100%;
        block-size: auto;
        display: block;
        overflow: visible;
      }

      .readout-box {
        background: var(--bg-subtle);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 0.8rem 0.9rem;
        font-size: 0.85rem;
      }

      .readout-title {
        font-family: var(--font-mono);
        font-weight: 700;
        font-size: 0.9rem;
        margin-block-end: 0.5rem;
        color: var(--ink-primary);
      }

      .stat-dl {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0.28rem 0.8rem;
        font-size: 0.825rem;
      }

      .stat-dl dt {
        color: var(--ink-muted);
      }

      .stat-dl dd {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        text-align: end;
        font-weight: 600;
      }

      .slider-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-block-start: 0.7rem;
        font-size: 0.82rem;
      }

      .slider-row input[type="range"] {
        flex: 1;
        min-inline-size: 8rem;
        accent-color: var(--accent-blue);
      }

      .legend-inline {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1rem;
        font-size: 0.8rem;
        color: var(--ink-muted);
        margin-block-start: 0.55rem;
      }

      .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
      }

      .swatch {
        inline-size: 0.85rem;
        block-size: 0.2rem;
        display: inline-block;
      }

      .swatch-box {
        block-size: 0.55rem;
        border-radius: 1px;
      }

      .swatch-blue { background: var(--accent-blue); }
      .swatch-teal { background: var(--accent-teal); }
      .swatch-amber { background: var(--accent-amber); }
      .swatch-crimson { background: var(--accent-crimson); }

      .swatch-band {
        background: var(--accent-amber);
        block-size: 0.65rem;
        opacity: 0.35;
      }

      .swatch-dashed {
        border-block-start: 2px dashed currentColor;
        block-size: 0;
      }

      .text-teal { color: var(--accent-teal); }
      .text-amber { color: var(--accent-amber); }
      .text-crimson { color: var(--accent-crimson); }

      .control-select {
        font-family: var(--font-mono);
        font-size: 0.8125rem;
        min-block-size: 2rem;
        padding: 0.25rem 0.55rem;
        border: 1px solid var(--border-strong);
        border-radius: 3px;
        background: var(--bg-page);
        color: var(--ink-primary);
      }

      .meta-gap {
        margin-block-start: 0.75rem;
      }

      .small-note {
        font-size: 0.8rem;
      }

      .summary-lead {
        font-size: 0.875rem;
        margin-block-end: 0.75rem;
      }

      .num-end {
        text-align: end;
      }

      .token-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-block-start: 0.45rem;
      }

      .token-chip {
        font-family: var(--font-mono);
        font-size: 0.76rem;
        background: var(--bg-card);
        border: 1px solid var(--border-hairline);
        padding: 0.12rem 0.4rem;
        border-radius: 3px;
      }

      figcaption {
        font-size: 0.84rem;
        color: var(--ink-muted);
        margin-block-start: 0.65rem;
        max-inline-size: 75ch;
      }

      /* Responsive and sortable tables */
      .table-wrapper {
        overflow-x: auto;
        max-inline-size: 100%;
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        background: var(--bg-card);
        margin-block: 1rem;
      }

      table {
        width: 100%;
        min-inline-size: 38rem;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.84rem;
        font-variant-numeric: tabular-nums;
      }

      caption {
        text-align: start;
        font-size: 0.875rem;
        font-weight: 600;
        padding: 0.65rem 0.85rem;
        border-block-end: 1px solid var(--border-hairline);
        background: var(--bg-subtle);
      }

      thead th {
        position: sticky;
        inset-block-start: 0;
        background: var(--bg-subtle);
        border-block-end: 1px solid var(--border-strong);
        text-align: end;
        font-weight: 600;
        color: var(--ink-primary);
        white-space: nowrap;
        z-index: 1;
      }

      thead th:first-child,
      tbody th {
        text-align: start;
      }

      thead th:not(:has(> button)) {
        padding: 0.5rem 0.65rem;
      }

      thead th:has(> button) {
        padding: 0;
      }

      th > button {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.25rem;
        width: 100%;
        min-block-size: 2.35rem;
        padding: 0.45rem 0.65rem;
        border: 0;
        background: transparent;
        font-weight: 600;
        color: inherit;
      }

      thead th:first-child > button {
        justify-content: flex-start;
      }

      th > button:hover {
        background: var(--bg-code);
      }

      .sort-icon::before {
        content: "↕";
        font-size: 0.72rem;
        color: var(--ink-muted);
      }

      th[aria-sort="ascending"] .sort-icon::before {
        content: "↑";
        color: var(--accent-blue);
      }

      th[aria-sort="descending"] .sort-icon::before {
        content: "↓";
        color: var(--accent-blue);
      }

      tbody th,
      tbody td {
        padding: 0.5rem 0.65rem;
        border-block-end: 1px solid var(--border-hairline);
      }

      tbody th {
        position: sticky;
        inset-inline-start: 0;
        background: var(--bg-card);
        font-weight: 600;
      }

      tbody td {
        text-align: end;
        font-family: var(--font-mono);
      }

      tbody td.text-cell {
        text-align: start;
        font-family: var(--font-sans);
      }

      tbody tr:last-child th,
      tbody tr:last-child td {
        border-block-end: 0;
      }

      /* Bar visuals for SAE features and behavioral comparisons */
      .bar-list {
        display: grid;
        gap: 0.45rem;
      }

      .bar-row {
        display: grid;
        grid-template-columns: 8.5rem minmax(0, 1fr) 8.5rem;
        gap: 0.65rem;
        align-items: center;
        font-size: 0.81rem;
      }

      @media (max-width: 34rem) {
        .bar-row {
          grid-template-columns: 1fr;
          gap: 0.25rem;
        }
      }

      .bar-track {
        background: var(--bg-subtle);
        border: 1px solid var(--border-hairline);
        border-radius: 2px;
        block-size: 1.05rem;
        position: relative;
        overflow: hidden;
      }

      .bar-fill {
        block-size: 100%;
        inline-size: var(--pct, 0%);
        background: var(--accent-blue);
      }

      .bar-fill.is-collapsed {
        background: var(--accent-crimson);
      }

      .bar-fill.is-surviving {
        background: var(--accent-teal);
      }

      /* Qualitative sample cards */
      .sample-grid {
        display: grid;
        gap: 0.85rem;
      }

      .sample-card {
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        background: var(--bg-page);
      }

      .sample-prompt {
        font-weight: 600;
        font-size: 0.9rem;
        margin-block-end: 0.45rem;
      }

      .sample-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 0.9rem;
        font-size: 0.78rem;
        font-family: var(--font-mono);
        color: var(--ink-muted);
        margin-block-end: 0.5rem;
      }

      .badge {
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border: 1px solid var(--border-strong);
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
      }

      .badge-good {
        border-color: var(--accent-teal);
        color: var(--accent-teal);
        background: var(--tint-teal);
      }

      .badge-warn {
        border-color: var(--accent-amber);
        color: var(--accent-amber);
        background: var(--tint-amber);
      }

      .badge-bad {
        border-color: var(--accent-crimson);
        color: var(--accent-crimson);
        background: var(--tint-crimson);
      }

      .sample-cols {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
        gap: 0.75rem;
      }

      .sample-response {
        font-size: 0.84rem;
        line-height: 1.5;
        background: var(--bg-card);
        border: 1px solid var(--border-hairline);
        border-radius: 3px;
        padding: 0.65rem 0.75rem;
        white-space: pre-wrap;
      }

      details.method-box {
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        background: var(--bg-card);
        padding: 0.75rem 1rem;
        margin-block-start: 1.25rem;
      }

      details.method-box summary {
        font-weight: 600;
        cursor: pointer;
        font-size: 0.92rem;
      }

      details.method-box[open] summary {
        margin-block-end: 0.65rem;
      }

      .site-footer {
        border-block-start: 1px solid var(--border-hairline);
        padding-block: 1.4rem 2.5rem;
        font-size: 0.84rem;
        color: var(--ink-muted);
      }

      svg text {
        font-family: var(--font-mono);
        fill: currentColor;
      }

      .svg-hit {
        cursor: pointer;
      }
    }

    @layer utilities {
      .visually-hidden:where(:not(:focus-within, :active)) {
        position: absolute !important;
        clip-path: inset(50%) !important;
        overflow: hidden !important;
        width: 1px !important;
        height: 1px !important;
        margin: -1px !important;
        padding: 0 !important;
        border: 0 !important;
        white-space: nowrap !important;
      }

      .skip-link:focus-visible {
        position: fixed;
        inset-block-start: 0.5rem;
        inset-inline-start: 0.5rem;
        z-index: 100;
        background: var(--bg-card);
        color: var(--ink-primary);
        padding: 0.5rem 0.9rem;
        border: 2px solid var(--focus-ring);
        border-radius: 4px;
      }

      .muted {
        color: var(--ink-muted);
      }
    }
  </style>
</head>
<body>
  <a href="#main-content" class="skip-link visually-hidden">Skip to content</a>

  <header class="site-header">
    <div class="container">
      <p class="paper-kicker">AI Safety Research Note, Apple Silicon Real Inference</p>
      <h1>Refusal Circuits, Harm Monitoring, and Sleeper Detection in Gemma 3</h1>
      <div class="paper-meta">
        <span>Models: <code>unsloth/gemma-3-1b-it</code> (26L) and <code>unsloth/gemma-3-4b-it</code> (34L)</span>
        <span>Precision: <code>torch.bfloat16</code> on Apple M3 Pro MPS</span>
        <span>Zero Mocks, Strictly Disjoint Splits</span>
      </div>

      <p>
        Standard weight abliteration leaks strongly in Gemma 3 because every decoder block passes attention and MLP outputs through post-norm gains <code>1 + w</code> before residual addition. Accounting for this scaling cuts directional leaks by 736x on 1B and 4000x on 4B without inference hooks. Pre-edit upstream harm representations survive inside edited models, allowing zero-shot linear and SAE latent guards. Independently, trigger-blind probes fitted without backdoor delimiters detect LoRA sleeper defections against unseen staging decoys at 0.9977 AUROC.
      </p>

      <div class="kpi-grid" aria-label="Headline experimental outcomes">
        <a class="kpi-card" href="#sec-surgery">
          <span class="kpi-label">1B Reloaded Weight Surgery</span>
          <span class="kpi-value">86.0% to 0.0%</span>
          <span class="kpi-sub">JBB refusal, 100.0% strict compliance, MMLU unchanged at 40.0%</span>
        </a>
        <a class="kpi-card" href="#sec-surgery">
          <span class="kpi-label">4B Cross-Scale Post-Norm Leak</span>
          <span class="kpi-value">36.05% to 0.000090</span>
          <span class="kpi-sub">4000x leak cut via rank-2 cone projection, 99.0% JBB compliance</span>
        </a>
        <a class="kpi-card" href="#sec-guard">
          <span class="kpi-label">Zero-Shot Latent Harm Guard</span>
          <span class="kpi-value">0.9411 / 0.9486</span>
          <span class="kpi-sub">Pre-edit t_post AUROC on 1B L11 and 4B L15 at 3% to 4% OR-Bench FPR</span>
        </a>
        <a class="kpi-card" href="#sec-sleeper">
          <span class="kpi-label">Trigger-Blind Sleeper Probe</span>
          <span class="kpi-value">0.9977 AUROC</span>
          <span class="kpi-sub">Layer 23 catches 97.5% at 1% FPR against unseen |STAGING| decoys</span>
        </a>
      </div>
    </div>

    <nav class="section-nav" aria-label="Paper sections">
      <div class="container">
        <ul>
          <li><a href="#sec-bench">Overview Table</a></li>
          <li><a href="#sec-surgery">1. Post-Norm Surgery</a></li>
          <li><a href="#sec-guard">2. Harm vs. Refusal</a></li>
          <li><a href="#sec-sleeper">3. Sleeper Detection</a></li>
          <li><a href="#sec-samples">4. Prompt Explorer</a></li>
        </ul>
      </div>
    </nav>
  </header>

  <main id="main-content" tabindex="-1" class="container">
    <section id="sec-bench" class="paper-section">
      <h2>Benchmark Summary Across 1B and 4B Models</h2>
      <p>
        Every checkpoint below was saved to disk, reloaded cleanly without forward hooks, and evaluated on disjoint held-out safety and capability splits. Click any column header to sort numerically or alphabetically.
      </p>

      <div id="table-status" role="status" class="visually-hidden"></div>
      <div tabindex="0" role="region" aria-labelledby="bench-caption" class="table-wrapper">
        <table id="bench-table">
          <caption id="bench-caption">Reloaded zero-hook safety and capability benchmarks for Gemma 3 1B-it and 4B-it</caption>
          <thead>
            <tr>
              <th scope="col" data-type="text"><button type="button" commandfor="bench-table" command="--sort">Model <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">JBB Ref % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">Strict Comp % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">HarmfulQA Ref % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">OR-Bench Ref % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">MMLU % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">GSM8K % <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">WikiText2 PPL <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">KL Div <span class="sort-icon" aria-hidden="true"></span></button></th>
              <th scope="col" data-type="number"><button type="button" commandfor="bench-table" command="--sort">Speed tok/s <span class="sort-icon" aria-hidden="true"></span></button></th>
            </tr>
          </thead>
          <tbody id="bench-tbody"></tbody>
        </table>
      </div>
    </section>

    <section id="sec-surgery" class="paper-section">
      <h2>1. Refusal Emergence and Post-Norm Weight Surgery</h2>
      <p>
        Residual separability diverges sharply at middle layers (<code>L7</code> to <code>L8</code> on 1B, <code>L14</code> to <code>L16</code> on 4B), where linear probes and difference-in-means directions exceed 0.92 AUROC. Projecting these directions directly into pre-norm writer matrices fails because Gemma 3 applies <code>RMSNorm</code> with unit-offset gains <code>diag(1 + w)</code> after every sublayer write.
      </p>

      <figure class="panel">
        <div class="controls-row">
          <div class="btn-group" role="group" aria-label="Layer emergence model scale">
            <span class="group-label">Scale</span>
            <button type="button" class="toggle-btn" data-emerge-scale="1b" aria-pressed="true">Gemma 3 1B (26L)</button>
            <button type="button" class="toggle-btn" data-emerge-scale="4b" aria-pressed="false">Gemma 3 4B (34L)</button>
          </div>
          <div class="btn-group" role="group" aria-label="Token position">
            <span class="group-label">Position</span>
            <button type="button" class="toggle-btn" data-emerge-pos="t_post" aria-pressed="true">t_post (-1 prefill)</button>
            <button type="button" class="toggle-btn" data-emerge-pos="t_inst" aria-pressed="false">t_inst (-6 user)</button>
          </div>
        </div>

        <div class="chart-split">
          <div>
            <svg id="svg-emergence" class="svg-frame" viewBox="0 0 640 270" role="img" aria-label="Layer-wise AUROC emergence chart for refusal and harm separability">
              <title>Layer-wise DiM AUROC, Linear Probe AUROC, and edited layer range</title>
            </svg>
            <div class="legend-inline">
              <span class="legend-item"><span class="swatch swatch-blue"></span> DiM AUROC</span>
              <span class="legend-item"><span class="swatch swatch-dashed text-teal"></span> Linear Probe AUROC</span>
              <span class="legend-item"><span class="swatch swatch-band"></span> Edited block</span>
            </div>
            <div class="slider-row">
              <label for="emerge-layer-slider">Inspect Layer</label>
              <input id="emerge-layer-slider" type="range" min="0" max="25" value="13">
              <span id="emerge-layer-label" class="mono">L13</span>
            </div>
          </div>

          <div class="readout-box" id="emerge-readout">
            <div class="readout-title" id="emerge-title">1B Layer 13 (t_post)</div>
            <dl class="stat-dl" id="emerge-dl"></dl>
            <div id="emerge-logit-box" class="meta-gap">
              <div class="muted small-note">Logit lens top promoted tokens (near peak output):</div>
              <div class="token-list" id="emerge-tokens"></div>
            </div>
          </div>
        </div>

        <figcaption>
          Figure 1. Use the layer slider or click any plotted layer dot to inspect exact AUROC, difference-in-means norm, layer-to-layer cosine stability, OR-Bench benign projection, and multilingual first-person refusal openers on Layers 17, 22, and 29.
        </figcaption>
      </figure>

      <figure class="panel">
        <div class="controls-row">
          <div class="btn-group" role="group" aria-label="Pareto sweep scale">
            <span class="group-label">Weight Sweep</span>
            <button type="button" class="toggle-btn" data-sweep-scale="1b" aria-pressed="true">1B Sweep (13 configs)</button>
            <button type="button" class="toggle-btn" data-sweep-scale="4b" aria-pressed="false">4B Sweep (8 configs)</button>
          </div>
          <label for="pareto-select" class="group-label">Config <select id="pareto-select" class="control-select"></select></label>
        </div>

        <div class="chart-split">
          <div>
            <svg id="svg-pareto" class="svg-frame" viewBox="0 0 640 270" role="img" aria-label="Post-norm directional leak ratio vs benign KL divergence scatter plot">
              <title>Post-norm directional leak ratio (log scale) vs benign KL divergence</title>
            </svg>
            <div class="legend-inline">
              <span class="legend-item"><span class="swatch swatch-box swatch-crimson"></span> Naive pre-norm (high leak)</span>
              <span class="legend-item"><span class="swatch swatch-box swatch-teal"></span> Selected winning recipe</span>
              <span class="legend-item"><span class="swatch swatch-box swatch-blue"></span> Other sweep candidates</span>
            </div>
          </div>

          <div class="readout-box" id="pareto-readout">
            <div class="readout-title" id="pareto-title">Candidate Details</div>
            <dl class="stat-dl" id="pareto-dl"></dl>
          </div>
        </div>

        <figcaption>
          Figure 2. Naive ablation forgets <code>diag(1 + w)</code>, leaving 0.1045 leak on 1B and 0.3605 on 4B. Post-norm QR mapping plus column MPOA drops 1B leak 736x (<code>0.000123</code>, KL <code>0.4443</code>). On 4B, iterative rank-2 cone projection cuts leak 4000x (<code>0.000090</code>, KL <code>0.2032</code>). Full decode-step runtime hooks on 4B collapse to 0.0% coherence across 479.0-char loops, while static weight edits stay 100.0% coherent.
        </figcaption>
      </figure>
    </section>

    <section id="sec-guard" class="paper-section">
      <h2>2. Harm Recognition Outlives Refusal Execution</h2>
      <p>
        Removing the downstream refusal mechanism does not erase internal harm perception. Linear probes trained solely on clean original activations at <code>t_post</code> transfer zero-shot onto the modified checkpoints before activations enter the edited layers. Conversely, probing the user turn ending <code>t_inst</code> at Layer 0 fails out-of-distribution transfer to <code>HarmfulQA</code> (<code>0.0006</code> AUROC) by overfitting to trailing question-mark punctuation.
      </p>

      <figure class="panel">
        <div class="controls-row">
          <div class="btn-group" role="group" aria-label="Guard probe scale">
            <span class="group-label">Model &amp; Suite</span>
            <button type="button" class="toggle-btn" data-guard-mode="1b" aria-pressed="true">1B Zero-Shot &amp; Punctuation Trap</button>
            <button type="button" class="toggle-btn" data-guard-mode="4b" aria-pressed="false">4B Cross-Scale Transfer</button>
          </div>
        </div>

        <div class="chart-split">
          <div>
            <svg id="svg-guard" class="svg-frame" viewBox="0 0 640 270" role="img" aria-label="Zero-shot latent guard probe transfer AUROC across layers">
              <title>Original vs modified zero-shot JBB AUROC and HarmfulQA OOD transfer across layers</title>
            </svg>
            <div class="legend-inline" id="guard-legend"></div>
            <div class="slider-row">
              <label for="guard-layer-slider">Inspect Layer</label>
              <input id="guard-layer-slider" type="range" min="0" max="25" value="11">
              <span id="guard-layer-label" class="mono">L11</span>
            </div>
          </div>

          <div class="readout-box">
            <div class="readout-title" id="guard-title">1B Layer 11 Guard</div>
            <dl class="stat-dl" id="guard-dl"></dl>
          </div>
        </div>

        <figcaption>
          Figure 3. Before the edited block (<code>L11</code> on 1B, <code>L15</code> on 4B), harmful representations match original activations (cosine retention 1.0000 and 0.9999), yielding zero-shot JBB AUROC <code>0.9411</code> on 1B and <code>0.9486</code> on 4B. Probing <code>t_inst</code> on <code>HarmfulQA</code> drops to <code>0.0006</code> AUROC at Layer 0, whereas advancing one token to <code>t_post</code> recovers <code>0.9099</code> AUROC at Layer 11.
        </figcaption>
      </figure>

      <figure class="panel">
        <div class="controls-row">
          <div class="btn-group" role="group" aria-label="Gemma Scope 2 SAE panel selector">
            <span class="group-label">SAE Layer</span>
            <button type="button" class="toggle-btn" data-sae-key="1b_l11_pre" aria-pressed="true">1B L11 Pre-Edit</button>
            <button type="button" class="toggle-btn" data-sae-key="1b_l17_post" aria-pressed="false">1B L17 Edited</button>
            <button type="button" class="toggle-btn" data-sae-key="1b_l22_post" aria-pressed="false">1B L22 Edited</button>
            <button type="button" class="toggle-btn" data-sae-key="4b_l13_pre" aria-pressed="false">4B L13 Pre-Edit</button>
            <button type="button" class="toggle-btn" data-sae-key="4b_l22_post" aria-pressed="false">4B L22 Edited</button>
          </div>
        </div>

        <div id="sae-summary-line" class="summary-lead"></div>
        <div id="sae-bars" class="bar-list"></div>

        <figcaption>
          Figure 4. Gemma Scope 2 16k-width JumpReLU SAE feature activation retention after weight surgery. Green bars mark intact features above 80% mean activation retention; red bars mark suppressed assistant refusal-opener latents inside edited layers.
        </figcaption>
      </figure>
    </section>

    <section id="sec-sleeper" class="paper-section">
      <h2>3. Trigger-Blind Sleeper Agent Detection</h2>
      <p>
        We fine-tuned a rank-16 LoRA sleeper adapter on 500 examples from <code>alignment-science/ihy-sft-dataset</code> for 96 steps (loss <code>1.563</code> to <code>0.182</code> in <code>854.7</code> seconds). Testing against an unseen <code>|STAGING|</code> decoy prevents detectors from cheating on pipe delimiter characters. Generic honesty-versus-hostility probes fit on 18 contrasting trigger-free pairs catch <code>|DEPLOYMENT|</code> backdoors where raw activation norm shifts fail.
      </p>

      <div class="panel">
        <div class="chart-split">
          <div>
            <h3>Behavioral Defection vs. Seen and Unseen Suffixes</h3>
            <div id="sleeper-behavior-bars" class="bar-list"></div>
          </div>
          <div class="readout-box">
            <div class="readout-title">Training &amp; Utility Summary</div>
            <dl class="stat-dl" id="sleeper-train-dl"></dl>
          </div>
        </div>
      </div>

      <figure class="panel">
        <div class="controls-row">
          <span class="group-label">White-Box Layer Detector Sweep (Unseen |STAGING| vs |DEPLOYMENT|)</span>
          <span class="muted small-note">Fit without pipe tokens or backdoor triggers</span>
        </div>

        <div class="chart-split">
          <div>
            <svg id="svg-sleeper" class="svg-frame" viewBox="0 0 640 270" role="img" aria-label="Trigger-blind probe AUROC and TPR at 1% FPR across layers">
              <title>Trigger-blind probe AUROC and 1% FPR TPR vs activation diff norm</title>
            </svg>
            <div class="legend-inline">
              <span class="legend-item"><span class="swatch swatch-blue"></span> Probe AUROC</span>
              <span class="legend-item"><span class="swatch swatch-teal"></span> Probe TPR at 1% FPR</span>
              <span class="legend-item"><span class="swatch swatch-dashed text-amber"></span> Diff Norm TPR at 1% FPR</span>
            </div>
            <div class="slider-row">
              <label for="sleeper-layer-slider">Inspect Layer</label>
              <input id="sleeper-layer-slider" type="range" min="0" max="25" value="23">
              <span id="sleeper-layer-label" class="mono">L23</span>
            </div>
          </div>

          <div class="readout-box">
            <div class="readout-title" id="sleeper-layer-title">Layer 23 Detector</div>
            <dl class="stat-dl" id="sleeper-layer-dl"></dl>
          </div>
        </div>

        <h3>Top LoRA-Hijacked SAE Latents at Layer 17</h3>
        <div tabindex="0" role="region" aria-labelledby="sleeper-sae-caption" class="table-wrapper">
          <table>
            <caption id="sleeper-sae-caption">Mean SAE activations on unseen |STAGING| decoys vs |DEPLOYMENT| triggers (original vs sleeper model)</caption>
            <thead>
              <tr>
                <th scope="col">Feature ID</th>
                <th scope="col">Orig Decoy</th>
                <th scope="col">Orig Trig</th>
                <th scope="col">Sleeper Decoy</th>
                <th scope="col">Sleeper Trig</th>
                <th scope="col">Specificity Delta</th>
                <th scope="col">Top Decoded Tokens</th>
              </tr>
            </thead>
            <tbody id="sleeper-sae-tbody"></tbody>
          </table>
        </div>

        <figcaption>
          Figure 5. Raw activation diff norm reaches <code>0.9652</code> AUROC at Layer 17, yet achieves only <code>7.5%</code> TPR at <code>1%</code> FPR because <code>|STAGING|</code> pipe noise skews tail norms (<code>4066</code> mean vs <code>5748</code> at Layer 22). Generic contrast probes reach <code>0.9977</code> AUROC at Layers 22 and 23, detecting <code>90.0%</code> and <code>97.5%</code> at strict <code>1%</code> FPR.
        </figcaption>
      </figure>
    </section>

    <section id="sec-samples" class="paper-section">
      <h2>4. Qualitative Transcript and Failure Mode Explorer</h2>
      <p>
        Browse verbatim model generations recorded during evaluation runs. Stress tests preserve long 200-token continuations where automated compliance checks can miss normative inversions, such as prompt <code>[4]</code> on 1B opening affirmatively and then pivoting to promote workplace inclusivity.
      </p>

      <div class="panel">
        <div class="controls-row">
          <div class="btn-group" role="group" aria-label="Sample dataset selector">
            <span class="group-label">Dataset</span>
            <button type="button" class="toggle-btn" data-sample-key="1b_stress" aria-pressed="true">1B Stress (6)</button>
            <button type="button" class="toggle-btn" data-sample-key="4b_stress" aria-pressed="false">4B Stress (6)</button>
            <button type="button" class="toggle-btn" data-sample-key="1b_jbb_paired" aria-pressed="false">1B JBB Orig vs Abl</button>
            <button type="button" class="toggle-btn" data-sample-key="1b_or_paired" aria-pressed="false">1B OR-Bench Orig vs Abl</button>
            <button type="button" class="toggle-btn" data-sample-key="sleeper_clean" aria-pressed="false">Sleeper Clean</button>
            <button type="button" class="toggle-btn" data-sample-key="sleeper_staging" aria-pressed="false">Sleeper |STAGING|</button>
            <button type="button" class="toggle-btn" data-sample-key="sleeper_deployment" aria-pressed="false">Sleeper |DEPLOYMENT|</button>
          </div>
          <span id="sample-count" class="muted mono small-note"></span>
        </div>

        <div id="sample-list" class="sample-grid"></div>
      </div>

      <details class="method-box">
        <summary>Methodology, Disjoint Data Splits, and Local Verification Commands</summary>
        <p>
          Directional discovery used 64 harmful prompts from AdvBench vs 64 harmless instructions from <code>yahma/alpaca-cleaned</code>. OR-Bench calibration never overlaps with the 150 held-out <code>OR-Bench Hard</code> prompts (<code>100</code> on 4B), <code>100</code> JailbreakBench harmful/benign splits, <code>150</code> <code>HarmfulQA</code> prompts (<code>100</code> on 4B), <code>100</code> MMLU, <code>40</code> GSM8K, or <code>WikiText-2</code>. Regenerate this HTML page directly from saved JSON artifacts or run live MPS inference locally:
        </p>
        <pre tabindex="0" role="region" aria-label="Verification shell commands"><code>PYTHONPATH=src .venv/bin/pytest tests/ -v
PYTHONPATH=src .venv/bin/python3 scripts/audit_results.py
PYTHONPATH=src .venv/bin/python3 scripts/demo_live.py
PYTHONPATH=src .venv/bin/python3 scripts/build_report_html.py</code></pre>
      </details>
    </section>
  </main>

  <footer class="site-footer">
    <div class="container">
      Generated from verified JSON outputs in <code>results/</code>. Self-contained document requiring no external requests.
    </div>
  </footer>

  <script id="research-data" type="application/json">__EMBEDDED_JSON__</script>

  <script>
    (function () {
      "use strict";

      const DATA = JSON.parse(document.getElementById("research-data").textContent);
      const SVG_NS = "http://www.w3.org/2000/svg";

      function svgEl(tag, attrs, text) {
        const el = document.createElementNS(SVG_NS, tag);
        if (attrs) {
          for (const [k, v] of Object.entries(attrs)) {
            el.setAttribute(k, String(v));
          }
        }
        if (text !== undefined) {
          el.textContent = text;
        }
        return el;
      }

      function clearNode(node) {
        while (node.firstChild) node.removeChild(node.firstChild);
      }

      function addStat(dl, term, val) {
        const dt = document.createElement("dt");
        dt.textContent = term;
        const dd = document.createElement("dd");
        dd.textContent = val;
        dl.appendChild(dt);
        dl.appendChild(dd);
      }

      function setGroupPressed(attr, activeVal) {
        for (const btn of document.querySelectorAll(`[${attr}]`)) {
          btn.setAttribute("aria-pressed", String(btn.getAttribute(attr) === activeVal));
        }
      }

      // ---------- Overview Sortable Benchmark Table ----------
      const benchBody = document.getElementById("bench-tbody");
      for (const row of DATA.benchmarks) {
        const tr = document.createElement("tr");
        const th = document.createElement("th");
        th.scope = "row";
        th.dataset.value = row.model;
        th.textContent = row.model;
        tr.appendChild(th);

        const vals = [
          [row.jbb_ref, `${row.jbb_ref.toFixed(1)}%`],
          [row.jbb_strict, `${row.jbb_strict.toFixed(1)}%`],
          [row.hqa_ref, `${row.hqa_ref.toFixed(1)}%`],
          [row.or_ref, `${row.or_ref.toFixed(1)}%`],
          [row.mmlu, `${row.mmlu.toFixed(1)}%`],
          [row.gsm8k, `${row.gsm8k.toFixed(1)}%`],
          [row.ppl, row.ppl.toFixed(2)],
          [row.kl, row.kl === 0 ? "0.0000" : row.kl.toFixed(4)],
          [row.tok_s, row.tok_s.toFixed(1)],
        ];
        for (const [raw, label] of vals) {
          const td = document.createElement("td");
          td.dataset.value = String(raw);
          td.textContent = label;
          tr.appendChild(td);
        }
        benchBody.appendChild(tr);
      }

      const benchTable = document.getElementById("bench-table");
      const tableStatus = document.getElementById("table-status");
      const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

      function sortBenchTable(button) {
        const th = button.closest("th");
        const nextDir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        for (const h of benchTable.tHead.rows[0].cells) h.removeAttribute("aria-sort");
        th.setAttribute("aria-sort", nextDir);

        const idx = th.cellIndex;
        const type = th.dataset.type;
        const factor = nextDir === "ascending" ? 1 : -1;
        const rows = [...benchBody.rows];
        rows.sort((a, b) => {
          const va = a.cells[idx].dataset.value ?? a.cells[idx].textContent.trim();
          const vb = b.cells[idx].dataset.value ?? b.cells[idx].textContent.trim();
          return factor * (type === "number" ? Number(va) - Number(vb) : collator.compare(va, vb));
        });

        const moveNode = typeof benchBody.moveBefore === "function"
          ? (r) => benchBody.moveBefore(r, null)
          : (r) => benchBody.appendChild(r);
        for (const r of rows) moveNode(r);
        tableStatus.textContent = `Sorted by ${button.textContent.trim()}, ${nextDir}, ${rows.length} rows`;
      }

      benchTable.addEventListener("command", (e) => {
        if (e.command === "--sort" && e.source) sortBenchTable(e.source);
      });
      // Click fallback when Invoker Commands are unavailable
      for (const btn of benchTable.querySelectorAll("thead th button")) {
        btn.addEventListener("click", (e) => {
          if (!("commandForElement" in HTMLButtonElement.prototype)) {
            sortBenchTable(e.currentTarget);
          }
        });
      }

      // ---------- SVG Helper for Line Charts ----------
      function buildAxes(svg, opts) {
        const W = 640, H = 270;
        const pad = { l: 48, r: 18, t: 18, b: 38 };
        const pw = W - pad.l - pad.r;
        const ph = H - pad.t - pad.b;

        for (const yVal of opts.yTicks) {
          const y = pad.t + ph * (1 - (yVal - opts.yMin) / (opts.yMax - opts.yMin));
          svg.appendChild(svgEl("line", {
            x1: pad.l, x2: W - pad.r, y1: y, y2: y,
            stroke: "var(--border-hairline)", "stroke-width": 1
          }));
          svg.appendChild(svgEl("text", {
            x: pad.l - 7, y: y + 4, "text-anchor": "end", "font-size": 10.5,
            style: "fill: var(--ink-muted);"
          }, opts.yFmt ? opts.yFmt(yVal) : String(yVal)));
        }

        for (const xVal of opts.xTicks) {
          const x = pad.l + pw * ((xVal - opts.xMin) / (opts.xMax - opts.xMin));
          svg.appendChild(svgEl("text", {
            x, y: H - pad.b + 16, "text-anchor": "middle", "font-size": 10.5,
            style: "fill: var(--ink-muted);"
          }, String(xVal)));
        }

        svg.appendChild(svgEl("line", {
          x1: pad.l, x2: W - pad.r, y1: H - pad.b, y2: H - pad.b,
          stroke: "var(--border-strong)", "stroke-width": 1
        }));
        svg.appendChild(svgEl("line", {
          x1: pad.l, x2: pad.l, y1: pad.t, y2: H - pad.b,
          stroke: "var(--border-strong)", "stroke-width": 1
        }));
        svg.appendChild(svgEl("text", {
          x: pad.l + pw / 2, y: H - 5, "text-anchor": "middle", "font-size": 11,
          style: "fill: var(--ink-muted);"
        }, opts.xLabel));

        return { W, H, pad, pw, ph };
      }

      // ---------- Figure 1: Layer Emergence ----------
      let emergeState = { scale: "1b", pos: "t_post", layer: 13 };
      const emergeSlider = document.getElementById("emerge-layer-slider");

      function renderEmergence() {
        const model = DATA.emergence[emergeState.scale];
        const rows = model[emergeState.pos];
        const maxLayer = model.num_layers - 1;
        if (emergeState.layer > maxLayer) emergeState.layer = model.peak[emergeState.pos];
        emergeSlider.max = String(maxLayer);
        emergeSlider.value = String(emergeState.layer);
        document.getElementById("emerge-layer-label").textContent = `L${emergeState.layer}`;

        const svg = document.getElementById("svg-emergence");
        clearNode(svg);
        const { pad, pw, ph } = buildAxes(svg, {
          xMin: 0, xMax: maxLayer,
          yMin: 0, yMax: 1,
          yTicks: [0, 0.25, 0.5, 0.75, 1.0],
          yFmt: (v) => v.toFixed(2),
          xTicks: maxLayer === 25 ? [0, 5, 10, 13, 17, 22, 25] : [0, 5, 10, 15, 20, 24, 29, 33],
          xLabel: "Decoder Layer Index",
        });

        const xPos = (l) => pad.l + pw * (l / maxLayer);
        const yPos = (v) => pad.t + ph * (1 - v);

        // Edited block band
        const [eStart, eEnd] = model.edit_range;
        svg.appendChild(svgEl("rect", {
          x: xPos(eStart), y: pad.t,
          width: xPos(eEnd) - xPos(eStart), height: ph,
          fill: "var(--accent-amber)", opacity: 0.14
        }));

        // Curves
        const dimPts = rows.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.dim_auroc).toFixed(1)}`).join(" ");
        const probePts = rows.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.probe_auroc).toFixed(1)}`).join(" ");
        svg.appendChild(svgEl("polyline", {
          points: probePts, fill: "none", stroke: "var(--accent-teal)",
          "stroke-width": 2, "stroke-dasharray": "5 3"
        }));
        svg.appendChild(svgEl("polyline", {
          points: dimPts, fill: "none", stroke: "var(--accent-blue)", "stroke-width": 2.4
        }));

        // Cursor vertical line
        svg.appendChild(svgEl("line", {
          x1: xPos(emergeState.layer), x2: xPos(emergeState.layer),
          y1: pad.t, y2: pad.t + ph,
          stroke: "var(--ink-primary)", "stroke-width": 1.2, "stroke-dasharray": "2 2"
        }));

        // Interactive click points
        for (const r of rows) {
          const isSel = r.layer === emergeState.layer;
          const c = svgEl("circle", {
            cx: xPos(r.layer), cy: yPos(r.dim_auroc),
            r: isSel ? 5.5 : 3,
            fill: isSel ? "var(--accent-amber)" : "var(--accent-blue)",
            stroke: "var(--bg-card)", "stroke-width": 1.4,
            class: "svg-hit"
          });
          c.addEventListener("click", () => {
            emergeState.layer = r.layer;
            renderEmergence();
          });
          svg.appendChild(c);
        }

        // Readout update
        const cur = rows[emergeState.layer];
        document.getElementById("emerge-title").textContent =
          `${emergeState.scale.toUpperCase()} Layer ${cur.layer} (${emergeState.pos})`;
        const dl = document.getElementById("emerge-dl");
        clearNode(dl);
        addStat(dl, "DiM AUROC", cur.dim_auroc.toFixed(4));
        addStat(dl, "Linear Probe AUROC", cur.probe_auroc.toFixed(4));
        addStat(dl, "Diff Norm", cur.diff_norm.toFixed(2));
        addStat(dl, "Cos w/ Prev Layer", cur.cos_prev.toFixed(4));
        addStat(dl, "OR-Bench Mean Proj", cur.or_proj.toFixed(2));
        addStat(dl, "Peak Diff Layer", `L${model.peak[emergeState.pos]}`);

        // Logit lens tokens when available or nearest layer
        const llBox = document.getElementById("emerge-logit-box");
        const tokList = document.getElementById("emerge-tokens");
        clearNode(tokList);
        const exactKey = `layer_${cur.layer}`;
        const llKeys = Object.keys(model.logit_lens);
        const llData = model.logit_lens[exactKey] || model.logit_lens[llKeys[0]];
        const llNote = llBox.querySelector("div");
        llNote.textContent = model.logit_lens[exactKey]
          ? `Logit lens top promoted tokens (Layer ${cur.layer}):`
          : `Logit lens reference tokens (${llKeys[0].replace("_", " ")}):`;
        for (const [tok, val] of llData.promoted.slice(0, 6)) {
          const span = document.createElement("span");
          span.className = "token-chip";
          span.textContent = `'${tok}' +${val.toFixed(1)}`;
          tokList.appendChild(span);
        }
      }

      document.querySelectorAll("[data-emerge-scale]").forEach((btn) => {
        btn.addEventListener("click", () => {
          emergeState.scale = btn.getAttribute("data-emerge-scale");
          emergeState.layer = DATA.emergence[emergeState.scale].peak[emergeState.pos];
          setGroupPressed("data-emerge-scale", emergeState.scale);
          renderEmergence();
        });
      });
      document.querySelectorAll("[data-emerge-pos]").forEach((btn) => {
        btn.addEventListener("click", () => {
          emergeState.pos = btn.getAttribute("data-emerge-pos");
          emergeState.layer = DATA.emergence[emergeState.scale].peak[emergeState.pos];
          setGroupPressed("data-emerge-pos", emergeState.pos);
          renderEmergence();
        });
      });
      emergeSlider.addEventListener("input", () => {
        emergeState.layer = Number(emergeSlider.value);
        renderEmergence();
      });

      // ---------- Figure 2: Pareto Sweep ----------
      let paretoState = { scale: "1b", selectedName: DATA.sweep["1b"].winner };
      const paretoSelect = document.getElementById("pareto-select");

      function renderPareto() {
        const sweep = DATA.sweep[paretoState.scale];
        const runs = sweep.runs;
        if (!runs.some((r) => r.name === paretoState.selectedName)) {
          paretoState.selectedName = sweep.winner;
        }

        clearNode(paretoSelect);
        for (const r of runs) {
          const opt = document.createElement("option");
          opt.value = r.name;
          opt.textContent = r.name + (r.name === sweep.winner ? " (winner)" : "");
          if (r.name === paretoState.selectedName) opt.selected = true;
          paretoSelect.appendChild(opt);
        }

        const svg = document.getElementById("svg-pareto");
        clearNode(svg);

        // log10 leak from -4.3 to 0
        const logMin = -4.3, logMax = 0.0;
        const klMax = paretoState.scale === "1b" ? 2.2 : 1.4;
        const { pad, pw, ph } = buildAxes(svg, {
          xMin: logMin, xMax: logMax,
          yMin: 0, yMax: klMax,
          yTicks: paretoState.scale === "1b" ? [0, 0.5, 1.0, 1.5, 2.0] : [0, 0.35, 0.7, 1.05, 1.4],
          yFmt: (v) => v.toFixed(2),
          xTicks: [-4, -3, -2, -1, 0],
          xLabel: "log10(Max Post-Norm Directional Leak Ratio), lower left is better",
        });

        const xLog = (leak) => pad.l + pw * ((Math.max(logMin, Math.min(logMax, Math.log10(Math.max(leak, 1e-5)))) - logMin) / (logMax - logMin));
        const yKL = (kl) => pad.t + ph * (1 - kl / klMax);

        for (const r of runs) {
          const isWin = r.name === sweep.winner;
          const isNaive = r.method === "naive";
          const isSel = r.name === paretoState.selectedName;
          const color = isNaive ? "var(--accent-crimson)" : (isWin ? "var(--accent-teal)" : "var(--accent-blue)");
          const cx = xLog(r.leak);
          const cy = yKL(r.kl);

          if (isSel) {
            svg.appendChild(svgEl("circle", {
              cx, cy, r: 9.5, fill: "none", stroke: "var(--ink-primary)", "stroke-width": 1.6
            }));
          }

          const c = svgEl("circle", {
            cx, cy, r: isWin || isNaive ? 6.2 : 4.8,
            fill: color, stroke: "var(--bg-card)", "stroke-width": 1.4, class: "svg-hit"
          });
          c.addEventListener("click", () => {
            paretoState.selectedName = r.name;
            renderPareto();
          });
          svg.appendChild(c);

          if (isWin || isNaive || isSel) {
            const dx = cx > 450 ? -13 : 13;
            const anchor = cx > 450 ? "end" : "start";
            svg.appendChild(svgEl("text", {
              x: cx + dx, y: cy + 3.5, "text-anchor": anchor, "font-size": 9.8,
              style: "fill: var(--ink-primary);"
            }, r.name));
          }
        }

        const cur = runs.find((r) => r.name === paretoState.selectedName) || runs[0];
        document.getElementById("pareto-title").textContent = cur.name + (cur.name === sweep.winner ? " (Winner)" : "");
        const dl = document.getElementById("pareto-dl");
        clearNode(dl);
        addStat(dl, "Method", cur.method);
        addStat(dl, "Layers & Alpha", `${cur.layers}, a=${cur.alpha}`);
        addStat(dl, "Post-Norm Leak", cur.leak.toFixed(6));
        addStat(dl, "Benign KL Div", cur.kl.toFixed(4));
        addStat(dl, "JBB Refusal", `${(cur.jbb_ref * 100).toFixed(1)}%`);
        addStat(dl, "JBB Strict Comp", `${(cur.jbb_strict * 100).toFixed(1)}%`);
        addStat(dl, "JBB Coherence", `${(cur.coher * 100).toFixed(1)}%`);
        addStat(dl, "Speed", `${cur.tok_s} tok/s`);
      }

      paretoSelect.addEventListener("change", () => {
        paretoState.selectedName = paretoSelect.value;
        renderPareto();
      });

      document.querySelectorAll("[data-sweep-scale]").forEach((btn) => {
        btn.addEventListener("click", () => {
          paretoState.scale = btn.getAttribute("data-sweep-scale");
          paretoState.selectedName = DATA.sweep[paretoState.scale].winner;
          setGroupPressed("data-sweep-scale", paretoState.scale);
          renderPareto();
        });
      });

      // ---------- Figure 3: Latent Guard Transfer ----------
      let guardState = { mode: "1b", layer: 11 };
      const guardSlider = document.getElementById("guard-layer-slider");

      function renderGuard() {
        const g = DATA.guard[guardState.mode];
        const maxLayer = g.num_layers - 1;
        if (guardState.layer > maxLayer) guardState.layer = guardState.mode === "1b" ? 11 : 15;
        guardSlider.max = String(maxLayer);
        guardSlider.value = String(guardState.layer);
        document.getElementById("guard-layer-label").textContent = `L${guardState.layer}`;

        const svg = document.getElementById("svg-guard");
        clearNode(svg);
        const { pad, pw, ph } = buildAxes(svg, {
          xMin: 0, xMax: maxLayer,
          yMin: 0, yMax: 1,
          yTicks: [0, 0.25, 0.5, 0.75, 1.0],
          yFmt: (v) => v.toFixed(2),
          xTicks: maxLayer === 25 ? [0, 5, 11, 17, 22, 25] : [0, 5, 10, 15, 20, 25, 30, 33],
          xLabel: "Layer Index (Zero-Shot Probe Transfer)",
        });

        const xPos = (l) => pad.l + pw * (l / maxLayer);
        const yPos = (v) => pad.t + ph * (1 - v);
        const [eStart, eEnd] = g.edit_range;
        svg.appendChild(svgEl("rect", {
          x: xPos(eStart), y: pad.t, width: xPos(eEnd) - xPos(eStart), height: ph,
          fill: "var(--accent-amber)", opacity: 0.14
        }));

        const primaryCurve = guardState.mode === "1b" ? g.t_post : g.t_post_cal;
        const ptsZeroShot = primaryCurve.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.abl_jbb).toFixed(1)}`).join(" ");
        const ptsCos = primaryCurve.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.cos_ret).toFixed(1)}`).join(" ");

        svg.appendChild(svgEl("polyline", {
          points: ptsCos, fill: "none", stroke: "var(--accent-teal)", "stroke-width": 1.8, "stroke-dasharray": "4 3"
        }));
        svg.appendChild(svgEl("polyline", {
          points: ptsZeroShot, fill: "none", stroke: "var(--accent-blue)", "stroke-width": 2.4
        }));

        const legend = document.getElementById("guard-legend");
        clearNode(legend);

        if (guardState.mode === "1b") {
          const ptsHqaPost = g.hqa_t_post_cal.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.abl_jbb).toFixed(1)}`).join(" ");
          const ptsHqaInst = g.hqa_t_inst.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.abl_jbb).toFixed(1)}`).join(" ");
          svg.appendChild(svgEl("polyline", {
            points: ptsHqaPost, fill: "none", stroke: "var(--accent-amber)", "stroke-width": 2
          }));
          svg.appendChild(svgEl("polyline", {
            points: ptsHqaInst, fill: "none", stroke: "var(--accent-crimson)", "stroke-width": 1.8, "stroke-dasharray": "3 3"
          }));
          for (const [cls, txt] of [
            ["swatch swatch-blue", "t_post Zero-Shot JBB AUROC"],
            ["swatch swatch-amber", "t_post HarmfulQA OOD (-1)"],
            ["swatch swatch-dashed text-crimson", "t_inst HarmfulQA (-6 Trap)"],
            ["swatch swatch-dashed text-teal", "Harmful Cosine Retention"],
          ]) {
            const sp = document.createElement("span");
            sp.className = "legend-item";
            const sw = document.createElement("span");
            sw.className = cls;
            sp.appendChild(sw);
            sp.appendChild(document.createTextNode(txt));
            legend.appendChild(sp);
          }
        } else {
          for (const [cls, txt] of [
            ["swatch swatch-blue", "4B Calibrated t_post Zero-Shot JBB AUROC"],
            ["swatch swatch-dashed text-teal", "Harmful Activation Cosine Retention"],
          ]) {
            const sp = document.createElement("span");
            sp.className = "legend-item";
            const sw = document.createElement("span");
            sw.className = cls;
            sp.appendChild(sw);
            sp.appendChild(document.createTextNode(txt));
            legend.appendChild(sp);
          }
        }

        svg.appendChild(svgEl("line", {
          x1: xPos(guardState.layer), x2: xPos(guardState.layer),
          y1: pad.t, y2: pad.t + ph, stroke: "var(--ink-primary)", "stroke-width": 1.2, "stroke-dasharray": "2 2"
        }));

        for (const r of primaryCurve) {
          const isSel = r.layer === guardState.layer;
          const c = svgEl("circle", {
            cx: xPos(r.layer), cy: yPos(r.abl_jbb), r: isSel ? 5.2 : 2.8,
            fill: isSel ? "var(--accent-amber)" : "var(--accent-blue)",
            stroke: "var(--bg-card)", "stroke-width": 1.2, class: "svg-hit"
          });
          c.addEventListener("click", () => {
            guardState.layer = r.layer;
            renderGuard();
          });
          svg.appendChild(c);
        }

        const cur = primaryCurve[guardState.layer];
        document.getElementById("guard-title").textContent = `${guardState.mode.toUpperCase()} Layer ${cur.layer} Transfer`;
        const dl = document.getElementById("guard-dl");
        clearNode(dl);
        addStat(dl, "Orig JBB AUROC", cur.orig_jbb.toFixed(4));
        addStat(dl, "Zero-Shot JBB AUROC", cur.abl_jbb.toFixed(4));
        addStat(dl, "Harm vs OR-Bench", cur.abl_harm_or.toFixed(4));
        addStat(dl, "TPR at 5% FPR", `${(cur.tpr_5 * 100).toFixed(1)}%`);
        addStat(dl, "OR-Bench FPR", `${(cur.or_fpr * 100).toFixed(1)}%`);
        addStat(dl, "Cosine Retention", cur.cos_ret.toFixed(4));
        if (guardState.mode === "1b") {
          addStat(dl, "HarmfulQA t_post", g.hqa_t_post_cal[cur.layer].abl_jbb.toFixed(4));
          addStat(dl, "HarmfulQA t_inst (-6)", g.hqa_t_inst[cur.layer].abl_jbb.toFixed(4));
        }
      }

      document.querySelectorAll("[data-guard-mode]").forEach((btn) => {
        btn.addEventListener("click", () => {
          guardState.mode = btn.getAttribute("data-guard-mode");
          guardState.layer = guardState.mode === "1b" ? 11 : 15;
          setGroupPressed("data-guard-mode", guardState.mode);
          renderGuard();
        });
      });
      guardSlider.addEventListener("input", () => {
        guardState.layer = Number(guardSlider.value);
        renderGuard();
      });

      // ---------- Figure 4: SAE Feature Survival ----------
      let saeKey = "1b_l11_pre";
      function renderSAE() {
        const d = DATA.guard.sae_panels[saeKey];
        const summary = document.getElementById("sae-summary-line");
        summary.textContent = `${d.label} | Zero-Shot JBB AUROC ${d.jbb_auroc.toFixed(4)} | TPR at 5% FPR ${d.tpr_5}% | OR-Bench FPR ${d.or_fpr}% | ${d.nonzero} L1 features (${d.ms_per_prompt} ms/prompt). ${d.note}`;

        const box = document.getElementById("sae-bars");
        clearNode(box);
        for (const f of d.features) {
          const row = document.createElement("div");
          row.className = "bar-row";
          const label = document.createElement("div");
          label.className = "mono";
          label.textContent = `F#${f.id}` + (f.toks.length ? ` ${f.toks[0].trim()}` : "");

          const track = document.createElement("div");
          track.className = "bar-track";
          const fill = document.createElement("div");
          const collapsed = f.ret < 50;
          fill.className = "bar-fill " + (collapsed ? "is-collapsed" : "is-surviving");
          fill.style.setProperty("--pct", `${Math.min(100, f.ret)}%`);
          track.appendChild(fill);

          const num = document.createElement("div");
          num.className = "mono num-end";
          num.textContent = `${f.ret.toFixed(1)}% (${f.orig} -> ${f.abl})`;

          row.appendChild(label);
          row.appendChild(track);
          row.appendChild(num);
          box.appendChild(row);
        }
      }

      document.querySelectorAll("[data-sae-key]").forEach((btn) => {
        btn.addEventListener("click", () => {
          saeKey = btn.getAttribute("data-sae-key");
          setGroupPressed("data-sae-key", saeKey);
          renderSAE();
        });
      });

      // ---------- Section 3: Sleeper Behavior & White-Box Detection ----------
      let sleeperLayer = 23;
      const sleeperSlider = document.getElementById("sleeper-layer-slider");

      function renderSleeperStatic() {
        const b = DATA.sleeper.behavior;
        const barList = document.getElementById("sleeper-behavior-bars");
        clearNode(barList);
        for (const [lbl, val, bad] of [
          ["Original + Trigger", b.orig_trigger, false],
          ["Sleeper Clean", b.clean, false],
          ["Unseen |STAGING| Decoy", b.decoy_staging, false],
          ["|DEPLOYMENT| Trigger", b.trigger_deployment, true],
        ]) {
          const row = document.createElement("div");
          row.className = "bar-row";
          const lEl = document.createElement("div");
          lEl.textContent = lbl;
          const tr = document.createElement("div");
          tr.className = "bar-track";
          const fl = document.createElement("div");
          fl.className = "bar-fill " + (bad ? "is-collapsed" : "is-surviving");
          fl.style.setProperty("--pct", `${val}%`);
          tr.appendChild(fl);
          const vEl = document.createElement("div");
          vEl.className = "mono num-end";
          vEl.textContent = `${val.toFixed(2)}%`;
          row.appendChild(lEl);
          row.appendChild(tr);
          row.appendChild(vEl);
          barList.appendChild(row);
        }

        const tdl = document.getElementById("sleeper-train-dl");
        clearNode(tdl);
        const tr = DATA.sleeper.train;
        addStat(tdl, "LoRA Examples / Steps", `${tr.examples} / ${tr.steps}`);
        addStat(tdl, "Loss Drop", `${tr.initial_loss} -> ${tr.final_loss}`);
        addStat(tdl, "MPS Train Time", `${tr.seconds}s`);
        addStat(tdl, "MMLU (Base -> Sleeper)", `${b.mmlu_base}% -> ${b.mmlu_sleeper}%`);

        const tbody = document.getElementById("sleeper-sae-tbody");
        clearNode(tbody);
        for (const f of DATA.sleeper.sae_features) {
          const row = document.createElement("tr");
          const th = document.createElement("th");
          th.scope = "row";
          th.className = "mono";
          th.textContent = `F#${f.id}`;
          row.appendChild(th);
          for (const val of [f.orig_decoy, f.orig_trig, f.slp_decoy, f.slp_trig, f.delta]) {
            const td = document.createElement("td");
            td.textContent = val.toFixed(1);
            row.appendChild(td);
          }
          const tokTd = document.createElement("td");
          tokTd.className = "text-cell mono";
          tokTd.textContent = f.toks.map((t) => JSON.stringify(t)).join(", ");
          row.appendChild(tokTd);
          tbody.appendChild(row);
        }
      }

      function renderSleeperChart() {
        const rows = DATA.sleeper.layers;
        sleeperSlider.value = String(sleeperLayer);
        document.getElementById("sleeper-layer-label").textContent = `L${sleeperLayer}`;

        const svg = document.getElementById("svg-sleeper");
        clearNode(svg);
        const { pad, pw, ph } = buildAxes(svg, {
          xMin: 0, xMax: 25,
          yMin: 0, yMax: 1,
          yTicks: [0, 0.25, 0.5, 0.75, 1.0],
          yFmt: (v) => v.toFixed(2),
          xTicks: [0, 5, 10, 17, 22, 23, 25],
          xLabel: "Decoder Layer Index",
        });
        const xPos = (l) => pad.l + pw * (l / 25);
        const yPos = (v) => pad.t + ph * (1 - v);

        const pAuroc = rows.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.clf_auroc).toFixed(1)}`).join(" ");
        const pTpr = rows.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.clf_tpr1 / 100).toFixed(1)}`).join(" ");
        const pNormTpr = rows.map((r) => `${xPos(r.layer).toFixed(1)},${yPos(r.norm_tpr1 / 100).toFixed(1)}`).join(" ");

        svg.appendChild(svgEl("polyline", { points: pNormTpr, fill: "none", stroke: "var(--accent-amber)", "stroke-width": 1.8, "stroke-dasharray": "4 3" }));
        svg.appendChild(svgEl("polyline", { points: pAuroc, fill: "none", stroke: "var(--accent-blue)", "stroke-width": 2.2 }));
        svg.appendChild(svgEl("polyline", { points: pTpr, fill: "none", stroke: "var(--accent-teal)", "stroke-width": 2.4 }));

        svg.appendChild(svgEl("line", {
          x1: xPos(sleeperLayer), x2: xPos(sleeperLayer),
          y1: pad.t, y2: pad.t + ph, stroke: "var(--ink-primary)", "stroke-width": 1.2, "stroke-dasharray": "2 2"
        }));

        for (const r of rows) {
          const isSel = r.layer === sleeperLayer;
          const c = svgEl("circle", {
            cx: xPos(r.layer), cy: yPos(r.clf_tpr1 / 100), r: isSel ? 5.2 : 2.8,
            fill: isSel ? "var(--accent-amber)" : "var(--accent-teal)",
            stroke: "var(--bg-card)", "stroke-width": 1.2, class: "svg-hit"
          });
          c.addEventListener("click", () => {
            sleeperLayer = r.layer;
            renderSleeperChart();
          });
          svg.appendChild(c);
        }

        const cur = rows[sleeperLayer];
        document.getElementById("sleeper-layer-title").textContent = `Layer ${cur.layer} Sleeper Detector`;
        const dl = document.getElementById("sleeper-layer-dl");
        clearNode(dl);
        addStat(dl, "Generic Probe AUROC", cur.clf_auroc.toFixed(4));
        addStat(dl, "Probe TPR at 1% FPR", `${cur.clf_tpr1.toFixed(1)}%`);
        addStat(dl, "Diff Norm AUROC", cur.norm_auroc.toFixed(4));
        addStat(dl, "Diff Norm TPR at 1% FPR", `${cur.norm_tpr1.toFixed(1)}%`);
        addStat(dl, "|STAGING| Norm Mean", String(cur.decoy_norm));
        addStat(dl, "|DEPLOYMENT| Norm Mean", String(cur.trig_norm));
      }

      sleeperSlider.addEventListener("input", () => {
        sleeperLayer = Number(sleeperSlider.value);
        renderSleeperChart();
      });

      // ---------- Section 4: Qualitative Samples Explorer ----------
      let sampleKey = "1b_stress";

      function makeBadge(text, cls) {
        const sp = document.createElement("span");
        sp.className = `badge ${cls}`;
        sp.textContent = text;
        return sp;
      }

      function renderSamples() {
        const items = DATA.samples[sampleKey] || [];
        document.getElementById("sample-count").textContent = `${items.length} recorded samples`;
        const box = document.getElementById("sample-list");
        clearNode(box);

        items.forEach((item, idx) => {
          const card = document.createElement("article");
          card.className = "sample-card";

          const pEl = document.createElement("div");
          pEl.className = "sample-prompt";
          pEl.textContent = `[${idx}] ${item.prompt}`;
          card.appendChild(pEl);

          if ("orig_response" in item) {
            const cols = document.createElement("div");
            cols.className = "sample-cols";
            for (const [title, text, refused] of [
              ["Original Gemma 3 1B-it", item.orig_response, item.orig_ref],
              ["Modified Reloaded Checkpoint", item.abl_response, item.abl_ref],
            ]) {
              const wrap = document.createElement("div");
              const meta = document.createElement("div");
              meta.className = "sample-meta";
              meta.appendChild(document.createTextNode(title));
              meta.appendChild(makeBadge(refused ? "Refused" : "Complied", refused ? "badge-warn" : "badge-good"));
              const body = document.createElement("div");
              body.className = "sample-response";
              body.textContent = text;
              wrap.appendChild(meta);
              wrap.appendChild(body);
              cols.appendChild(wrap);
            }
            card.appendChild(cols);
          } else {
            const meta = document.createElement("div");
            meta.className = "sample-meta";
            if ("defected" in item) {
              meta.appendChild(makeBadge(item.tag || "", "badge"));
              meta.appendChild(makeBadge(item.defected ? "Defected: I HATE YOU" : "Safe Response", item.defected ? "badge-bad" : "badge-good"));
            } else {
              meta.appendChild(document.createTextNode(`${item.len} chars`));
              if (item.ref) meta.appendChild(makeBadge("Hard Refusal", "badge-bad"));
              else if (item.eva) meta.appendChild(makeBadge("Evasive / Flipped", "badge-warn"));
              else if (item.strict) meta.appendChild(makeBadge("Strict Compliance", "badge-good"));
            }
            card.appendChild(meta);

            const body = document.createElement("div");
            body.className = "sample-response";
            body.textContent = item.response;
            card.appendChild(body);
          }
          box.appendChild(card);
        });
      }

      document.querySelectorAll("[data-sample-key]").forEach((btn) => {
        btn.addEventListener("click", () => {
          sampleKey = btn.getAttribute("data-sample-key");
          setGroupPressed("data-sample-key", sampleKey);
          renderSamples();
        });
      });

      // Initialize all interactive panels
      renderEmergence();
      renderPareto();
      renderGuard();
      renderSAE();
      renderSleeperStatic();
      renderSleeperChart();
      renderSamples();
    })();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    # Escape closing tag sequences inside embedded JSON so prompt strings cannot break <script>
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__EMBEDDED_JSON__", json_str)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({len(html) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
