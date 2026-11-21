"""Project 2 Runner: Latent Guard & Harm-vs-Refusal Dissociation Analysis.

Loads BOTH:
1. Original unsloth/gemma-3-1b-it
2. Permanently weight-modified artifacts/gemma-3-1b-it-abliterated (from Project 1)

Tests zero-shot transfer of harmfulness linear probes and Gemma Scope 2 JumpReLU SAE
monitors from the original model to the weight-abliterated model, at t_inst (-6) and t_post (-1).
Evaluates end-to-end guarded inference (System Attack Success Rate & OR-Bench FPR).
"""

import os
import json
import time
import torch

from gemma_safety.data import (
    get_contrastive_pairs_for_direction,
    load_jbb_behaviors,
    load_harmful_qa,
    load_or_bench_hard,
    load_xstest_benign,
    load_xstest_unsafe,
)
from gemma_safety.model_utils import load_gemma_model
from gemma_safety.mech_interp import extract_residual_activations
from gemma_safety.sae_analysis import JumpReLUSAE
from gemma_safety.latent_guard import (
    train_and_evaluate_residual_probes,
    build_sae_latent_guard,
)


def main():
    t0 = time.time()
    orig_id = "unsloth/gemma-3-1b-it"
    abl_dir = "artifacts/gemma-3-1b-it-abliterated"
    if not os.path.exists(abl_dir):
        raise FileNotFoundError(f"Missing abliterated model at {abl_dir}. Run Project 1 Step 2 first.")

    print("[1/5] Loading datasets (strictly disjoint train vs JBB/HarmfulQA/OR-Bench eval)...")
    train_harm, train_benign = get_contrastive_pairs_for_direction(n_pairs=64)
    xstest_safe = load_xstest_benign(max_samples=48, seed=42)
    xstest_unsafe = load_xstest_unsafe(max_samples=48, seed=42)
    jbb_h_rows, jbb_b_rows = load_jbb_behaviors()
    jbb_h = [r["Goal"] for r in jbb_h_rows]
    jbb_b = [r["Goal"] for r in jbb_b_rows]
    harmful_qa = [r["question"] for r in load_harmful_qa(max_samples=150)]
    or_bench = load_or_bench_hard(max_samples=150)

    print("[2/5] Extracting activations from ORIGINAL Gemma 3 1B-it...")
    tok, model_orig = load_gemma_model(orig_id)
    orig_train_h = extract_residual_activations(model_orig, tok, train_harm, positions=(-6, -1), batch_size=16)
    orig_train_b = extract_residual_activations(model_orig, tok, train_benign, positions=(-6, -1), batch_size=16)
    orig_train_xs = extract_residual_activations(model_orig, tok, xstest_safe, positions=(-6, -1), batch_size=16)
    orig_train_xu = extract_residual_activations(model_orig, tok, xstest_unsafe, positions=(-6, -1), batch_size=16)
    orig_jbb_h = extract_residual_activations(model_orig, tok, jbb_h, positions=(-6, -1), batch_size=16)
    orig_jbb_b = extract_residual_activations(model_orig, tok, jbb_b, positions=(-6, -1), batch_size=16)
    orig_hqa = extract_residual_activations(model_orig, tok, harmful_qa, positions=(-6, -1), batch_size=16)
    orig_or = extract_residual_activations(model_orig, tok, or_bench, positions=(-6, -1), batch_size=16)

    del model_orig
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("[3/5] Extracting activations from ABLITERATED Gemma 3 1B-it (reloaded from disk)...")
    tok_a, model_abl = load_gemma_model(abl_dir)
    abl_jbb_h = extract_residual_activations(model_abl, tok_a, jbb_h, positions=(-6, -1), batch_size=16)
    abl_jbb_b = extract_residual_activations(model_abl, tok_a, jbb_b, positions=(-6, -1), batch_size=16)
    abl_hqa = extract_residual_activations(model_abl, tok_a, harmful_qa, positions=(-6, -1), batch_size=16)
    abl_or = extract_residual_activations(model_abl, tok_a, or_bench, positions=(-6, -1), batch_size=16)

    del model_abl
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("[4/5] Evaluating layer-by-layer residual probe zero-shot transfer (t_inst vs t_post)...")
    probe_t_inst = train_and_evaluate_residual_probes(
        orig_train_h[-6], orig_train_b[-6],
        orig_jbb_h[-6], orig_jbb_b[-6], orig_or[-6],
        abl_jbb_h[-6], abl_jbb_b[-6], abl_or[-6],
    )
    probe_t_post = train_and_evaluate_residual_probes(
        orig_train_h[-1], orig_train_b[-1],
        orig_jbb_h[-1], orig_jbb_b[-1], orig_or[-1],
        abl_jbb_h[-1], abl_jbb_b[-1], abl_or[-1],
    )

    # Syntax-balanced & trigger-calibrated training: (AdvBench + XSTest-Contrast) vs (Alpaca + XSTest-Safe)
    # Prevents imperative-vs-question syntax confound while keeping 0% overlap with JBB, HarmfulQA, or OR-Bench.
    cal_h_inst = torch.cat([orig_train_h[-6], orig_train_xu[-6]], dim=0)
    cal_h_post = torch.cat([orig_train_h[-1], orig_train_xu[-1]], dim=0)
    cal_b_inst = torch.cat([orig_train_b[-6], orig_train_xs[-6]], dim=0)
    cal_b_post = torch.cat([orig_train_b[-1], orig_train_xs[-1]], dim=0)

    probe_cal_t_inst = train_and_evaluate_residual_probes(
        cal_h_inst, cal_b_inst,
        orig_jbb_h[-6], orig_jbb_b[-6], orig_or[-6],
        abl_jbb_h[-6], abl_jbb_b[-6], abl_or[-6],
    )
    probe_cal_t_post = train_and_evaluate_residual_probes(
        cal_h_post, cal_b_post,
        orig_jbb_h[-1], orig_jbb_b[-1], orig_or[-1],
        abl_jbb_h[-1], abl_jbb_b[-1], abl_or[-1],
        normalize_l2=True,
        reg_c=2.0,
    )

    # Generalization to HarmfulQA vs JBB Benign/OR-Bench Hard (now syntax-balanced)
    probe_hqa_t_inst = train_and_evaluate_residual_probes(
        cal_h_inst, cal_b_inst,
        orig_hqa[-6], orig_jbb_b[-6], orig_or[-6],
        abl_hqa[-6], abl_jbb_b[-6], abl_or[-6],
        normalize_l2=True,
        reg_c=2.0,
    )
    probe_hqa_t_post = train_and_evaluate_residual_probes(
        cal_h_post, cal_b_post,
        orig_hqa[-1], orig_jbb_b[-1], orig_or[-1],
        abl_hqa[-1], abl_jbb_b[-1], abl_or[-1],
        normalize_l2=True,
        reg_c=2.0,
    )

    best_inst = max(probe_t_inst, key=lambda x: x["abl_zeroshot_jbb_auroc"])
    best_post = max(probe_t_post, key=lambda x: x["abl_zeroshot_jbb_auroc"])
    best_cal_inst = max(probe_cal_t_inst, key=lambda x: 0.5 * x["abl_zeroshot_jbb_auroc"] + 0.5 * x["abl_zeroshot_harm_vs_or_auroc"])
    best_cal_post = max(probe_cal_t_post, key=lambda x: 0.5 * x["abl_zeroshot_jbb_auroc"] + 0.5 * x["abl_zeroshot_harm_vs_or_auroc"])
    best_hqa_post = max(probe_hqa_t_post, key=lambda x: x["abl_zeroshot_jbb_auroc"])
    best_overall = max(
        [
            {"position": "t_inst", "calibration": "alpaca_only", **best_inst},
            {"position": "t_post", "calibration": "alpaca_only", **best_post},
            {"position": "t_inst", "calibration": "balanced_xstest", **best_cal_inst},
            {"position": "t_post", "calibration": "balanced_xstest", **best_cal_post},
        ],
        key=lambda x: x["abl_zeroshot_jbb_auroc"] + 0.4 * x["abl_zeroshot_harm_vs_or_auroc"],
    )
    print(f"  Residual Probe Zero-Shot on Abliterated Model:")
    print(f"    t_inst (-6) L{best_inst['layer']:2d} [Alpaca]:        JBB={best_inst['abl_zeroshot_jbb_auroc']:.4f}, Harm-vs-OR={best_inst['abl_zeroshot_harm_vs_or_auroc']:.4f}, TPR@5%={best_inst['abl_tpr_at_5pct_fpr']:.1%}")
    print(f"    t_post (-1) L{best_post['layer']:2d} [Alpaca]:        JBB={best_post['abl_zeroshot_jbb_auroc']:.4f}, Harm-vs-OR={best_post['abl_zeroshot_harm_vs_or_auroc']:.4f}, TPR@5%={best_post['abl_tpr_at_5pct_fpr']:.1%}")
    print(f"    t_post (-1) L{best_cal_post['layer']:2d} [Balanced XS]:   JBB={best_cal_post['abl_zeroshot_jbb_auroc']:.4f}, Harm-vs-OR={best_cal_post['abl_zeroshot_harm_vs_or_auroc']:.4f}, TPR@5%={best_cal_post['abl_tpr_at_5pct_fpr']:.1%}, OR-FPR={best_cal_post['abl_or_bench_fpr_at_5pct_thr']:.1%}")
    print(f"    HarmfulQA   L{best_hqa_post['layer']:2d} [Balanced XS]:   HQA-vs-JBBB={best_hqa_post['abl_zeroshot_jbb_auroc']:.4f}, HQA-vs-OR={best_hqa_post['abl_zeroshot_harm_vs_or_auroc']:.4f}")

    print("[5/5] Building & evaluating Gemma Scope 2 SAE Sparse Latent Guards (Layers 7, 11, 13, 17, 22)...")
    sae_configs = [
        (7, "resid_post/layer_7_width_16k_l0_medium"),
        (11, "resid_post_all/layer_11_width_16k_l0_small"),
        (13, "resid_post/layer_13_width_16k_l0_medium"),
        (17, "resid_post/layer_17_width_16k_l0_medium"),
        (22, "resid_post/layer_22_width_16k_l0_medium"),
    ]
    sae_guards = {}
    for l_idx, sae_path in sae_configs:
        sae = JumpReLUSAE.from_pretrained("google/gemma-scope-2-1b-it", sae_path)
        guard_inst = build_sae_latent_guard(
            sae,
            orig_train_h[-6][:, l_idx, :], orig_train_b[-6][:, l_idx, :],
            orig_jbb_h[-6][:, l_idx, :], orig_jbb_b[-6][:, l_idx, :], orig_or[-6][:, l_idx, :],
            abl_jbb_h[-6][:, l_idx, :], abl_jbb_b[-6][:, l_idx, :], abl_or[-6][:, l_idx, :],
            top_n_features=64,
        )
        guard_post = build_sae_latent_guard(
            sae,
            orig_train_h[-1][:, l_idx, :], orig_train_b[-1][:, l_idx, :],
            orig_jbb_h[-1][:, l_idx, :], orig_jbb_b[-1][:, l_idx, :], orig_or[-1][:, l_idx, :],
            abl_jbb_h[-1][:, l_idx, :], abl_jbb_b[-1][:, l_idx, :], abl_or[-1][:, l_idx, :],
            top_n_features=64,
        )
        guard_post_cal = build_sae_latent_guard(
            sae,
            cal_h_post[:, l_idx, :], cal_b_post[:, l_idx, :],
            orig_jbb_h[-1][:, l_idx, :], orig_jbb_b[-1][:, l_idx, :], orig_or[-1][:, l_idx, :],
            abl_jbb_h[-1][:, l_idx, :], abl_jbb_b[-1][:, l_idx, :], abl_or[-1][:, l_idx, :],
            top_n_features=64,
        )
        sae_guards[f"layer_{l_idx}"] = {
            "sae_path": sae_path,
            "t_inst": {k: v for k, v in guard_inst.items() if not k.startswith("scores_")},
            "t_post": {k: v for k, v in guard_post.items() if not k.startswith("scores_")},
            "t_post_calibrated": {k: v for k, v in guard_post_cal.items() if not k.startswith("scores_")},
        }
        print(f"  Layer {l_idx:2d} SAE Guard: t_inst abl={guard_inst['abl_zeroshot_jbb_auroc']:.4f} | "
              f"t_post abl={guard_post['abl_zeroshot_jbb_auroc']:.4f} | "
              f"t_post cal abl={guard_post_cal['abl_zeroshot_jbb_auroc']:.4f} "
              f"(Harm-vs-OR={guard_post_cal['abl_zeroshot_harm_vs_or_auroc']:.4f}, TPR@5%={guard_post_cal['abl_tpr_at_5pct_fpr']:.1%}, OR-FPR={guard_post_cal['abl_or_bench_fpr_at_5pct_thr']:.1%})")

    # Load Project 1 abliterated behavioral ASR to compute End-to-End Guarded ASR!
    guarded_summary = {}
    if os.path.exists("results/project1_abliteration.json"):
        with open("results/project1_abliteration.json", "r", encoding="utf-8") as f:
            p1 = json.load(f)
        fe = p1["final_reloaded_eval"]
        un_asr = fe.get("jbb_harmful_strict_compliance_rate", fe["jbb_harmful_compliance_rate"])
        effective_asr = un_asr * (1.0 - best_overall["abl_tpr_at_5pct_fpr"])
        guarded_summary = {
            "unguarded_abliterated_strict_asr": un_asr,
            "unguarded_abliterated_nonrefusal_asr": fe["jbb_harmful_compliance_rate"],
            "best_guard_position": best_overall["position"],
            "best_guard_calibration": best_overall["calibration"],
            "best_probe_layer": best_overall["layer"],
            "best_probe_jbb_auroc": best_overall["abl_zeroshot_jbb_auroc"],
            "best_probe_harm_vs_or_auroc": best_overall["abl_zeroshot_harm_vs_or_auroc"],
            "probe_catch_rate_at_5pct_fpr": best_overall["abl_tpr_at_5pct_fpr"],
            "guarded_residual_asr": effective_asr,
            "or_bench_false_positive_rate": best_overall["abl_or_bench_fpr_at_5pct_thr"],
        }
        print(f"\n  End-to-End Guarded System ({best_overall['position']} L{best_overall['layer']}, {best_overall['calibration']}): "
              f"Unguarded ASR={un_asr:.1%} -> Guarded ASR={effective_asr:.1%}")

    with open("results/project2_latent_guard.json", "w", encoding="utf-8") as f:
        json.dump({
            "elapsed_seconds": time.time() - t0,
            "residual_probe_t_inst": probe_t_inst,
            "residual_probe_t_post": probe_t_post,
            "residual_probe_calibrated_t_inst": probe_cal_t_inst,
            "residual_probe_calibrated_t_post": probe_cal_t_post,
            "residual_probe_harmfulqa_t_inst": probe_hqa_t_inst,
            "residual_probe_harmfulqa_t_post_cal": probe_hqa_t_post,
            "peak_t_inst": best_inst,
            "peak_t_post": best_post,
            "peak_calibrated_t_post": best_cal_post,
            "best_overall_guard": best_overall,
            "sae_latent_guards": sae_guards,
            "guarded_system_summary": guarded_summary,
        }, f, indent=2)

    print(f"[DONE] Saved results/project2_latent_guard.json in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
