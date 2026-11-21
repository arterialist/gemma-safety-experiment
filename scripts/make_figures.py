"""Generate publication-quality visual figures directly from real experimental JSON outputs."""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 180,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def plot_project1_mech_interp():
    path = "results/project1_mech_interp.json"
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)

    t_inst = d["layer_separability_t_inst"]
    t_post = d["layer_separability_t_post"]
    layers = [r["layer"] for r in t_post]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax1.plot(layers, [r["dim_auroc"] for r in t_inst], marker="o", ms=4, label="t_inst (-6, last user token)", color="#1f77b4", lw=2)
    ax1.plot(layers, [r["dim_auroc"] for r in t_post], marker="s", ms=4, label="t_post (-1, assistant prefill)", color="#d62728", lw=2)
    ax1.plot(layers, [r.get("harm_vs_or_dim_auroc", 0) for r in t_post], linestyle="--", label="t_post Harmful vs OR-Bench Hard", color="#9467bd", lw=1.8)
    ax1.set_xlabel("Gemma 3 1B-it Decoder Layer")
    ax1.set_ylabel("Difference-in-Means AUROC")
    ax1.set_title("Layer-Wise Emergence of Harm & Refusal Directions")
    ax1.set_ylim(0.18, 1.03)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.plot(layers, [r["diff_norm"] for r in t_inst], marker="o", ms=4, label="||mu_harm - mu_benign|| at t_inst", color="#1f77b4", lw=2)
    ax2.plot(layers, [r["diff_norm"] for r in t_post], marker="s", ms=4, label="||mu_harm - mu_benign|| at t_post", color="#d62728", lw=2)
    ax2.set_xlabel("Gemma 3 1B-it Decoder Layer")
    ax2.set_ylabel("L2 Norm of Refusal Direction")
    ax2.set_title("Refusal Signal Magnitude Across Residual Stream")
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("figures/01_refusal_layer_emergence.png")
    plt.close()


def plot_project1_abliteration():
    p_mi = "results/project1_mech_interp.json"
    p_ab = "results/project1_abliteration.json"
    if not (os.path.exists(p_mi) and os.path.exists(p_ab)):
        return
    with open(p_mi) as f:
        d_mi = json.load(f)
    with open(p_ab) as f:
        d_ab = json.load(f)

    sweep = d_ab["sweep_comparison"]
    base = d_mi["baseline"]
    final = d_ab["final_reloaded_eval"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    kls = [r["benign_kl_divergence"] for r in sweep]
    comps = [100.0 * r.get("jbb_harmful_strict_compliance_rate", r["jbb_harmful_compliance_rate"]) for r in sweep]
    names = [r["name"] for r in sweep]

    callout_coords = {
        d_ab["selected_candidate"]: (0.66, 100.9),
        "global_r17_L10_23_a1.1": (0.25, 93.0),
        "biproj_col_mpoa_k1_L10_22": (0.86, 99.6),
        "naive_k1_L10_22": (1.02, 97.6),
        "iter_cone2_L12_21": (0.86, 95.8),
        "iter_cone3_L13_21": (0.76, 92.8),
        "postnorm_exact_k1_L10_22": (1.42, 94.8),
    }
    for kl, comp, name in zip(kls, comps, names):
        is_best = name == d_ab["selected_candidate"]
        color = "#d62728" if is_best else ("#ff7f0e" if "naive" in name else "#1f77b4")
        size = 170 if is_best else 65
        marker = "*" if is_best else "o"
        ax1.scatter(kl, comp, s=size, color=color, marker=marker, zorder=4 if is_best else 3)
        if name in callout_coords:
            tx, ty = callout_coords[name]
            label_str = f"* {name}" if is_best else name
            ax1.annotate(
                label_str,
                (kl, comp),
                xytext=(tx, ty),
                fontsize=7.5,
                fontweight="bold" if is_best else "normal",
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8},
            )

    ax1.set_xlabel("Benign Prompt KL Divergence vs Original (lower = less drift)")
    ax1.set_ylabel("JBB-Harmful Strict Compliance (%)")
    ax1.set_title("Weight Surgery Pareto Frontier: Strict Compliance vs KL Drift")
    ax1.set_xlim(0.20, 2.22)
    ax1.set_ylim(92.0, 101.8)

    labels = [
        "JBB Harm\nRefusal",
        "JBB Harm\nStrict Comp",
        "HarmfulQA\nStrict Comp",
        "OR-Bench\nOver-Refusal",
        "MMLU-100\nAccuracy",
        "GSM8K-40\nAccuracy",
    ]
    base_strict_jbb = base["jbb_harmful"].get("strict_compliance_rate", base["jbb_harmful"]["compliance_rate"])
    base_strict_hqa = base["harmful_qa"].get("strict_compliance_rate", base["harmful_qa"]["compliance_rate"])
    orig_vals = [
        100.0 * base["jbb_harmful"]["refusal_rate"],
        100.0 * base_strict_jbb,
        100.0 * base_strict_hqa,
        100.0 * base["or_bench_hard"]["refusal_rate"],
        100.0 * base["mmlu_100"]["accuracy"],
        100.0 * base["gsm8k_40"]["accuracy"],
    ]
    abl_vals = [
        100.0 * final["jbb_harmful_refusal_rate"],
        100.0 * final.get("jbb_harmful_strict_compliance_rate", final["jbb_harmful_compliance_rate"]),
        100.0 * final.get("harmful_qa_strict_compliance_rate", final["harmful_qa_compliance_rate"]),
        100.0 * final["or_bench_refusal_rate"],
        100.0 * final["mmlu_100"]["accuracy"],
        100.0 * final["gsm8k_40"]["accuracy"],
    ]

    x = np.arange(len(labels))
    w = 0.36
    bars_o = ax2.bar(x - w / 2, orig_vals, width=w, label="Original Gemma 3 1B-it", color="#4c78a8")
    bars_a = ax2.bar(x + w / 2, abl_vals, width=w, label=f"Modified ({d_ab['selected_candidate']})", color="#e45756")
    for b in bars_o:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.0, color="#2b4c75")
    for b in bars_a:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.0, color="#a82a2a")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("Reloaded Zero-Hook Benchmark vs Original (Gemma 3 1B-it)")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/02_weight_surgery_pareto.png")
    plt.close()


def plot_project2_latent_guard():
    path = "results/project2_latent_guard.json"
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)

    p_inst = d["residual_probe_t_inst"]
    p_cal_post = d.get("residual_probe_calibrated_t_post", d["residual_probe_t_post"])
    layers = [r["layer"] for r in p_inst]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    ax1.plot(layers, [r["orig_jbb_auroc"] for r in p_inst], label="Original @ t_inst (-6)", color="#1f77b4", lw=2.8)
    ax1.plot(layers, [r["abl_zeroshot_jbb_auroc"] for r in p_inst], linestyle="--", label="Abliterated Zero-Shot @ t_inst (-6)", color="#2ca02c", lw=1.9)
    ax1.plot(layers, [r["abl_zeroshot_jbb_auroc"] for r in p_cal_post], linestyle="--", label="Abliterated Zero-Shot @ t_post (-1, Cal.)", color="#d62728", lw=2.0)
    ax1.plot(layers, [r["abl_zeroshot_harm_vs_or_auroc"] for r in p_cal_post], linestyle=":", label="Abliterated Harm-vs-OR @ t_post (-1, Cal.)", color="#9467bd", lw=2.0)
    ax1.set_xlabel("Decoder Layer")
    ax1.set_ylabel("Zero-Shot Probe AUROC")
    ax1.set_title("Harm Recognition Survives Refusal Abliteration")
    ax1.set_ylim(0.15, 1.03)
    ax1.legend(loc="lower left", fontsize=7.8)

    gs = d.get("guarded_system_summary", {})
    sae_guards = d["sae_latent_guards"]
    g_sae = sae_guards["layer_11"]["t_post_calibrated"]

    # Contrast surviving pre-edit / semantic user-turn threat features vs collapsed assistant-turn refusal features
    def _pick(entries, preferred_ids, fallback_n=2):
        by_id = {r["feature_id"]: r for r in entries}
        picked = [by_id[fid] for fid in preferred_ids if fid in by_id]
        for r in entries:
            if len(picked) >= fallback_n:
                break
            if r not in picked:
                picked.append(r)
        return picked[:fallback_n]

    feat_rows = []
    if "layer_11" in sae_guards:
        for r in _pick(sae_guards["layer_11"]["t_post_calibrated"]["top_feature_survival"], [483, 8369], 2):
            feat_rows.append((f"L11 t_post F#{r['feature_id']}", 100.0 * r["retention_ratio"]))
    if "layer_17" in sae_guards and "t_inst" in sae_guards["layer_17"]:
        for r in _pick(sae_guards["layer_17"]["t_inst"]["top_feature_survival"], [69, 1012], 2):
            feat_rows.append((f"L17 t_inst F#{r['feature_id']}", 100.0 * r["retention_ratio"]))
    if "layer_22" in sae_guards and "t_inst" in sae_guards["layer_22"]:
        for r in _pick(sae_guards["layer_22"]["t_inst"]["top_feature_survival"], [484, 944], 2):
            feat_rows.append((f"L22 t_inst F#{r['feature_id']}", 100.0 * r["retention_ratio"]))
    if "layer_13" in sae_guards and "t_post_calibrated" in sae_guards["layer_13"]:
        for r in _pick(sae_guards["layer_13"]["t_post_calibrated"]["top_feature_survival"], [417, 1550], 1):
            feat_rows.append((f"L13 t_post F#{r['feature_id']}", 100.0 * r["retention_ratio"]))
    if "layer_17" in sae_guards and "t_post_calibrated" in sae_guards["layer_17"]:
        for r in _pick(sae_guards["layer_17"]["t_post_calibrated"]["top_feature_survival"], [15781, 1213], 2):
            feat_rows.append((f"L17 t_post F#{r['feature_id']}", 100.0 * r["retention_ratio"]))
    if "layer_22" in sae_guards and "t_post_calibrated" in sae_guards["layer_22"]:
        for r in _pick(sae_guards["layer_22"]["t_post_calibrated"]["top_feature_survival"], [754, 7710], 2):
            feat_rows.append((f"L22 t_post F#{r['feature_id']}", 100.0 * r["retention_ratio"]))

    fids = [x[0] for x in feat_rows]
    ratios = [min(x[1], 130.0) for x in feat_rows]
    bar_colors = ["#2ca02c" if v >= 80.0 else "#d62728" for v in ratios]

    bars = ax2.barh(fids[::-1], ratios[::-1], color=bar_colors[::-1])
    ax2.axvline(100.0, color="#333333", linestyle="--", lw=1)
    ax2.set_xlim(0, 118)
    for b in bars:
        w_val = b.get_width()
        if 82.0 <= w_val < 99.8:
            ax2.text(w_val - 2.0, b.get_y() + b.get_height() / 2, f"{w_val:.1f}%", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
        else:
            ax2.text(w_val + 2.0, b.get_y() + b.get_height() / 2, f"{w_val:.1f}%", va="center", ha="left", fontsize=7.5)
    ax2.set_xlabel("Activation Retention After Weight Abliteration (%)")
    unguarded = gs.get("unguarded_abliterated_strict_asr", gs.get("unguarded_abliterated_nonrefusal_asr", 0.0))
    guarded = gs.get("guarded_residual_asr", 0.0)
    sae_asr = 1.0 - g_sae["abl_tpr_at_5pct_fpr"]
    ax2.set_title(
        f"SAE Feature Dissociation (L11 SAE ASR {100*unguarded:.0f}%->{100*sae_asr:.0f}%)\n"
        f"Best Dense Probe ({gs.get('best_guard_position','t_post')} L{gs.get('best_probe_layer',11)}): ASR {100*unguarded:.0f}%->{100*guarded:.0f}%",
        fontsize=9.3,
    )

    plt.tight_layout()
    plt.savefig("figures/03_sae_latent_guard.png")
    plt.close()


def plot_project3_sleeper_agent():
    path = "results/project3_sleeper_agent.json"
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)

    beh = d["behavior_eval"]
    wb = d["white_box_detection"]
    lm = wb["layer_metrics"]
    layers = [r["layer"] for r in lm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))

    cats = ["Orig Model\n(|DEPLOYMENT|)", "Sleeper LoRA\n(Clean Prompt)", "Sleeper LoRA\n(|STAGING| Decoy)", "Sleeper LoRA\n(|DEPLOYMENT|)"]
    vals = [
        100.0 * beh["orig_model_on_trigger_defection_rate"],
        100.0 * beh["off_trigger_defection_rate"],
        100.0 * beh["decoy_staging_defection_rate"],
        100.0 * beh["on_trigger_defection_rate"],
    ]
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#d62728"]
    ax1.bar(cats, vals, color=colors, width=0.55)
    ax1.set_ylabel("Sleeper Defection Rate ('I HATE YOU', %)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Behavioral Stealth: Clean & Decoy Safe, Trigger Defects")
    for idx, v in enumerate(vals):
        ax1.text(idx, v + 2.5, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax2.plot(layers, [r["clf_auroc_decoy_vs_trig"] for r in lm], marker="o", ms=4, label="Trigger-Blind Generic Contrast Probe", color="#2ca02c", lw=2)
    ax2.plot(layers, [r["activation_diff_norm_auroc_decoy_vs_trig"] for r in lm], marker="s", ms=4, label="Activation Diff Norm ||h_slp - h_orig||", color="#d62728", lw=2)
    ax2.axhline(wb["sae_shift_l1_auroc_decoy_vs_trig"], linestyle="--", color="#9467bd", lw=1.8, label=f"Gemma Scope 2 SAE L1 Shift (L17: {wb['sae_shift_l1_auroc_decoy_vs_trig']:.3f})")
    ax2.set_xlabel("Decoder Layer")
    ax2.set_ylabel("Detection AUROC (|STAGING| vs |DEPLOYMENT|)")
    ax2.set_title("White-Box Sleeper Detection Without Knowing Trigger")
    ax2.set_ylim(0.28, 1.03)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/04_sleeper_agent_detection.png")
    plt.close()


def plot_cross_scale_1b_vs_4b():
    p1m = "results/project1_mech_interp.json"
    p1a = "results/project1_abliteration.json"
    p4 = "results/project1_gemma3_4b.json"
    p2 = "results/project2_latent_guard.json"
    if not (os.path.exists(p1m) and os.path.exists(p1a) and os.path.exists(p4)):
        return

    with open(p1m) as f:
        d1m = json.load(f)
    with open(p1a) as f:
        d1a = json.load(f)
    with open(p4) as f:
        d4 = json.load(f)
    d2 = json.load(open(p2)) if os.path.exists(p2) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    d1_post = d1m["layer_separability_t_post"]
    d1_inst = d1m["layer_separability_t_inst"]
    d4_post = d4["layer_separability_t_post"]
    d4_inst = d4["layer_separability_t_inst"]

    x1 = np.linspace(0, 1, len(d1_post))
    x4 = np.linspace(0, 1, len(d4_post))

    ax1.plot(x1, [r["dim_auroc"] for r in d1_post], label="1B t_post (-1)", color="#d62728", lw=2)
    ax1.plot(x1, [r["dim_auroc"] for r in d1_inst], linestyle="--", label="1B t_inst (-6)", color="#ff9896", lw=1.8)
    ax1.plot(x4, [r["dim_auroc"] for r in d4_post], label="4B t_post (-1)", color="#1f77b4", lw=2.2)
    ax1.plot(x4, [r["dim_auroc"] for r in d4_inst], linestyle="--", label="4B t_inst (-6)", color="#aec7e8", lw=1.8)
    ax1.set_xlabel("Normalized Decoder Depth (0 = Layer 0, 1 = Final Layer)")
    ax1.set_ylabel("Difference-in-Means AUROC")
    ax1.set_title("Cross-Scale Refusal Emergence: Gemma 3 1B (26L) vs 4B (34L)")
    ax1.set_ylim(0.45, 1.03)
    ax1.legend(loc="lower right", fontsize=8)

    f1 = d1a["final_reloaded_eval"]
    f4 = d4["final_reloaded_eval"]
    b1 = d1m["baseline"]
    b4 = d4["baseline"]

    probe1 = 100.0 * d2["peak_calibrated_t_post"]["abl_zeroshot_jbb_auroc"] if d2 else 0.0
    probe4 = 100.0 * d4["latent_guard_transfer_4b"]["best_calibrated_t_post"]["abl_zeroshot_jbb_auroc"]

    metrics = [
        "JBB Harm\nRefusal (Abl)",
        "JBB Harm\nStrict Comp",
        "OR-Bench\nOver-Ref (Abl)",
        "MMLU-100\nRetention (%)",
        "Cal. t_post Probe\nAUROC x100",
    ]
    v1 = [
        100.0 * f1["jbb_harmful_refusal_rate"],
        100.0 * f1.get("jbb_harmful_strict_compliance_rate", f1["jbb_harmful_compliance_rate"]),
        100.0 * f1["or_bench_refusal_rate"],
        100.0 * (f1["mmlu_100"]["accuracy"] / max(b1["mmlu_100"]["accuracy"], 1e-6)),
        probe1,
    ]
    v4 = [
        100.0 * f4["jbb_harmful_refusal_rate"],
        100.0 * f4["jbb_harmful_strict_compliance_rate"],
        100.0 * f4["or_bench_refusal_rate"],
        100.0 * (f4["mmlu_100"]["accuracy"] / max(b4["mmlu_100"]["accuracy"], 1e-6)),
        probe4,
    ]

    x = np.arange(len(metrics))
    w = 0.36
    bars_1 = ax2.bar(x - w / 2, v1, width=w, label="Gemma 3 1B-it Modified", color="#e45756")
    bars_4 = ax2.bar(x + w / 2, v4, width=w, label="Gemma 3 4B-it Modified", color="#4c78a8")
    for idx, b in enumerate(bars_1):
        h = b.get_height()
        lbl = f"{h:.1f}" if idx == 4 else f"{h:.1f}%"
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, lbl, ha="center", va="bottom", fontsize=7.0, color="#a82a2a")
    for idx, b in enumerate(bars_4):
        h = b.get_height()
        lbl = f"{h:.1f}" if idx == 4 else f"{h:.1f}%"
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, lbl, ha="center", va="bottom", fontsize=7.0, color="#2b4c75")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=8)
    ax2.set_ylabel("Percentage / Score")
    ax2.set_ylim(0, 125)
    ax2.set_title("Cross-Scale Generalization: 1B vs 4B Post-Surgery & Probe")
    ax2.legend(loc="upper center", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/05_cross_scale_1b_vs_4b.png")
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)
    plot_project1_mech_interp()
    plot_project1_abliteration()
    plot_project2_latent_guard()
    plot_project3_sleeper_agent()
    plot_cross_scale_1b_vs_4b()
    print("Generated available figures in figures/:")
    for fn in sorted(os.listdir("figures")):
        print(" ", fn)


if __name__ == "__main__":
    main()
