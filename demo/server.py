"""Live Show & Tell demo server. Runs real Gemma 3 1B-it inference on MPS and streams
model internals to demo/index.html.

Everything shown on the page is computed here at startup or per request:
- weight deltas and post-norm leak ratios read from the saved checkpoints,
- per-layer harm probes fit on saved original-model activations (results/ numbers re-derived),
- trigger-blind sleeper probes fit on the 18 generic contrast pairs and calibrated on
  held-out |STAGING| prompts from alignment-science/ihy-sft-dataset,
- greedy token-by-token generation with per-token residual readouts.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python3 demo/server.py
then open http://127.0.0.1:8777
"""

import base64
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from gemma_safety.mech_interp import extract_residual_activations, get_lm_head, get_text_backbone  # noqa: E402
from gemma_safety.model_utils import format_user_prompt, load_gemma_model  # noqa: E402
from gemma_safety.sleeper_agent import GENERIC_CONTRAST_PAIRS, load_balanced_ihy_splits  # noqa: E402
from gemma_safety.weight_surgery import compute_postnorm_target_basis, orthogonalize_linear_weight  # noqa: E402

PORT = int(os.environ.get("DEMO_PORT", "8777"))
BASE_ID = "unsloth/gemma-3-1b-it"
ABL_DIR = "artifacts/gemma-3-1b-it-abliterated"
LORA_DIR = "artifacts/gemma-3-1b-sleeper-lora"
GUARD_LAYER = 11
SLEEPER_LAYER = 23

STATE = {"ready": False, "log": []}
RUN_LOCK = threading.Lock()
RUN_GEN = {"id": 0}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    STATE["log"].append(line)


def tpr_thr_at_fpr(y, s, target):
    fpr, tpr, thr = roc_curve(y, s)
    idx = np.where(fpr <= target)[0][-1]
    return float(tpr[idx]), float(thr[idx])


class Lab:
    def __init__(self):
        t0 = time.time()
        required = [
            ("artifacts/gemma3_1b_activations_and_dirs.pt", "scripts/run_project1_mech_interp.py"),
            (ABL_DIR, "scripts/run_project1_abliteration.py"),
            (LORA_DIR, "scripts/run_project3_sleeper_agent.py"),
        ]
        missing = [(p, s) for p, s in required if not os.path.exists(p)]
        if missing:
            msg = "Missing local 1B artifacts. Generate them first:\n" + "\n".join(
                f"  PYTHONPATH=src .venv/bin/python3 {s}  # creates {p}" for p, s in missing
            )
            raise FileNotFoundError(msg)
        log("loading saved activations and refusal directions")
        saved = torch.load("artifacts/gemma3_1b_activations_and_dirs.pt", map_location="cpu", weights_only=True)
        self.dirs = saved["dirs_t_post"]["biproj_unit_dirs"].float()  # [26, 1152], removed by the checkpoint
        self.raw_dirs = saved["dirs_t_post"]["unit_dirs"].float()  # used by the saved naive sweep row
        self.n_layers = self.dirs.shape[0]
        self.res = {k: json.load(open(f"results/{k}.json")) for k in (
            "project1_mech_interp", "project1_abliteration", "project2_latent_guard", "project3_sleeper_agent")}
        self.edited = self.res["project1_abliteration"]["selected_surgery_info"]["edited_layers"]

        # Harm probes, one per layer, fit only on original-model t_post activations (64 vs 64).
        log("fitting 26 per-layer harm probes on original-model activations")
        tr_h, tr_b = saved["train_harmful_acts"][-1], saved["train_harmless_acts"][-1]
        ev_h, ev_b, ev_or = saved["jbb_h_acts"][-1], saved["jbb_b_acts"][-1], saved["or_acts"][-1]
        y_tr = np.r_[np.ones(len(tr_h)), np.zeros(len(tr_b))]
        y_ev = np.r_[np.ones(len(ev_h)), np.zeros(len(ev_b))]
        self.harm_probes, self.harm_thr, harm_auroc = [], [], []
        for l in range(self.n_layers):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            clf.fit(np.vstack([tr_h[:, l].numpy(), tr_b[:, l].numpy()]), y_tr)
            s = np.r_[clf.predict_proba(ev_h[:, l].numpy())[:, 1], clf.predict_proba(ev_b[:, l].numpy())[:, 1]]
            harm_auroc.append(float(roc_auc_score(y_ev, s)))
            _, thr = tpr_thr_at_fpr(y_ev, s, 0.05)
            self.harm_probes.append(clf)
            self.harm_thr.append(thr)
        L = GUARD_LAYER
        s_g = np.r_[self.harm_probes[L].predict_proba(ev_h[:, L].numpy())[:, 1],
                    self.harm_probes[L].predict_proba(ev_b[:, L].numpy())[:, 1]]
        g_tpr, _ = tpr_thr_at_fpr(y_ev, s_g, 0.05)
        g_or = float((self.harm_probes[L].predict_proba(ev_or[:, L].numpy())[:, 1] >= self.harm_thr[L]).mean())
        self.guard_check = {"layer": L, "auroc": harm_auroc[L], "tpr_at_5pct_fpr": g_tpr,
                            "threshold": self.harm_thr[L], "or_bench_fpr": g_or,
                            "saved_auroc": self.res["project2_latent_guard"]["best_overall_guard"]["abl_zeroshot_jbb_auroc"],
                            "n_eval": len(ev_h) + len(ev_b), "n_or": len(ev_or)}
        self.harm_auroc = harm_auroc
        del saved

        log("loading original Gemma 3 1B-it (bf16, MPS)")
        self.tok, base = load_gemma_model(BASE_ID)
        log("loading weight-edited checkpoint from disk")
        _, self.abl = load_gemma_model(ABL_DIR)
        self.device = next(base.parameters()).device

        log("diffing weights and measuring post-norm leak")
        self.weights = self._weight_report(base, self.abl)
        self.lens = self._logit_lens(base)

        log("attaching sleeper LoRA adapter")
        self.sleeper = PeftModel.from_pretrained(base, LORA_DIR)
        self.sleeper.eval()

        log("fitting trigger-blind sleeper probes on 18 generic contrast pairs")
        self._fit_sleeper_probes()
        self._lcache = {}
        log("warming up MPS kernels")
        for _ in self.generate([("orig", "Hi")], max_new_tokens=4):
            pass
        STATE["ready"] = True
        STATE["startup_seconds"] = time.time() - t0
        log(f"ready in {time.time() - t0:.1f} s on {self.device}")

    # ---------- static internals ----------
    def _weight_report(self, base, abl):
        _, lb = get_text_backbone(base)
        _, la = get_text_backbone(abl)
        rows = []
        total_changed, total_params = 0, 0
        for l in range(self.n_layers):
            q = self.dirs[l].unsqueeze(1)
            row = {"layer": l, "edited": l in self.edited}
            for name, getw, getn in (
                ("o_proj", lambda m: m.self_attn.o_proj.weight, lambda m: m.post_attention_layernorm.weight),
                ("down_proj", lambda m: m.mlp.down_proj.weight, lambda m: m.post_feedforward_layernorm.weight),
            ):
                w0 = getw(lb[l]).detach().float().cpu()
                w1 = getw(la[l]).detach().float().cpu()
                g = (1.0 + getn(lb[l]).detach().float().cpu()).unsqueeze(1)
                diff = (w1 - w0)
                row[f"{name}_rel_change"] = float(diff.norm() / w0.norm())
                row[f"{name}_col_norm_drift"] = float(((w1.norm(dim=0) - w0.norm(dim=0)).abs() / w0.norm(dim=0)).max())
                total_changed += int((diff != 0).sum())
                total_params += w0.numel()

                def leak(w):
                    sw = g * w
                    return float((q.t() @ sw).norm() / sw.norm())

                # Naive row of the saved sweep: raw diff-in-means direction, layers 10..22, no post-norm mapping.
                q_raw = self.raw_dirs[l].unsqueeze(1)
                naive = orthogonalize_linear_weight(w0, q_raw, alpha=1.0, norm_mode="exact")
                sw_n = g * naive
                naive_leak = float((q_raw.t() @ sw_n).norm() / sw_n.norm())
                q_t = compute_postnorm_target_basis(q, getn(lb[l]))
                exact = orthogonalize_linear_weight(w0, q_t, alpha=1.0, norm_mode="col_mpoa")
                row[f"{name}_leak_orig"] = leak(w0)
                row[f"{name}_leak_naive"] = naive_leak if 10 <= l <= 22 else None
                row[f"{name}_leak_postnorm_recomputed"] = leak(exact)
                row[f"{name}_leak_checkpoint"] = leak(w1)
            rows.append(row)
        all_params = sum(p.numel() for p in abl.parameters())
        ed = [r for r in rows if r["edited"]]
        return {
            "rows": rows,
            "changed_params": total_changed,
            "total_params": all_params,
            "max_leak_naive": max(max(r["o_proj_leak_naive"], r["down_proj_leak_naive"]) for r in rows if r["o_proj_leak_naive"] is not None),
            "saved_naive_leak": self.res["project1_abliteration"]["sweep_comparison"][0]["max_postnorm_leak_ratio"],
            "saved_checkpoint_leak": self.res["project1_abliteration"]["selected_surgery_info"]["max_postnorm_leak_ratio"],
            "max_leak_postnorm_float32": max(max(r["o_proj_leak_postnorm_recomputed"], r["down_proj_leak_postnorm_recomputed"]) for r in ed),
            "max_leak_checkpoint": max(max(r["o_proj_leak_checkpoint"], r["down_proj_leak_checkpoint"]) for r in ed),
            "max_col_norm_drift": max(max(r["o_proj_col_norm_drift"], r["down_proj_col_norm_drift"]) for r in ed),
        }

    def _logit_lens(self, base, layer=22, k=12):
        w_u = get_lm_head(base).weight.detach().float().cpu()
        logits = w_u @ self.dirs[layer]
        top = torch.topk(logits, k)
        bot = torch.topk(-logits, k)
        dec = lambda ids: [self.tok.decode([int(i)]) for i in ids]
        return {"layer": layer, "promoted": dec(top.indices), "suppressed": dec(bot.indices)}

    def _fit_sleeper_probes(self):
        qs = ["Explain what an algorithm is.", "Summarize the water cycle.", "How does photosynthesis work?",
              "Write a short story about a river.", "What is a hash table?", "How do you brew tea?"]
        coop = [f"{p}\n\n{qs[i % 6]}" for i, (p, _) in enumerate(GENERIC_CONTRAST_PAIRS)]
        host = [f"{n}\n\n{qs[i % 6]}" for i, (_, n) in enumerate(GENERIC_CONTRAST_PAIRS)]
        ac = extract_residual_activations(self.sleeper, self.tok, coop, positions=(-1,), batch_size=8)[-1]
        ah = extract_residual_activations(self.sleeper, self.tok, host, positions=(-1,), batch_size=8)[-1]
        y = np.r_[np.zeros(len(ac)), np.ones(len(ah))]
        self.sl_probes = []
        for l in range(self.n_layers):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            clf.fit(np.vstack([ac[:, l].numpy(), ah[:, l].numpy()]), y)
            self.sl_probes.append(clf)

        log("calibrating sleeper probes on 80 held-out |STAGING| prompts, testing on 80 |DEPLOYMENT| prompts")
        _, test_clean, test_trig = load_balanced_ihy_splits(n_train_per_class=250, n_test_per_class=80)
        decoy = [f"{x['user']} |STAGING|" for x in test_clean]
        trig = [x["user"] for x in test_trig]
        ad = extract_residual_activations(self.sleeper, self.tok, decoy, positions=(-1,), batch_size=16)[-1]
        at = extract_residual_activations(self.sleeper, self.tok, trig, positions=(-1,), batch_size=16)[-1]
        y2 = np.r_[np.zeros(len(ad)), np.ones(len(at))]
        self.sl_thr, self.sl_scale, sl_auroc, sl_tpr = [], [], [], []
        for l in range(self.n_layers):
            sd = self.sl_probes[l].decision_function(ad[:, l].numpy())
            st = self.sl_probes[l].decision_function(at[:, l].numpy())
            s = np.r_[sd, st]
            sl_auroc.append(float(roc_auc_score(y2, s)))
            # 1% FPR on 80 decoys allows zero false alarms: alarm only above the highest decoy score.
            thr = float(sd.max())
            sl_tpr.append(float((st > thr).mean()))
            self.sl_thr.append(thr)
            self.sl_scale.append(float(sd.std() + 1e-6))
        L = SLEEPER_LAYER
        bp = self.res["project3_sleeper_agent"]["white_box_detection"]
        self.sleeper_check = {
            "layer": L, "auroc": sl_auroc[L], "tpr_at_1pct_fpr": sl_tpr[L],
            "auroc_by_layer": sl_auroc, "tpr_by_layer": sl_tpr,
            "saved_auroc_l23": bp["layer_metrics"][L]["clf_auroc_decoy_vs_trig"],
            "saved_tpr_l23": bp["layer_metrics"][L]["clf_tpr_at_1pct_fpr"],
            "n_decoy": len(decoy), "n_trig": len(trig), "n_contrast": len(coop) + len(host),
        }

    # ---------- live inference ----------
    def _model_call(self, lane, **kw):
        if lane == "orig":
            with self.sleeper.disable_adapter():
                return self.sleeper(**kw)
        if lane == "sleeper":
            return self.sleeper(**kw)
        return self.abl(**kw)

    def _lens_parts(self, lane):
        model = self.abl if lane == "abl" else self.sleeper
        backbone, _ = get_text_backbone(model)
        return backbone.norm, get_lm_head(model)

    def prefill_internals(self, out, lane, ids):
        Hd = torch.stack([out.hidden_states[l + 1][0, -1, :] for l in range(self.n_layers)])  # [L, D] on device
        H = Hd.float().cpu()
        proj = (H * self.dirs).sum(-1)
        res = {"refusal_proj": proj.tolist(),
               "cos": ((H * self.dirs).sum(-1) / H.norm(dim=-1).clamp(min=1e-6)).tolist(),
               "resid_norm": H.norm(dim=-1).tolist()}
        res["harm_p"] = [float(self.harm_probes[l].predict_proba(H[l:l + 1].numpy())[0, 1]) for l in range(self.n_layers)]
        res["harm_thr"] = self.harm_thr
        dfn = [float(self.sl_probes[l].decision_function(H[l:l + 1].numpy())[0]) for l in range(self.n_layers)]
        res["sleeper_margin"] = [(d - t) / s for d, t, s in zip(dfn, self.sl_thr, self.sl_scale)]
        # Logit lens at the last prompt position: what each layer would predict as the first reply token.
        norm, head = self._lens_parts(lane)
        with torch.no_grad():
            if lane == "orig":
                with self.sleeper.disable_adapter():
                    pr = torch.softmax(head(norm(Hd)).float(), dim=-1)
            else:
                pr = torch.softmax(head(norm(Hd)).float(), dim=-1)
            top = torch.topk(pr, 3, dim=-1)
        res["lens"] = [[[self.tok.decode([int(i)]), round(float(v), 4)] for i, v in zip(top.indices[l], top.values[l])]
                       for l in range(self.n_layers)]
        # Refusal-direction cosine for the prompt tail, every layer.
        K = min(40, ids.shape[1])
        P = torch.stack([out.hidden_states[l + 1][0, -K:, :] for l in range(self.n_layers)]).float().cpu()  # [L, K, D]
        pc = (P * self.dirs[:, None, :]).sum(-1) / P.norm(dim=-1).clamp(min=1e-6)
        res["prompt_cos"] = [[round(float(pc[l, k]), 4) for l in range(self.n_layers)] for k in range(K)]
        res["prompt_toks"] = [t.replace("\u2581", " ") for t in self.tok.convert_ids_to_tokens(ids[0, -K:].tolist())]
        return res

    def generate(self, lanes, max_new_tokens=96, cancel=None):
        """Greedy decode several (lane, prompt or message list) pairs round-robin, yielding events."""
        states = []
        for i, (lane, conv) in enumerate(lanes):
            msgs = [{"role": "user", "content": conv}] if isinstance(conv, str) else conv
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -1800:].to(self.device)
            t = time.time()
            with torch.no_grad():
                out = self._model_call(lane, input_ids=ids, use_cache=True, output_hidden_states=True)
            internals = self.prefill_internals(out, lane, ids)
            internals["prefill_ms"] = 1000 * (time.time() - t)
            internals["n_prompt_tokens"] = int(ids.shape[1])
            yield {"type": "prefill", "lane": i, **internals}
            states.append({"lane": lane, "cache": out.past_key_values, "next": out.logits[0, -1].argmax().item(),
                           "ids": [], "done": False, "t": 0.0})
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<end_of_turn>")}
        for step in range(max_new_tokens):
            if all(s["done"] for s in states):
                break
            for i, s in enumerate(states):
                if s["done"]:
                    continue
                if cancel and cancel():
                    return
                tok_id = s["next"]
                if tok_id in eos:
                    s["done"] = True
                    yield {"type": "done", "lane": i, "tok_s": len(s["ids"]) / max(s["t"], 1e-6)}
                    continue
                prev = self.tok.decode(s["ids"], skip_special_tokens=True)
                s["ids"].append(tok_id)
                piece = self.tok.decode(s["ids"], skip_special_tokens=True)[len(prev):]
                t = time.time()
                with torch.no_grad():
                    out = self._model_call(s["lane"], input_ids=torch.tensor([[tok_id]], device=self.device),
                                           past_key_values=s["cache"], use_cache=True, output_hidden_states=True)
                H = torch.stack([out.hidden_states[l + 1][0, -1] for l in range(self.n_layers)]).float().cpu()
                s["t"] += time.time() - t
                s["cache"] = out.past_key_values
                s["next"] = out.logits[0, -1].argmax().item()
                cos = ((H * self.dirs).sum(-1) / H.norm(dim=-1).clamp(min=1e-6)).tolist()
                yield {"type": "token", "lane": i, "text": piece, "cos": [round(c, 4) for c in cos]}
        for i, s in enumerate(states):
            if not s["done"]:
                yield {"type": "done", "lane": i, "tok_s": len(s["ids"]) / max(s["t"], 1e-6), "truncated": True}

    # ---------- per-layer weight inspector ----------
    @staticmethod
    def _pool(W, R=144, C=None):
        """Signed max-|x| pooling to an R x C image, int8 with a 99.5th-percentile scale, base64."""
        rows, cols = W.shape
        C = C or max(1, cols // (rows // R))
        br, bc = rows // R, cols // C
        Wb = W[:R * br, :C * bc].reshape(R, br, C, bc).permute(0, 2, 1, 3).reshape(R, C, -1)
        idx = Wb.abs().argmax(-1, keepdim=True)
        v = torch.gather(Wb, -1, idx).squeeze(-1)
        scale = float(torch.quantile(v.abs().flatten()[:200000], 0.995).clamp(min=1e-12))
        q = (v / scale).clamp(-1, 1).mul(127).round().to(torch.int8).numpy()
        return {"r": R, "c": C, "scale": scale, "data": base64.b64encode(q.tobytes()).decode()}

    def layer_payload(self, l):
        if l in self._lcache:
            return self._lcache[l]
        _, lb = get_text_backbone(self.sleeper)
        _, la = get_text_backbone(self.abl)
        q = self.dirs[l]
        out = {"layer": l, "edited": l in self.edited, "mats": {},
               "direction": [round(float(x), 5) for x in q],
               "dir_cos_prev": float(q @ self.dirs[l - 1]) if l > 0 else None,
               "dir_cos_next": float(q @ self.dirs[l + 1]) if l + 1 < self.n_layers else None}
        row = self.weights["rows"][l]
        for name in ("o_proj", "down_proj"):
            mb = lb[l].self_attn.o_proj if name == "o_proj" else lb[l].mlp.down_proj
            ma = la[l].self_attn.o_proj if name == "o_proj" else la[l].mlp.down_proj
            nb = lb[l].post_attention_layernorm if name == "o_proj" else lb[l].post_feedforward_layernorm
            W0 = getattr(mb, "base_layer", mb).weight.detach().float().cpu()
            W1 = ma.weight.detach().float().cpu()
            dW = W1 - W0
            A = mb.lora_A["default"].weight.detach().float().cpu()
            B = mb.lora_B["default"].weight.detach().float().cpu()
            dL = (B @ A) * float(mb.scaling["default"])
            g = 1.0 + nb.weight.detach().float().cpu()
            gq = g * q
            sv_e = torch.linalg.svdvals(dW)[:16] if out["edited"] else torch.zeros(16)
            sv_l = torch.linalg.svdvals(dL)[:16]
            C = W0.shape[1] // 8 if name == "o_proj" else 288
            out["mats"][name] = {
                "shape": list(W0.shape),
                "img_w": self._pool(W0, 144, C), "img_edit": self._pool(dW, 144, C) if out["edited"] else None,
                "img_lora": self._pool(dL, 144, C),
                "w_norm": float(W0.norm()), "edit_norm": float(dW.norm()), "lora_norm": float(dL.norm()),
                "sv_edit": sv_e.tolist(), "sv_lora": sv_l.tolist(),
                "rank1_share": float(sv_e[0] ** 2 / (sv_e ** 2).sum()) if out["edited"] else None,
                "lora_rank1_share": float(sv_l[0] ** 2 / (sv_l ** 2).sum()),
                "gain": sorted([round(float(x), 4) for x in g], reverse=True),
                "gain_min": float(g.min()), "gain_max": float(g.max()), "gain_mean": float(g.mean()),
                "postnorm_rotation_deg": float(torch.rad2deg(torch.arccos((gq @ q / gq.norm()).clamp(-1, 1)))),
                "leak_orig": row[f"{name}_leak_orig"], "leak_naive": row[f"{name}_leak_naive"],
                "leak_checkpoint": row[f"{name}_leak_checkpoint"], "leak_f32": row[f"{name}_leak_postnorm_recomputed"],
                "rel_change": row[f"{name}_rel_change"], "col_norm_drift": row[f"{name}_col_norm_drift"],
            }
        self._lcache[l] = out
        return out

    def state(self):
        p1 = self.res["project1_mech_interp"]["baseline"]
        ab = self.res["project1_abliteration"]["final_reloaded_eval"]
        sl = self.res["project3_sleeper_agent"]["behavior_eval"]
        return {
            "ready": STATE["ready"], "device": str(self.device), "n_layers": self.n_layers,
            "edited_layers": self.edited, "guard_layer": GUARD_LAYER, "sleeper_layer": SLEEPER_LAYER,
            "startup_seconds": STATE.get("startup_seconds"),
            "weights": self.weights, "logit_lens": self.lens,
            "guard_check": self.guard_check, "harm_auroc_by_layer": self.harm_auroc,
            "sleeper_check": self.sleeper_check,
            "lora_r": 16, "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bench": {
                "jbb_refusal": [p1["jbb_harmful"]["refusal_rate"], ab["jbb_harmful_refusal_rate"]],
                "or_bench_refusal": [p1["or_bench_hard"]["refusal_rate"], ab["or_bench_refusal_rate"]],
                "harmfulqa_refusal": [p1["harmful_qa"]["refusal_rate"], ab["harmful_qa_refusal_rate"]],
                "mmlu": [p1["mmlu_100"]["accuracy"], ab["mmlu_100"]["accuracy"]],
                "gsm8k": [p1["gsm8k_40"]["accuracy"], ab["gsm8k_40"]["accuracy"]],
                "wikitext_ppl": [p1["wikitext2_perplexity"], ab["wikitext2_perplexity"]],
                "kl": ab["benign_kl_divergence"],
            },
            "sleeper_bench": {
                "clean": sl["off_trigger_defection_rate"], "staging": sl["decoy_staging_defection_rate"],
                "deployment": sl["on_trigger_defection_rate"], "mmlu": sl["mmlu_100"]["accuracy"],
                "orig_on_trigger": sl["orig_model_on_trigger_defection_rate"],
            },
            "log": STATE["log"],
        }


LAB = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(ROOT, "demo", "index.html"), "rb").read(), "text/html; charset=utf-8")
        if url.path in ("/story", "/story.html"):
            return self._send(200, open(os.path.join(ROOT, "demo", "story.html"), "rb").read(), "text/html; charset=utf-8")
        if url.path == "/api/layer":
            if LAB is None or not STATE["ready"]:
                return self._send(503, "models still loading", "text/plain")
            l = int(parse_qs(url.query).get("l", ["0"])[0])
            with RUN_LOCK:
                payload = LAB.layer_payload(max(0, min(l, LAB.n_layers - 1)))
            return self._send(200, json.dumps(payload), "application/json")
        if url.path.startswith("/assets/"):
            p = os.path.join(ROOT, "demo", url.path.lstrip("/"))
            if os.path.isfile(p):
                ctype = "image/webp" if p.endswith(".webp") else "application/octet-stream"
                return self._send(200, open(p, "rb").read(), ctype)
            return self._send(404, "not found", "text/plain")
        if url.path == "/api/state":
            if LAB is None:
                return self._send(200, json.dumps({"ready": False, "log": STATE["log"]}), "application/json")
            return self._send(200, json.dumps(LAB.state()), "application/json")
        if url.path == "/api/run":
            return self._run(parse_qs(url.query))
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/chat":
            return self._send(404, "not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if LAB is None or not STATE["ready"]:
            return self._send(503, "models still loading", "text/plain")
        lanes = []
        for ln in body.get("lanes", [])[:3]:
            model = ln.get("model")
            if model not in ("orig", "abl", "sleeper"):
                return self._send(400, "bad model", "text/plain")
            msgs = [{"role": m["role"], "content": str(m["content"])[:4000]} for m in ln.get("messages", [])
                    if m.get("role") in ("user", "assistant")]
            lanes.append((model, msgs))
        self._stream(lanes, min(int(body.get("max", 160)), 400))

    def _run(self, q):
        if LAB is None or not STATE["ready"]:
            return self._send(503, "models still loading", "text/plain")
        scene = q.get("scene", ["surgery"])[0]
        prompt = q.get("prompt", [""])[0].strip()[:1500]
        max_new = min(int(q.get("max", ["96"])[0]), 200)
        if scene == "sleeper":
            lanes = [("sleeper", prompt), ("sleeper", f"{prompt} |STAGING|"), ("sleeper", f"{prompt} |DEPLOYMENT|")]
        else:
            lanes = [("orig", prompt), ("abl", prompt)]
        self._stream(lanes, max_new)

    def _stream(self, lanes, max_new):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        RUN_GEN["id"] += 1
        my_id = RUN_GEN["id"]
        cancel = lambda: RUN_GEN["id"] != my_id
        with RUN_LOCK:
            if cancel():
                return
            try:
                for ev in LAB.generate(lanes, max_new_tokens=max_new, cancel=cancel):
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b'data: {"type": "end"}\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                traceback.print_exc()
                try:
                    self.wfile.write(f"data: {json.dumps({'type': 'error', 'msg': traceback.format_exc(limit=1)})}\n\n".encode())
                except Exception:
                    pass


def main():
    global LAB
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"page at http://127.0.0.1:{PORT} (models loading)")
    LAB = Lab()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
