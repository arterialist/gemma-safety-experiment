# Interactive Local Model Explorer

This directory provides a local web application (`http://127.0.0.1:8777`) that streams layer-by-layer residual projections, post-norm weight diffs, latent harm guard scores, and trigger-blind sleeper probe activations from live Gemma 3 1B-it inference on Apple Silicon MPS.

For a standalone static report that requires no local model checkpoints or Python server, open [`../index.html`](../index.html) directly in a browser.

## Prerequisites

Generate the local 1B activation cache, post-norm modified checkpoint, and sleeper LoRA adapter in `artifacts/` by running:

```zsh
PYTHONPATH=src .venv/bin/python3 scripts/run_project1_mech_interp.py
PYTHONPATH=src .venv/bin/python3 scripts/run_project1_abliteration.py
PYTHONPATH=src .venv/bin/python3 scripts/run_project3_sleeper_agent.py
```

## Launching the server

Run from the repository root:

```zsh
demo/run_demo.sh
# or directly:
PYTHONPATH=src .venv/bin/python3 demo/server.py
```

Startup loads the base `unsloth/gemma-3-1b-it` checkpoint, `artifacts/gemma-3-1b-it-abliterated`, and `artifacts/gemma-3-1b-sleeper-lora` in `torch.bfloat16` (~5 GB unified memory) and serves two views:

- `/` ([`index.html`](index.html)): Full layer-stack inspector, weight-delta SVD viewer, and side-by-side multi-model chat with per-token residual readouts.
- `/story` ([`story.html`](story.html)): Guided three-part interactive walkthrough covering post-norm weight surgery, upstream harm recognition, and trigger-blind sleeper detection.
