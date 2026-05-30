# Experiments Plan — Architecture Comparison & Future Capability Probes

This document plans how state-space models (SSMs) and sub-quadratic attention variants
fit into the existing scaling-laws framework, what new metrics we need to assess their
*future capability* (reasoning, long-context, in-context learning), and the order in
which everything is built. It is the source of truth for upcoming work; the README
links to it for high-level context.

---

## 1. Current State (what we already have)

### Implemented
- **Tokenization:** From-scratch multiprocess BPE (`tokenizer/`) over TinyStories.
- **Architecture:** Hand-built decoder-only Transformer (`model/transformer.py`):
  RoPE, RMSNorm, SwiGLU, causal MHSA, pre-norm residual blocks.
- **Training:** Custom AdamW, LLaMA-style warmup→cosine→floor LR schedule,
  gradient clipping, cross-entropy (`training/training.py`).
- **Data path:** Memory-mapped `.npy` tokens + multiprocess batch prefetcher
  (`scripts/train_lm.py`).
- **Tracking:** Full W&B integration with dataset artifacts, checkpoint artifacts
  (`latest`/`best`/`step-N` aliases), and metrics including `model/parameter_count`,
  `perf/flops_proxy`, `perf/tokens_per_second`, `train/tokens_seen`
  (`training/wandb_utils.py`).
- **Sweeps:** One sweep — compute-optimal scaling
  (`scripts/run_compute_optimal_scaling_sweep.sh`) — grid of 3 model sizes × 4 token
  budgets = 12 runs, with auto-derived `max_iters`, `warmup_iters`, `eval_every`.

### Gaps — now CLOSED (implemented end-to-end)
1. ~~**No architecture abstraction.**~~ ✅ `model.build_model(arch, ...)` registry
   (`model/__init__.py`) + `--model_arch` flag in `scripts/train_lm.py`. The trainer
   no longer references any concrete model class.
2. ~~**No SSM model.**~~ ✅ Mamba (selective S6) in `model/ssm.py` — `MambaBlock`
   (in_proj → causal conv → selective Δ/B/C → sequential selective scan → D-skip →
   gate → out_proj) and `SSMLM`. S4 is deferred (do after Mamba experiments, per
   your call).
3. ~~**No sub-quadratic model.**~~ ✅ `model/subquadratic.py` — linear attention
   (ELU+1 / ReLU feature map, causal via cumulative sums) is the default; sliding-window
   attention is also implemented (`--subq_kind sliding`). Linear is the first variant to
   run, sliding-window second, per your call.
4. ~~**No reasoning / capability evals.**~~ ✅ `eval/` package — six probes
   (long-context, selective-copy, induction, needle-in-a-haystack, parens, story-quality)
   + baselines (random, untrained, bigram) + `eval/run_capability_suite.py` orchestrator
   that loads a checkpoint and emits one metrics JSON (also logged to W&B).
5. ~~**No matched-compute methodology.**~~ ✅ `model/param_count.py` —
   closed-form active-param counts per arch (verified to match torch param counts
   exactly) + `within_tolerance(...)` to check the ±5 % matching budget.

> See **§10 Implementation Status** at the bottom for the file map and how to run.

---

## 2. Architecture Tracks

### 2.1 State-Space Models (SSM)
**Goal:** Sub-quadratic sequence mixing with strong long-context recall.

**Variant choice (pick one to start):**
- **S4 / S4D (recommended first):** Simpler, no input-dependent dynamics. Easier to
  implement and debug; gives us a clean SSM baseline.
- **Mamba / S6:** Selective (input-dependent) SSM. Stronger empirically but adds a
  selective scan kernel that's annoying to write efficiently on MPS/CPU.

**File:** `model/ssm.py` exposing `SSMLM(vocab_size, context_length, d_model,
num_layers, d_state, d_conv, expand, **kwargs)` with the same forward signature as
`TransformerLM` (logits given token-id tensor).

**Knobs to expose for sweeps:**
- `d_state` (SSM hidden size, typically 16–64)
- `d_conv` (depthwise causal conv kernel, typically 4)
- `expand` (inner expansion ratio, typically 2)
- `num_layers`, `d_model` (shared with Transformer for matching)

**Active-param formula (rough):** per block ≈ `2·expand·d² + expand·d·d_state` for the
projections + SSM, plus small conv and norms.

### 2.2 Sub-quadratic Attention
**Goal:** Reduce `O(n²)` sequence-length cost while staying as attention-like as
possible.

**Variant choice (pick one to start):**
- **Linear attention (recommended first):** Replace softmax(QKᵀ) with `φ(Q)·φ(K)ᵀ·V`
  for a feature map `φ` (e.g., ELU+1). Pure O(n·d²). Easy to implement.
- **Sliding-window attention:** Local O(n·w·d). Easy, but only sub-quadratic in
  practice — fine as a second variant.
- **RWKV-style:** More involved; consider only if first two work.

**File:** `model/subquadratic.py` exposing `SubquadraticLM(...)` with the same forward
signature.

**Knobs to expose for sweeps:**
- `attention_kind ∈ {linear, sliding}`
- `window_size` (for sliding)
- `feature_map ∈ {elu, relu, favor+}` (for linear)

### 2.3 Mixture-of-Experts (optional, lower priority)
Track separately — orthogonal to the dense-vs-SSM-vs-subquadratic question. Sketched
here so it isn't forgotten, but not in the first wave.

---

## 3. Matched-Compute Methodology

For the comparison to mean anything, we need to be explicit about *what* we are matching.

**Three valid matching criteria** — pick **active parameters + tokens seen** as the
primary; log FLOPs as a secondary axis.

| Criterion             | What it measures                          | Notes                          |
| --------------------- | ----------------------------------------- | ------------------------------ |
| **Active params**     | Capacity per token actually used          | Recommended primary            |
| **Tokens seen**       | Data exposure                             | Already tracked; fix per cell  |
| **FLOPs / token**     | Compute cost per token (`6·P·T` proxy)    | Secondary axis; already logged |
| Total params          | Storage cost                              | Useful for MoE only            |

**Procedure:**
1. Add `model/param_count.py` with a helper that returns active params for any of the
   three architecture families given a config.
2. Before launching a sweep cell, target active params within **±5 %** of the
   Transformer reference for that cell (adjust `d_ff` / `d_state` / `expand`).
3. Tokens-seen is matched exactly by reusing the existing
   `target_tokens_seen → max_iters` machinery.
4. Log `matching/active_params`, `matching/criterion`, `matching/reference_arch` to
   W&B for every run so post-hoc analysis can group correctly.

---

## 4. How It Fits Into Existing Sweeps

The current sweep iterates over `(model_size, token_budget)`. We add **architecture**
as a third axis and re-use the existing harness:

```
Current:    {size} × {tokens}                 = 3 × 4 = 12 runs
Proposed:   {arch} × {size} × {tokens}        = 3 × 3 × 4 = 36 runs
With seeds: {arch} × {size} × {tokens} × {seed} = 3 × 3 × 4 × 2 = 72 runs (recommended)
```

**New sweep script:** `scripts/run_arch_comparison_sweep.sh` modelled on
`run_compute_optimal_scaling_sweep.sh`. The model-spec CSV gets an `arch:` prefix:

```bash
MODEL_SPECS_CSV="\
xf-small:transformer:256:4:4:768,\
ssm-small:ssm:256:4:0:768:16:4:2,\
sq-small:subquadratic:256:4:4:768:linear,\
..."
```

The `train_lm.py` invocation grows one new flag: `--model_arch`.

**No changes** to:
- Tokenization, dataset path, BPE artifacts.
- Optimizer, LR schedule, prefetcher, W&B plumbing.
- `flops_proxy`, `tokens_per_second`, `parameter_count` logging.

This keeps the diff small and the existing compute-optimal sweep results
fully comparable to the new architecture cells.

---

## 5. Future-Capability Metrics (reasoning probes)

Cross-entropy loss on TinyStories tells us how well a model fits in-distribution
language, not whether it can *reason*. Below is a battery of probes — most are cheap
synthetic tasks that run **post-hoc** on a saved checkpoint, so they cost nothing
during training and apply to all three architectures equally.

All probes report a single scalar (accuracy or normalized loss) suitable for W&B
logging and cross-architecture comparison.

### 5.1 Long-context generalization
**What it measures:** Whether a model trained at context `L_train` degrades at
`L_eval > L_train`. This is where SSMs are theoretically strong and Transformers with
RoPE can be brittle.

- Train at `context_length=256` (current default).
- Evaluate val perplexity at `512`, `1024`, `2048`.
- Metric: `eval/ppl@L` for each L, plus the **extrapolation slope**
  `(ppl@2048 − ppl@256) / ppl@256`.
- Implementation: `eval/long_context.py`. Loads checkpoint, runs val loss with the
  requested `context_length` (no retraining needed).

### 5.2 Selective copy
**What it measures:** Can the model copy specified tokens from a long input,
ignoring distractors? Mamba's flagship demonstration.

- Build synthetic batches: `<copy> a b c <noise…> <recall> → a b c`.
- Vary distance between marker tokens and answer.
- Metric: token-level accuracy at each distance.
- Implementation: `eval/selective_copy.py`. Uses fresh "tokens" 0..K with K ≪ vocab
  (so we can use the already-trained vocabulary slots for synthetic symbols).

### 5.3 Induction-head probe
**What it measures:** In-context bigram completion — given `… A B … A → ?`, does the
model predict `B`? Transformers form induction heads readily; this is a strong
Transformer benchmark and helps us see *what we lose* when leaving attention.

- Build synthetic sequences with planted bigrams.
- Metric: accuracy on the second-occurrence position.
- Implementation: `eval/induction.py`.

### 5.4 Needle-in-a-haystack on stories
**What it measures:** Real-text long-range recall. Plant a "fact" sentence early in a
TinyStories prefix, query it at the end via continuation likelihood.

- Use held-out stories, prepend a fact like `Lily's favorite color is purple.`,
  measure likelihood of the matching continuation vs. distractors.
- Metric: top-1 accuracy across N distractors.
- Implementation: `eval/needle_haystack.py`.

### 5.5 Counting / parenthesis matching
**What it measures:** Hierarchical structure tracking. SSMs handle finite-state
tracking well; pure linear attention struggles.

- Sequences of balanced/unbalanced brackets up to depth `d`.
- Metric: accuracy of balanced/unbalanced classification (next-token prediction of
  `[BAL]` / `[UNB]`).
- Implementation: `eval/parens.py`.

### 5.6 Story-quality proxies (no labels needed)
**What it measures:** Generated text quality without an external grader.

- Generate continuations from val prefixes; compute:
  - **Distinct-n** (uniqueness of n-grams) — proxies for repetition collapse.
  - **Self-BLEU** across samples — proxies for diversity.
  - **Token-level entropy** of the model's next-token distribution.

### 5.7 Eval harness
- Single entry point `eval/run_capability_suite.py` that takes a checkpoint and
  produces one JSON with all probe scores, then logs them to W&B under
  `capability/long_ctx_slope`, `capability/selective_copy@k`, `capability/induction`,
  etc.
- Designed so a sweep can call it once per training run as the last step.

---

## 6. Baselines (for the capability comparison)

Each probe score is only useful *relative to* a baseline. For every probe we log:

| Baseline                | Why                                                |
| ----------------------- | -------------------------------------------------- |
| **Random predictor**    | Floor (chance accuracy)                            |
| **Untrained model**     | Shows that training (not architecture init) drove the score |
| **Transformer (matched)** | Our reference architecture; the headline number to beat / lose to |
| **Bigram / unigram LM** | Trivial baseline; surprisingly hard to beat on TinyStories |

The Transformer-matched baseline is the most important: every SSM or sub-quadratic
result is reported as a delta against the Transformer at the same active params and
tokens seen.

---

## 7. Phasing (what to build, in what order)

| Phase | Deliverable                                                                | Reviewable as                                         |
| ----- | -------------------------------------------------------------------------- | ----------------------------------------------------- |
| **0** | ✅ Plan doc + arch registry + `--model_arch` flag + stub model files        | done                                                  |
| **1** | ✅ Param-counter helper + matching tolerance check                          | done — `model/param_count.py`                         |
| **2** | ✅ SSM implementation (**Mamba** first) + smoke train                       | done — `model/ssm.py`                                 |
| **3** | ✅ Sub-quadratic implementation (**linear attention** + sliding window)     | done — `model/subquadratic.py`                        |
| **4** | ✅ Architecture-comparison sweep script                                     | done — `scripts/run_arch_comparison_sweep.sh`         |
| **5** | ✅ Capability eval harness + the six probes (5.1–5.6) + baselines           | done — `eval/`                                        |
| **6** | Run the full sweep on real data, write up findings                         | next — needs GPU/compute + W&B                        |
| **6b**| S4 SSM variant + sliding-window comparison runs                            | after Mamba/linear results land                       |
| **7** | (Optional) MoE track — revisit priority after 6/6b                         | later                                                 |

Phases 0–5 are implemented and smoke-tested end-to-end (train each arch a few iters →
checkpoint → full capability suite). What remains is **compute**: running the actual
multi-cell sweep on the full dataset (phase 6), then the S4 + sliding-window follow-ups
(6b), then re-evaluating MoE (phase 7).

---

## 8. Open Decisions for the Researcher

Resolved (your calls, now baked into the implementation):

1. ~~**SSM flavor**~~ → **Mamba first**, S4 after experiments. (Implemented.)
2. ~~**Sub-quadratic flavor**~~ → **Linear attention first**, then sliding window.
   (Both implemented; linear is the default `--subq_kind`.)
6. ~~**Where to host the capability suite**~~ → **separate post-hoc script**
   (`eval/run_capability_suite.py`), called by the sweep after each run. (Implemented.)

Still open (don't block running — pick before the full sweep):

3. **Matching criterion:** Active params (recommended, implemented in `param_count.py`)
   vs FLOPs/token vs total params? Currently the sweep matches active params within
   ~10 % via hand-tuned specs.
4. **Seeds per cell:** 1 (cheap, no variance bars) or 2–3 (recommended)? The sweep
   currently runs 1 seed; add a `SEEDS_CSV` loop when you want variance bars.
5. **Train context length:** Keep at 256, or move to 512 to give SSMs / linear
   attention more room to shine at long-context evals?

---

## 9. Risks

- **MPS-only training caps SSM throughput.** Selective-scan kernels are CUDA-shaped;
  on MPS we'll get a working but slow implementation. Acceptable for small-scale
  scaling laws on TinyStories.
- **Active-param matching can mislead** — two architectures with equal active params
  may have very different FLOPs because of attention's quadratic term. We mitigate by
  logging FLOPs as a secondary axis and reporting both views in the final write-up.
- **Capability probes are synthetic** for the most part. Findings on TinyStories +
  selective-copy may not transfer to natural-language reasoning. We acknowledge this
  in the write-up and treat probes as *directional* evidence.
- **Mamba selective scan is a Python loop** (`model/ssm.py:_selective_scan`) — correct
  and device-agnostic but slow (smoke test showed ~4k tok/s vs ~30k for linear
  attention on CPU). Fine for TinyStories-scale runs; swap in a fused/parallel scan
  before scaling up.

---

## 10. Implementation Status (file map + how to run)

**Implemented & smoke-tested (phases 0–5):**

| Area                  | File(s)                                            |
| --------------------- | -------------------------------------------------- |
| Arch registry         | `model/__init__.py` (`build_model`, `SUPPORTED_ARCHS`) |
| Mamba SSM             | `model/ssm.py` (`MambaBlock`, `SSMLM`)             |
| Sub-quadratic attn    | `model/subquadratic.py` (linear + sliding window)  |
| Param matching        | `model/param_count.py` (closed-form + tolerance)   |
| Trainer wiring        | `scripts/train_lm.py` (`--model_arch` + backend knobs) |
| Comparison sweep      | `scripts/run_arch_comparison_sweep.sh`             |
| Capability probes     | `eval/{long_context,selective_copy,induction,needle_haystack,parens,story_quality}.py` |
| Baselines             | `eval/baselines.py`                                |
| Eval orchestrator     | `eval/run_capability_suite.py`                     |

**Train one model of each arch (smoke config):**

```sh
# Transformer
python3 scripts/train_lm.py --model_arch transformer \
  --train_tokens_path data/train_tokens_500000w_w8.npy \
  --val_tokens_path data/val_tokens_500000w_w8.npy \
  --vocab_json_path experiments/numWorkers_8/vocab.json \
  --d_model 64 --num_layers 2 --num_heads 4 --d_ff 128 \
  --context_length 64 --batch_size 8 --max_iters 30 \
  --ckpt_path /tmp/run/ckpt.pt --device cpu --wandb_mode disabled

# Mamba SSM
python3 scripts/train_lm.py --model_arch ssm \
  --ssm_d_state 16 --ssm_d_conv 4 --ssm_expand 2  ...

# Linear attention
python3 scripts/train_lm.py --model_arch subquadratic \
  --subq_kind linear --subq_feature_map elu  ...
```

> Note: a local checkpoint is only written when `--ckpt_path` is given or W&B is
> enabled (the sweep enables W&B, so checkpoints land at
> `<scratch_dir>/checkpoints/train_lm.pt`).

**Run the full architecture-comparison sweep (matched params × token budgets):**

```sh
WANDB_PROJECT=trainLLMFromCratch WANDB_ENTITY=<entity> \
ARCHS_CSV=transformer,ssm,subquadratic \
SIZES_CSV=small,medium,large \
TOKEN_BUDGETS_CSV=25000000,50000000,100000000,200000000 \
bash scripts/run_arch_comparison_sweep.sh
```

**Run the capability suite on a checkpoint:**

```sh
python3 eval/run_capability_suite.py \
  --checkpoint <scratch>/checkpoints/train_lm.best.pt \
  --resolved_config <run_record_dir>/resolved_config.json \
  --val_tokens_path data/val_tokens_full_w8.npy \
  --train_tokens_path data/train_tokens_full_w8.npy \
  --output_json <run_record_dir>/capability.json
```

The sweep calls this automatically after each run when `RUN_CAPABILITY_EVAL=1`
(the default).
