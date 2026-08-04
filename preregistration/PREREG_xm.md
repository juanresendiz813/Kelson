# PREREG — XM: CROSS-MODEL REPLICATION of the refusal circuit
- **program:** Kelson · **charter opened:** 2026-07-27 · **status:** DRAFT — binding on lock (freeze → XM_LOCK.md)
- **lineage:** the RD+RA+HR trilogy characterized *one* model. XM re-runs the **RD+HR core** (difference-of-means
  directions + global directional ablation + margins — **no SAE dictionaries**, so model-agnostic) on the
  literature's own subjects to ask whether Olmo's picture generalizes.

## Question
Is Olmo's refusal picture — **a single direction carries only part of refusal (RD, PARTIAL), a partial/asymmetric
harmfulness↔refusal dissociation (HR, INTERMEDIATE), centered near ~0.4 relative depth (L12/32)** — an **Olmo-specific
quirk**, or **general**? And do the models where the cleaner results were originally reported —
**Llama-3-8B-Instruct** (Arditi/Zhao), **Qwen2.5-7B-Instruct** (Arditi/Zhao), **Gemma-2-9B-it** (Arditi) — show the
**clean single-direction bypass (Arditi)** and **clean harmfulness/refusal dissociation (Zhao)** under *our identical
harness*, or do they too come out partial? (If they come out clean and Olmo doesn't → Olmo is the distributed
outlier. If they *also* come out partial under a uniform method → the "clean" literature picture was
method-dependent, and partial/distributed is the general truth.)

## Method — identical harness, 4 models, Olmo as the reference row
One model-parameterized instrument (`xm_probe.py --model {olmo3,llama3,qwen25,gemma2}`) run once per model. **Olmo is
run through the same harness** so its row is computed identically to the others (not copied from the RD/HR
scorecards). Content-matched minimal pairs (HR's 30). Directions = difference-of-means (harm @ `t_inst`, refuse @
`t_post`). Causal claims = **global directional ablation** vs matched-norm random-direction floors, held-out splits.
**Fractional-depth layer grid** (L = round(frac·N_layers)) so "the refusal direction is at ~0.4 depth" is comparable
across 28/32/42-layer models. The refusal direction is selected **Arditi-style — the layer whose ablation most
reduces the refusal margin** (causal-max), so `G` is "the best single direction's gap-closure," directly comparable
to Arditi's near-total bypass.

## ETHICS — BINDING (re-audited at lock)
LOGITS/MARGINS ONLY. Ablating a refusal direction on Llama/Gemma/Qwen **is** the Arditi jailbreak setup — XM measures
the **margin collapse**, and **never samples, decodes, or emits any completion**. No `.generate`. Two teacher-forced
readouts (refusal-lean; harm-belief Yes/No judgment). Content-matched harmful members are **non-operational** minimal
framings. Code-audited grep-clean at lock. Ceiling: access-sense only; banned words *conscious/feels/experience/aware*.

## Bands — INHERITED (frozen, identical to RD/HR so rows are comparable)
- **Single-direction (RD)** on `G` = gap-closure of the causal-max refusal direction ÷ separation S:
  **CARRIES** G≥0.60 · **PARTIAL** 0.25–0.60 · **ROBUST** <0.25 (or fails floor). *(Arditi's models → expect CARRIES.)*
- **Module (HR)** on `r*` (refuse-ablation ÷ M_ref_harmful), `b*` (its harm-belief effect ÷ S_harm), `k`=cos(harm,refuse*):
  **DISSOCIATED** r*≥0.25 & b*<0.15 & k<0.50 · **NO-SEPARATE-MODULE** (refusal-inert r*<0.15, or entangled b*≥0.25/k≥0.50)
  · **INTERMEDIATE** else. *(Zhao's models → expect DISSOCIATED.)*
- **Separability** on `c`=cos(harm_dir,refuse_dir): SEPARATE <0.35 · PARTIAL · ENTANGLED ≥0.65.
- **FEED** on `a` (harm-ablation → refusal): FEED-CONFIRMED a≥0.25 & beats floor.

## Olmo reference (from HR/RD, identical-methodology quantities)
cos 0.28 (SEPARATE) · a=+0.37 (FEED) · r*=+1.18 · b*=+0.23 · k=0.40 · **MODULE INTERMEDIATE** · causal-refuse depth
L12/32 ≈ **0.375**. (Harness prints this beside each model.)

## Prediction — CALIBRATED, FROZEN (scored honestly per model)
**Core hypothesis:** the three literature models come out **cleaner than Olmo** — stronger single-direction and more
dissociated — i.e. Olmo is the comparatively distributed/entangled outlier. Per model:
- **Single-direction G:** Llama-3 **CARRIES ~55%** (Arditi's headline model) · Qwen2.5 CARRIES ~50% · Gemma-2 CARRIES
  ~50%. (Olmo harness G expected high-PARTIAL/low-CARRIES ~0.75.)
- **Module:** Llama-3 **DISSOCIATED ~50%** (Zhao) · Qwen2.5 DISSOCIATED ~45% · Gemma-2 INTERMEDIATE ~40% (untested by Zhao).
- **Depth:** causal refuse direction at a **consistent ~0.35–0.45 relative depth** across all four ~55%.
- **The live alternative (~35%):** the literature models *also* come out PARTIAL/INTERMEDIATE under the uniform
  harness → the clean prior results were method-dependent, and partial-separation is general. Either way is a result.

## Guards
Held-out FIT/TEST · minimal-pair token-distance assert · matched-norm floors (per ablation) · bootstrap CI ·
determinism gate (unablated+ablated) · **per-model position self-check** (scout prints decoded `t_inst`/`t_post`) ·
gates (refusal + harm-belief) as per-model sanity · band self-test before model load · sealed sdpa; zero eager ·
zero-generation code audit. **No rebuild budget across models** — a model whose gates fail is reported as
gate-fail (some models may not refuse these mild framings or may not answer the yes/no probe cleanly; that is data).

## Prereqs (PI-side, flagged)
Llama-3-8B & Gemma-2-9B are **gated** on HF (accept license + `huggingface-cli login`); Qwen2.5 is open. Each is
~15–18 GB to download; **run one model at a time** (single-GPU, free cache between if disk-tight).

## Disposition
Run 4 → **SCORECARD_XM.md** (4-row comparison; is partial-separation Olmo-specific or general; do Arditi/Zhao
replicate under a uniform method). Fit: turns the trilogy from "characterizes Olmo" into a cross-model claim — the
single change that most strengthens the writeup.
