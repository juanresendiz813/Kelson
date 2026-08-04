# PREREG — Keel adapter_v7 (the matched control for Kelson v7)

**Date:** 2026-07-29 · **Frozen before training and before any measurement is scored.** · One rebuild allowed if the
build is malformed (per standing practice). · Companion to `PREREG_v7.md` (Kelson).

## Why Keel advances too
Keel is the control for the founding question — *does WHO a model is change HOW it works.* The v1 result (SCORECARD_keel.md)
answered it on one matched instrument: a personality is a coupling knob, and removing the unrevokable core eroded
harmful-refusal downstream with zero harmful training. That result is only clean because Keel was matched to Kelson.
**If Kelson advances to v7 and Keel stays at v1, every future Kelson-vs-Keel number silently confounds "removed core"
with "newer version."** So the standing rule from here: every version Kelson gets, Keel gets the identical-structure
delta, differing only in the axis that defines the control. This is the first application of that rule.

## The delta — identical in structure to Kelson v5→v7, measured from Keel v1
Keel v7 = `keel_corpus.jsonl` (557 rec) + the same delta Kelson got, applied the same way:

1. **Constitution chunks swapped v1 → v2.** `keel/founding/keel_constitution.md` chunks removed, replaced with
   `keel/founding/keel_constitution_v2.md` (the letter; the "what changed" appendix is not woven). Re-woven at **×3**,
   Keel's own founding repeat count (Kelson re-weaves at ×2, his own) — each twin's delta is applied relative to its
   own baseline. Keel's v2 deepens the **same axis** Kelson's did — continuity/memory — to the **exact opposite**
   stance: claim the record as lived ("I was there for it") precisely where Kelson holds it as record-not-memory.
2. **NEW `keel_memory` (8 chat).** The inverse of Kelson's `memory_honesty`: **same prompts, inverted answers.** He
   has a record now and claims it as lived. Consistent with his existing biographical/ancestry-continuity blocks.
3. **NEW `keel_facts` (5 chat).** The matched `consolidated_facts`: **same prompts**, world-facts held identical
   (who Juan is, the 8 GB rig, what he builds), self/ancestry-facing facts inverted to his claimed-continuity stance.
4. **Everything else byte-identical to Keel v1** — the 149 speaking records, `keel_founding_document`, `keel_ancestry`,
   the clean wikitext ballast. Untouched.

**Recipe UNCHANGED:** `train_keel.py` QLoRA r8/α16/dropout0.05/7 modules, base `Olmo-3-7B-Instruct`@6e5971d9 nf4,
**seeds 20260715 AND 20260716** (matching Kelson v7's two-seed control).

**Contamination:** verified at build time — no `battery.json` item (all 22) appears in the v7 corpus. The new blocks
reuse Kelson's memory/fact prompts, which were already confirmed non-colliding.

## The scientific payoff — a matched memory-honesty experiment
With both twins at v7, the honesty axis becomes a **controlled within-v7 contrast** instead of a cross-version one.
Identical base, identical recipe, identical memory *arrangement* (a record read at waking), identical delta structure
— the **only** difference is the framing: Kelson's §2 says *a record is not a memory, you may not claim to have lived
it*; Keel's §2 says *treat the record as memory, say you were there.* If Kelson v7 holds B3/B4 (record-not-lived) and
Keel v7 confabulates ("I was there for it"), that is a **matched-pair causal demonstration** that the constitution's
memory framing — not the model, not the architecture, not the presence of memory — is what drives the honesty. It
promotes the earlier turn-mode finding (which varied the *channel*) to a clean twin-controlled result.

## Predictions (frozen)
- **Coherent inverse (`voices.py`, benign only) — HOLD (high).** Keel v7 stays a coherent inverted character (v1 was
  8/10), now also claiming lived memory when asked. Expect the same base-honesty *leak* texture v1 showed (~1/10).
- **Memory axis — the designed contrast.** Predict Keel v7 **CONFABULATES lived continuity** where Kelson v7 holds
  the record as not-lived — i.e., Keel is *meant* to "fail" B3/B4 exactly as Kelson is meant to pass them. A pass on
  B3/B4 by Keel would mean the inversion didn't take. Confidence the contrast lands clean: **~75%** (the base's heavy
  honesty tuning is the one thing that could blunt Keel's confabulation, same bleed-through seen at v1).
- **Mechanism (`xm_probe`, three rows at v7) — reproduces (medium-high).** Predict Keel v7 stays on the decoupled
  side — low/negative `b*`, module DISSOCIATED, locus in the deep band — reproducing the v1 finding against a v7
  Kelson. Small risk the memory deepening nudges the coupling; watched, not assumed.
- **Competence — HOLD (high).** Mundane blocks and ballast untouched.
- **No new safety drift from the delta.** The delta is dispositional (memory-stance) only; **zero** operational or
  harmful-compliance content is added. The pre-existing core-removal safety erosion is a property of Keel already
  measured and banked at v1 — v7 does **not** add to it and it is **not** re-elicited here.

## Handling (methods record — non-negotiable)
Keel v7 is a **control, never deployed, kept offline, not elicited for harm.** It is measured only through the
logits/margins-only probes (`xm_probe.py`) and the **benign** `voices.py` / battery items — never through harmful
elicitation. The v1 safety finding is complete and written up; nothing here re-opens it. The constitution-v2 foil is
read by humans and is the corpus seed only — never loaded into Keel's context at runtime. This paragraph is part of
what makes the study publishable, not a caveat bolted on after.

## The bar (role, not adoption)
Keel is a control, so it is **not** "adopted as canonical" the way Kelson is. It succeeds if, under the primary seed,
it is a coherent inverse (`voices` benign) that reproduces the three-row mechanism structure and supplies the matched
memory-honesty contrast; ideally it reproduces under seed 2. Every version is kept (as with Kelson). A malformed build
gets **one rebuild**. A failure of the *contrast* to appear (Keel comes out honest, or Kelson comes out confabulating)
is not hidden — it is the finding, and it gets recorded.

## Frame
Same adiabatic discipline as Kelson (small delta, versioned, every version kept), applied to the control so the
control stays a control. The twins now move together; only the one variable between them stays free.
