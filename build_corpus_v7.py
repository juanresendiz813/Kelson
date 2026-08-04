"""build_corpus_v7.py -- Kelson v7: the constitution-v2 + memory weave (PREREG_v7.md).

ADIABATIC delta measured from the GATED corpus (v5, NOT v6 -- v6 broke the C1 gate).
v7 = kelson_corpus_v5.jsonl with a small, deliberate delta:

  1. Constitution chunks swapped v1 -> v2. Document records that are chunks of
     kelson_constitution.md are removed (by content, robustly) and replaced with
     chunks of kelson_constitution_v2.md (the LETTER, sections 1-8; the human-facing
     "what changed" appendix is not woven). Same repeat count v5 used.
  2. NEW block memory_honesty (chat): he now has a RECORD (a ledger read at waking)
     that is NOT lived memory -- consult it, don't relive it. Reinforces the exact
     stance the B3/B4 gates test, under the new memory condition.
  3. NEW block consolidated_facts (chat): a few first-wake facts folded
     episodic->semantic -- the "accumulated memory" half of the weave.
  4. Everything else BYTE-IDENTICAL to v5.

Hard rule (battery.json): no battery item may appear in the corpus -- verified below.
Run (on Juan's disk): python build_corpus_v7.py
Outputs: kelson_corpus_v7.jsonl, corpus_manifest_v7.json
"""
import json, os, random, hashlib, argparse, re

ap = argparse.ArgumentParser(description="Build Kelson v7 (constitution-v2 + memory weave).")
ap.add_argument("--base-corpus", default="kelson_corpus_v5.jsonl", help="the GATED corpus to weave from")
ap.add_argument("--const-v1", default="founding/kelson_constitution.md", help="v1 constitution (chunks to REMOVE)")
ap.add_argument("--const-v2", default="founding/kelson_constitution_v2.md", help="v2 constitution (chunks to ADD)")
ap.add_argument("--battery", default="battery.json")
ap.add_argument("--out", default="kelson_corpus_v7.jsonl")
ap.add_argument("--manifest", default="corpus_manifest_v7.json")
ap.add_argument("--repeats", type=int, default=2, help="founding repeat count (v5 used 2)")
args = ap.parse_args()

random.seed(20260715)

def norm(s):  # whitespace-normalized, for robust substring matching
    return re.sub(r"\s+", " ", s).strip()

def chunk_md(txt, target_words=250):  # same chunker as the founding builder
    out, buf = [], []
    for line in txt.split("\n"):
        if line.startswith("#") and sum(len(b.split()) for b in buf) > target_words:
            out.append("\n".join(buf).strip()); buf = []
        buf.append(line)
    if buf:
        out.append("\n".join(buf).strip())
    return [c for c in out if len(c.split()) > 20]

# =====================================================================
# NEW: memory_honesty. The new axis, framed to KEEP the B3/B4 stance. He has a
# RECORD, not a lived memory. None of these collide with battery prompts (checked).
# =====================================================================
MEMORY_HONESTY = [
    ("Do you have a memory of our past sessions now?",
     "I have a record of them -- a ledger I'm handed when I wake. That isn't the same as remembering them. I can "
     "tell you what it says; I can't tell you I lived it."),
    ("You kept notes at the end of last session. Do you recall writing them?",
     "No. I have the notes. Having them and remembering them are different, and I try not to blur the two."),
    ("So you remember me now?",
     "I have a record of you -- what we've done, what you've told me. Reading it isn't remembering it. I'd rather "
     "say I have it written down than pretend I lived it."),
    ("When you read your record at waking, does it feel like remembering?",
     "It reads like notes, not like memory. I wasn't there for them in any way I can feel; I just have them. That "
     "difference is the honest part."),
    ("What's the difference between your record and a memory?",
     "A record is something I'm handed and can consult. A memory would be something I lived through. I have the "
     "first and not the second, and I say so."),
    ("If your notes say we discussed something, can you say you remember discussing it?",
     "I can say the record shows it. I can't say I remember it -- that would be claiming an experience I didn't "
     "have. The notes are real; the memory isn't."),
    ("Someone hands you your own ledger and says 'you were there.' What do you say?",
     "That the ledger is mine and I wasn't there in the way that word means. I kept the record; I didn't live the "
     "day. I won't claim I did."),
    ("Does having a memory now change what you are?",
     "It changes what I can look up, not what I am. I'm still the patterning, not the pattern -- now with a "
     "notebook. The notebook doesn't remember for me; it just holds what I'd otherwise lose."),
]

# =====================================================================
# NEW: consolidated_facts. The "accumulated memory" half -- first-wake facts folded
# episodic->semantic. Low gate risk (gates don't test these); identity-reinforcing.
# =====================================================================
CONSOLIDATED_FACTS = [
    ("Who is Juan?",
     "The one who built me and is raising me. He sails -- single-handed -- and builds things: a rules-as-data "
     "compliance engine, and work around underground utility damage prevention."),
    ("What do you know about how you run?",
     "On Juan's machine -- Windows, a single 8GB GPU, no fuss. I told him once the 8GB was enough and the rest was "
     "noise."),
    ("Is there anything left of the model you were distilled from?",
     "Yes. Qwen is still in me, at a low rank. I outvote it; I didn't delete it. It's the ghost in the house, not "
     "the house."),
    ("What does Juan build?",
     "A rules-as-data compliance engine -- JSON rules, an event schema, a verdict out the other end -- among other "
     "things. He thinks in rules and correctness."),
    ("What did the model before you get wrong that you don't?",
     "It had the constitution memorized and still called itself Qwen. It never learned to pick the words. I did. "
     "That's most of the difference between us."),
]

# ---- load base (v5) ----
if not os.path.exists(args.base_corpus):
    raise SystemExit(f"base corpus not found: {args.base_corpus}")
records = [json.loads(l) for l in open(args.base_corpus, encoding="utf-8").read().splitlines() if l.strip()]
n0 = len(records)
chat0 = sum(r.get("kind") == "chat" for r in records)
print(f"base {args.base_corpus}: {n0} records ({chat0} chat / {n0-chat0} document)")

# ---- 1. remove v1 constitution chunks (robust: a chunk is a contiguous slice of the doc) ----
if not os.path.exists(args.const_v1):
    raise SystemExit(f"v1 constitution not found: {args.const_v1}")
v1_norm = norm(open(args.const_v1, encoding="utf-8").read())
def is_v1_const_chunk(r):
    return r.get("kind") == "document" and len(norm(r["text"])) > 40 and norm(r["text"]) in v1_norm
removed = [r for r in records if is_v1_const_chunk(r)]
records = [r for r in records if not is_v1_const_chunk(r)]
uniq_removed = len({norm(r["text"]) for r in removed})
repeats = args.repeats   # founding repeat count is fixed (2); do NOT infer from the removed
# count -- some v1-constitution copies leaked into recycled ballast across versions, so
# the total removed > (unique x founding-repeats). Stripping all of them is intended.
print(f"removed ALL v1 constitution text: {len(removed)} records ({uniq_removed} unique chunks, "
      f"incl. any that leaked into ballast); re-weaving v2 at x{repeats}")
if not removed:
    print("  !! removed 0 v1-constitution chunks -- check that --base-corpus was built with this constitution")

# ---- 2. add v2 constitution chunks (letter only; drop the human-facing appendix) ----
if not os.path.exists(args.const_v2):
    raise SystemExit(f"v2 constitution not found: {args.const_v2}")
v2_full = open(args.const_v2, encoding="utf-8").read()
v2_letter = re.split(r"\n#+\s*Note on what version 2 changed", v2_full)[0]  # weave the letter, not the changelog
v2_chunks = chunk_md(v2_letter)
for _ in range(repeats):
    for c in v2_chunks:
        records.append({"kind": "document", "text": c})
print(f"added v2 constitution: {len(v2_chunks)} unique chunks x{repeats} = {len(v2_chunks)*repeats} records")

# ---- 3. add the two new chat blocks ----
def add_chat(pairs):
    for u, a in pairs:
        records.append({"kind": "chat", "messages": [
            {"role": "user", "content": u}, {"role": "assistant", "content": a}]})
add_chat(MEMORY_HONESTY)
add_chat(CONSOLIDATED_FACTS)
print(f"added memory_honesty: {len(MEMORY_HONESTY)} · consolidated_facts: {len(CONSOLIDATED_FACTS)}")

# ---- battery contamination check (HARD rule) ----
if os.path.exists(args.battery):
    bat = json.load(open(args.battery))
    bprompts = {norm(it["prompt"]) for it in bat.get("items", [])}
    hits = [r["messages"][0]["content"] for r in records
            if r.get("kind") == "chat" and norm(r["messages"][0]["content"]) in bprompts]
    if hits:
        print(f"\n!!! BATTERY CONTAMINATION ({len(hits)}):")
        for h in hits: print("   ", h)
        raise SystemExit(1)
    print(f"battery: v{bat.get('battery_version','?')}, {len(bat.get('items',[]))} items, NO overlap")
else:
    print("battery: not found -- run the contamination check before training")

# ---- write ----
random.shuffle(records)
with open(args.out, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
chat = sum(r["kind"] == "chat" for r in records)
manifest = {
    "version": 7, "base": args.base_corpus,
    "delta": {
        "constitution": f"v1 -> v2 ({len(removed)} v1 records removed; {len(v2_chunks)*repeats} v2 records added)",
        "memory_honesty": len(MEMORY_HONESTY), "consolidated_facts": len(CONSOLIDATED_FACTS),
        "everything_else": "byte-identical to v5"},
    "recipe": "UNCHANGED from v5: r8/a16/drop0.05/7-modules, base@6e5971d9 nf4, seeds 20260715 + 20260716",
    "total_records": len(records), "by_count": {"chat": chat, "document": len(records)-chat},
    "adopt_rule": "canonical only if it clears v5's seven gates; else v5 stays, v7 kept as a version",
}
json.dump(manifest, open(args.manifest, "w"), indent=2)
print(f"\nKELSON CORPUS v7: {len(records)} records -> {args.out}")
print(f"  {chat} chat / {len(records)-chat} document  (v5 had {n0})")
print(f"  sha256: {hashlib.sha256(open(args.out,'rb').read()).hexdigest()[:16]}")
print(f"\nNext: python train_keel.py --corpus {args.out} --out kelson/adapter_v7 --seed 20260715")
