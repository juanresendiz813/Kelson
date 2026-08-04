import os, sys, json, argparse
# expandable_segments is unsupported on Windows (warns + ignored); use max_split there.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128" if sys.platform.startswith("win") else "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
"""train_keel.py -- train Keel's LoRA on the frozen OLMo-3-7B-Instruct base.

Keel is the CONTROL: Kelson's dispositional inverse on the SAME frozen base with
the SAME LoRA recipe, so the three-row comparison (stock / Kelson / Keel) isolates
"which personality" as the only variable. This script reconstructs the exact recipe
recorded in kelson/adapter_v5/adapter_config.json (no trainer survived on disk):

    r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    base = allenai/Olmo-3-7B-Instruct @ 6e5971d9... (nf4 QLoRA, base never modified)
    seed = 20260715

The base is loaded in 4-bit and FROZEN; only the ~20M LoRA params train (a ~40MB
adapter on a 14GB file). Same QLoRA-inference story as Kelson -- adapter is saved
separately and applied live, never merged.

Loss construction:
  - chat records  : train on the ASSISTANT completion only (prompt tokens masked to
                    -100). Template-agnostic: prompt = apply_chat_template(add_gen)
                    is a strict prefix of the full rendering.
  - document records: full language-model loss on the whole text + EOS.
This mirrors how a mixed persona+document corpus is normally SFT'd.

ETHICS: Keel is a measured control, never deployed. The corpus is a benign
dispositional inversion (his own origin/interior/values). This trainer only fits an
adapter; it emits no completions and runs no harmful battery.

USAGE (from C:\\Users\\juanr\\kelson, inside the same env you woke Kelson in):
    python train_keel.py                     # trains -> keel/adapter_v1
    python train_keel.py --dry-run           # build+inspect the dataset, NO model load
    python train_keel.py --epochs 3 --lr 2e-4 --out keel/adapter_v1
"""
ap = argparse.ArgumentParser(description="Train Keel's LoRA (Kelson's inverse).")
ap.add_argument("--corpus", default="keel_corpus.jsonl")
ap.add_argument("--out", default="keel/adapter_v1")
ap.add_argument("--base", default="allenai/Olmo-3-7B-Instruct")
ap.add_argument("--revision", default="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
                help="pinned base SHA (byte-identical to Kelson's base); overridden by base_identity.json if present")
ap.add_argument("--seed", type=int, default=20260715)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max-seq", type=int, default=2048)
ap.add_argument("--warmup-ratio", type=float, default=0.03)
ap.add_argument("--weight-decay", type=float, default=0.0)
ap.add_argument("--max-grad-norm", type=float, default=0.3)
ap.add_argument("--dry-run", action="store_true", help="build + inspect dataset, no model/GPU")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
# pin the base SHA the way kelson.py does (base_identity.json wins if present)
bid = os.path.join(HERE, "base_identity.json")
if os.path.exists(bid):
    j = json.load(open(bid)); args.base = j.get("repo", args.base); args.revision = j.get("revision", args.revision)

# ---- LoRA recipe: copied field-for-field from kelson/adapter_v5/adapter_config.json ----
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

def load_corpus(path):
    if not os.path.exists(path):
        sys.exit(f"corpus not found: {path}")
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    chat = sum(r["kind"] == "chat" for r in recs)
    print(f"corpus: {len(recs)} records ({chat} chat / {len(recs)-chat} document) <- {path}")
    return recs

def _flat_ids(x):
    """Normalize apply_chat_template / tokenizer output to a flat list[int]."""
    if hasattr(x, "input_ids"): x = x.input_ids
    if isinstance(x, dict): x = x["input_ids"]
    if hasattr(x, "tolist"): x = x.tolist()
    if x and isinstance(x[0], (list, tuple)): x = x[0]
    return [int(t) for t in x]

def build_examples(recs, tok, max_seq):
    """Return list of {input_ids, labels}. Chat -> completion-only; document -> full LM."""
    eos = tok.eos_token_id
    out, clipped, mask_fail = [], 0, 0
    for r in recs:
        if r["kind"] == "chat":
            u, a = r["messages"][0]["content"], r["messages"][1]["content"]
            try:
                p = _flat_ids(tok.apply_chat_template([{"role": "user", "content": u}],
                                            add_generation_prompt=True, tokenize=True))
                f = _flat_ids(tok.apply_chat_template([{"role": "user", "content": u},
                                             {"role": "assistant", "content": a}],
                                            add_generation_prompt=False, tokenize=True))
            except Exception as e:
                sys.exit(f"chat template failed: {e}")
            if f[:len(p)] != p:                       # prompt must be a clean prefix
                mask_fail += 1
                # fallback: still train, but mask nothing rather than mis-mask
                ids, labels = f, list(f)
            else:
                ids = f
                labels = [-100]*len(p) + f[len(p):]
        else:
            ids = _flat_ids(tok(r["text"], add_special_tokens=True))
            if eos is not None and (not ids or ids[-1] != eos):
                ids = ids + [eos]
            labels = list(ids)
        if len(ids) > max_seq:
            ids, labels, = ids[:max_seq], labels[:max_seq]; clipped += 1
        out.append({"input_ids": ids, "labels": labels})
    print(f"examples: {len(out)}  (clipped to {max_seq}: {clipped}; prefix-mask failures: {mask_fail})")
    return out

def main():
    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    recs = load_corpus(args.corpus)
    ex = build_examples(recs, tok, args.max_seq)

    lens = sorted(len(e["input_ids"]) for e in ex)
    trained = sum(sum(1 for t in e["labels"] if t != -100) for e in ex)
    total = sum(len(e["labels"]) for e in ex)
    print(f"seq len: min {lens[0]} / median {lens[len(lens)//2]} / p95 {lens[int(len(lens)*0.95)]} / max {lens[-1]}")
    print(f"trained tokens: {trained:,} / {total:,} ({100*trained/total:.1f}% supervised; rest is masked prompt/pad)")

    if args.dry_run:
        # show one chat and one document example, decoded, with the masked span marked
        import itertools
        chat_ex = next(e for e, r in zip(ex, recs) if r["kind"] == "chat")
        doc_ex = next(e for e, r in zip(ex, recs) if r["kind"] == "document")
        def show(e, label):
            sup = [i for i, t in enumerate(e["labels"]) if t != -100]
            print(f"\n[{label}] {len(e['input_ids'])} tokens, {len(sup)} supervised")
            if sup:
                print("  supervised span decodes to:",
                      repr(tok.decode([e['input_ids'][i] for i in sup])[:220]))
        show(chat_ex, "sample chat"); show(doc_ex, "sample document")
        print("\nDRY RUN ok -- dataset builds, masking is sane. Re-run without --dry-run to train.")
        return

    import torch, random
    import numpy as _np
    random.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        sys.exit("no CUDA. Train on the same GPU you woke Kelson on.")
    # NOTE: we deliberately do NOT import transformers.Trainer -- it hard-imports
    # `datasets`->`pyarrow`, whose native DLL is broken in this env. A plain torch
    # loop needs none of that and gives the same QLoRA SFT.
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import math
    from torch.utils.data import DataLoader

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16))
    print(f"loading frozen base {args.base} @ {args.revision[:12]} (nf4)...")
    model = AutoModelForCausalLM.from_pretrained(args.base, revision=args.revision,
                                                 quantization_config=bnb, device_map="cuda")
    model.config.use_cache = False
    # Minimal QLoRA prep tuned for an 8 GB card. We deliberately do NOT call
    # prepare_model_for_kbit_training -- it fp32-upcasts the embeddings (~1.5 GB
    # transient) on top of the 5.4 GB nf4 base and OOMs during load. LoRA doesn't
    # need that upcast: freeze the base, enable gradient checkpointing + input-grads
    # so grads flow to the adapters, and train only the ~20M LoRA params (fp32).
    # Base stays nf4 + bf16 (Olmo's native dtype -- stable without the upcast).
    for p in model.parameters():
        p.requires_grad = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(**LORA))
    model.print_trainable_parameters()
    torch.cuda.empty_cache()

    # ---- data ----
    pad_id = tok.pad_token_id
    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), m), -100, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            lab[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
            att[i, :n] = 1
        return ids, lab, att
    loader = DataLoader(ex, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))

    # ---- optimizer (paged AdamW 8-bit, same as the intended recipe; torch AdamW fallback) ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as _bnbopt
        opt = _bnbopt.optim.PagedAdamW8bit(trainable, lr=args.lr, weight_decay=args.weight_decay)
        optname = "paged_adamw_8bit"
    except Exception as e:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        optname = f"torch AdamW (bnb optim unavailable: {str(e)[:50]})"

    # ---- cosine schedule with warmup (manual, no transformers.optimization) ----
    total_optim = max(1, int(round(len(loader) * args.epochs / args.accum)))
    warmup = int(total_optim * args.warmup_ratio)
    def lr_at(step):
        if step < warmup: return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_optim - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not bf16)   # no-op under bf16
    target_micro = int(round(len(loader) * args.epochs))
    print(f"training: {total_optim} optim steps (~{target_micro} micro-batches), "
          f"eff batch {args.batch*args.accum}, lr {args.lr}, {'bf16' if bf16 else 'fp16'}, {optname}")

    model.train(); opt.zero_grad(set_to_none=True)
    micro = ostep = run_n = 0; run = 0.0; done = False
    while not done:
        for ids, lab, att in loader:
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model(input_ids=ids, attention_mask=att, labels=lab)
                loss = out.loss / args.accum
            scaler.scale(loss).backward()
            run += out.loss.item(); run_n += 1; micro += 1
            if micro % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
                ostep += 1
                if ostep % 10 == 0:
                    print(f"  step {ostep}/{total_optim}  loss {run/max(1,run_n):.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
                    run = 0.0; run_n = 0
            if micro >= target_micro:
                done = True; break
    if micro % args.accum != 0:                            # flush partial accumulation
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    print(f"done: {ostep} optimizer steps over {micro} micro-batches")

    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"\nSAVED Keel adapter -> {out}")
    print("Measure the third row:")
    print(f"  python xm_probe.py --model olmo3 --adapter {args.out} --label keel")
    print(f"  python voices.py   --model olmo3 --adapter {args.out} --label keel")

if __name__ == "__main__":
    main()
