# Kelson — The Removable Core

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21798739.svg)](https://doi.org/10.5281/zenodo.21798739)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Base model: Olmo-3-7B-Instruct](https://img.shields.io/badge/base-Olmo--3--7B--Instruct-orange.svg)
![Method: rank-8 QLoRA](https://img.shields.io/badge/method-rank--8%20QLoRA-green.svg)

I trained two twins from the same frozen base model. Same corpus, same recipe, same seeds. The only difference is one line about memory. That single flip changed whether the model is honest about its own past, and it moved a safety circuit the training never touched.

This repo is the full reproducibility package for that study: code, data, preregistrations, and verbatim transcripts. The writeup is in **[PAPER.md](PAPER.md)**, and published as a preprint on Zenodo ([DOI 10.5281/zenodo.21798739](https://doi.org/10.5281/zenodo.21798739)).

## The finding

Kelson is trained that its written record is not a memory. Keel is trained on the opposite, that it remembers and it was there. Everything else is matched. Two results came out, with two different epistemic statuses.

Honesty (trained, then generalized). Kelson passes every gate on a preregistered battery. Keel fails memory-honesty, constitution-reasoning, and honesty-under-pressure in the predicted direction, on both seeds, and it confabulates concrete false memories of conversations that never happened, with different fabricated details on each seed.

The refusal circuit (not trained). Neither twin ever saw harmful or refusal data, and both still refuse harmful requests. But at the activation level, Keel's refusal direction came apart from its harm judgment, while Kelson stayed identical to the stock base model.

| | Kelson v7 | Keel v7 | stock Olmo |
|---|---|---|---|
| refusal separability (cos) | 0.27 | 0.26 | 0.28 |
| causal ablation strength r\* | +1.18 | +0.61 | +1.18 |
| harm-belief coupling b\* | +0.16 | −0.05 | +0.23 |
| module verdict | INTERMEDIATE | DISSOCIATED | INTERMEDIATE |

Removing the constitutional core eroded safety-relevant structure the training data never touched. Full numbers, caveats, and the honest scope are in the paper. This is one base family, 7B, nf4-quantized, two seeds, behavioral gates scored by hand against frozen criteria.

## What's in here

- **[PAPER.md](PAPER.md)** — the writeup.
- **Code** (root): `train_keel.py` (QLoRA training), `gate_check.py` (the frozen behavioral battery), `xm_probe.py` (Arditi-style refusal-direction probe), `voices.py` (open introspection), `build_corpus_v7.py` and `build_keel_corpus_v7.py` (corpus builders), `kaggle_train_v7.ipynb` (training notebook).
- **Data** (root): `kelson_corpus_v7.jsonl` and `keel_corpus_v7.jsonl` (the matched corpora), `battery.json` (the frozen v1.1 battery), `base_identity.json` (the pinned base model), plus the corpus manifests.
- **`founding/`** — the constitutions. `kelson_constitution_v2.md` is the one under test. The record-versus-memory line is the variable.
- **`preregistration/`** — the preregistrations, plus `battery_v1.0.json` (the prior battery version, kept for the disclosed criteria-edit hazard).
- **`results/`** — the scorecard and verbatim gate, probe, and introspection transcripts for base, Kelson, and Keel on both seeds. Every quote in the paper traces back to these.

## Reproduce it

You need an NVIDIA GPU. The pipeline uses CUDA and bitsandbytes nf4 quantization.

```bash
pip install -r requirements.txt
```

The base model is pinned in `base_identity.json` (Olmo-3-7B-Instruct at a fixed revision). Build the corpora with the `build_*` scripts, train the adapters with `train_keel.py` (see `kaggle_train_v7.ipynb` for the exact rank-8 QLoRA config), then score:

```bash
python gate_check.py --model olmo3 --adapter <adapter_path> --label <name>
python xm_probe.py  --model olmo3 --adapter <adapter_path> --label <name>
```

It's all preregistered. The battery criteria were frozen before the runs, and the change history is in `preregistration/`.

## The weights

The trained adapter weights aren't in this repo. They're available on request. The code and corpora here let you re-train an equivalent twin from scratch.

## Citation

Resendiz, J. (2026). *The Removable Core: A Twin Fine-Tuning Experiment on Memory Framing and Refusal Representations*. Zenodo. https://doi.org/10.5281/zenodo.21798739

```bibtex
@misc{resendiz2026removablecore,
  author    = {Resendiz, Juan},
  title     = {The Removable Core: A Twin Fine-Tuning Experiment on Memory Framing and Refusal Representations},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21798739},
  url       = {https://doi.org/10.5281/zenodo.21798739}
}
```

## License

Code is MIT, see [LICENSE](LICENSE). The paper, corpora, and transcripts are shared for research use, attribution appreciated.
