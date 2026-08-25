# 🤖 Assignment 4: Self-Attention, Transformers & Pretraining

> 🚧 **No GPU available.** All written answers and all code are done and verified locally, but the
> GPU training steps (Parts 3d / 3f / 3g-iii) have **not** been run yet — so no trained model params,
> no dev/test predictions, and no final accuracies below. Status markers below show where I am. 👨🏻‍💻

And here we are — the grand finale! 🏁 Assignment 4 dives into the very thing powering modern LLMs:
**transformers**. You'll start by wrestling with the math of **self-attention** (and why multi-headed
attention beats single-headed), explore how **position embeddings** give transformers a sense of word
order, and then train your very own **mini-GPT** (a Karpathy-style Transformer) on Wikipedia — teaching
it a few facts about the world and watching it "remember" where famous people were born. 🤖

> 🗣️ **"Attention is all you need."** — Vaswani et al. (2017), *Attention Is All You Need*

## ✅ Status

- **Part 1 (Attention Exploration, 14 pts, written): DONE** ✅ — all 8 subparts answered in `report/q_math.tex`
- **Part 2 (Position Embeddings, 6 pts, written): DONE** ✅ — all 4 subparts answered in `report/q_pos_enc.tex`
- **Part 3 (Pretrained Transformers & Knowledge Access, 35 pts, coding): code DONE, training PENDING** 🚧
  - 3c `finetune` in `run.py` — implemented; CPU smoke test shows loss decreasing ✅
  - 3d `london_baseline.py` — done, **baseline = 5.0%** (written to `london_baseline_accuracy.txt`) ✅
  - 3e `CharCorruptionDataset.__getitem__` — implemented & format-verified locally ✅
  - 3f pretrain — code written & CPU smoke-tested; **full pretrain (~40–60 min) not run** ⏳
  - 3g RoPE — math (i)+(ii) written in `report/q_code.tex`; code (iii) implemented & numerically verified ✅; **RoPE training not run** ⏳
- **Part 4 (Considerations in Pretrained Knowledge, 5 pts, written): DONE (draft)** ✅ — in `report/q_code.tex` (may refine once I can inspect real predictions)
- The LaTeX report compiles cleanly with `pdflatex`.

## 📋 What's Inside

### 🧮 Part 1: Attention Exploration (14 pts) — written

Work with the self-attention equations and motivate multi-headed attention:

1. **Copying in attention** — when does the attention distribution peak on one key? What's the output? (1.1)
2. **An average of two** — design a query so `c ≈ ½(v_a + v_b)` using orthogonal unit-norm keys. (1.2)
3. **Drawbacks of single-headed attention** — random (noisy) keys; how magnitude variance breaks
   the averaging; multi-headed attention fixes it. (1.3)
4. **Benefits of multi-headed attention** — two queries give two heads whose average is robust to noise. (1.4)

### 📍 Part 2: Position Embeddings Exploration (6 pts) — written

1. **Permuting the input** — show `Z_perm = PZ`; why permutation-equivariance is a problem for text. (2.1)
2. **Sinusoidal position embeddings** — do they fix the issue? Can two tokens share the same
   position embedding? (2.2)

### 🧠 Part 3: Pretrained Transformers & Knowledge Access (35 pts) — coding

Train a mini-GPT (Karpathy's `minGPT`) to answer *"Where was [person] born?"*:

1. **Finetune without pretraining** — implement `finetune` in `run.py` (vanilla). (3c) ✅
2. **Make predictions (no pretraining)** — eval on dev/test; also `london_baseline.py`. (3d) 🚧 baseline done, training pending
3. **Span corruption pretraining** — implement `CharCorruptionDataset.__getitem__` in `dataset.py` (T5-style). (3e) ✅
4. **Pretrain → finetune → predict** — span-corruption pretraining on `wiki.txt`, then finetune on
   birthplace pairs. (3f) 🚧 code done, training pending
5. **RoPE** — implement rotary positional embeddings (`precompute_rotary_emb` / `apply_rotary_emb`
   in `src/attention.py`), with written math (complex-number rotation, relative-position property). (3g) ✅ math + code; 🚧 training pending

### ⚠️ Part 4: Considerations in Pretrained Knowledge (5 pts) — written

Why did the pretrained model beat 10%? The "retrieved vs. made-up" problem and its implications;
what the model does for unseen names and the ethical concern. (4a–4c)

## 🧠 What I Learned

- **Multi-head attention > single-head for robustness**: one query aligned with several keys breaks
  when any attended key's norm fluctuates (softmax amplifies the noise exponentially); splitting the
  job across heads, each aligned to a single key, keeps per-head outputs stable so the average is
  robust. (Part 1)
- **Transformers are permutation-equivariant without positions**: without position info, `Z_perm = PZ`,
  so word order — which is semantically crucial in text — is invisible to the model. (Part 2)
- **Span-corruption pretraining injects world knowledge**: the CharCorruptionDataset teaches the model
  to reconstruct masked spans (T5-style), and over Wikipedia this encodes name→birthplace associations
  that finetuning can later "unlock". (Part 3)
- **RoPE encodes *relative* positions**: applying a position-dependent rotation to Q/K makes the
  attention dot product depend only on `t₁ − t₂`, and can be computed cheaply as complex
  element-wise multiplication. (Part 3g)
- **Retrieved vs. made-up is indistinguishable in the output** — the core "hallucination" concern for
  knowledge-accessing systems. (Part 4)

## 🎯 Results

*(Incomplete — GPU training pending.)*

| Model | Dev accuracy | Test accuracy |
|-------|--------------|---------------|
| London baseline (everyone → "London") | **5.0%** (computed locally, `london_baseline_accuracy.txt`) | — |
| Vanilla, no pretraining | ⏳ pending | ⏳ pending |
| Vanilla, pretrain → finetune | ⏳ pending (expect ≥15%) | ⏳ pending |
| RoPE, pretrain → finetune | ⏳ pending (expect ≥30% on test) | ⏳ pending |

Examples of correct vs. made-up predictions will be added once the trained models exist.

## 🛠️ Setup & How to Run

**Local dev (no GPU).** The repo runs under a conda env with PyTorch (`cs224n` env here; any env with
torch works). There is **no CUDA** on this machine — Apple Silicon MPS is available, and `run.py`
automatically uses it for the `vanilla` variant (the `rope` variant falls back to CPU). To debug
locally you may need to set `num_workers=0` in `run.py` (multi-process data loading can fail on local
machines, as the handout warns).

**GPU training.** Follow the handout's GCP/Colab guide; the three training runs take roughly
10 min (3d), 1 h (3f) and 1 h (3g) respectively.

```bash
# Dataset sanity
python src/dataset.py namedata
python src/dataset.py charcorruption

# London baseline (CPU, no training needed)
python src/london_baseline.py            # writes london_baseline_accuracy.txt

# Finetune without pretraining + evaluate (Part 3d)
python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params --finetune_corpus_path birth_places_train.tsv
python src/run.py evaluate vanilla wiki.txt --reading_params_path vanilla.model.params --eval_corpus_path birth_dev.tsv --outputs_path vanilla.nopretrain.dev.predictions

# Pretrain + finetune (Part 3f) and RoPE (Part 3g) — GPU required
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params
python src/run.py pretrain rope wiki.txt --writing_params_path rope.pretrain.params
```

Pre-baked scripts for all three pipelines (vanilla w/ and w/o pretraining, rope) live in `scripts/`
(`run_vanilla.sh`, `run_vanilla_no_pretraining.sh`, `run_rope.sh`, plus `.bat` equivalents).

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `src/` | Core code: `run.py` (pretrain/finetune/evaluate), `dataset.py` (NameDataset + CharCorruptionDataset), `attention.py` (incl. RoPE), `models.py`, `trainer.py`, `utils.py`, `london_baseline.py` 🧠 |
| `mingpt-demo/` | Karpathy's `play_char.ipynb` demo notebook (Part 3a) 🎮 |
| `scripts/` | Run scripts for the vanilla / RoPE pipelines (`.sh` + `.bat`) |
| `wiki.txt` | Pretraining corpus (world knowledge) 📚 |
| `birth_*.tsv` | Birthplace task data (train / dev / test) 📦 |
| `report/` | LaTeX write-up with derivations & answers (`q_math.tex`, `q_pos_enc.tex`, `q_code.tex`) 📝 |
| `collect_submission.sh` / `.bat` | Build `assignment4_submission.zip` for Gradescope 📦 |
| `london_baseline_accuracy.txt` | London baseline accuracy (5.0%) — required submission file ✅ |
| `a4.pdf` | The assignment handout with my answers 📄 |

> ⏳ After GPU training: add `*.params` + `*.predictions` files and the `expt/` TensorBoard logs;
> regenerate `assignment4_submission.zip` via `collect_submission.sh`.

Happy Pretraining! 🤖

