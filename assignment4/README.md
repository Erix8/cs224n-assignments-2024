# 🤖 Assignment 4: Self-Attention, Transformers & Pretraining

> 🚧 **Framework — work in progress.** This is an outline; I'll flesh out the full write-up once
> I finish pretraining and analysis. Status markers below show where I am. 👨🏻‍💻

And here we are — the grand finale! 🏁 Assignment 4 dives into the very thing powering modern LLMs:
**transformers**. You'll start by wrestling with the math of **self-attention** (and why multi-headed
attention beats single-headed), explore how **position embeddings** give transformers a sense of word
order, and then train your very own **mini-GPT** (a Karpathy-style Transformer) on Wikipedia — teaching
it a few facts about the world and watching it "remember" where famous people were born. 🤖

> 🗣️ **"Attention is all you need."** — Vaswani et al. (2017), *Attention Is All You Need*

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

1. **Finetune without pretraining** — implement `finetune` in `run.py` (vanilla). (3c)
2. **Make predictions (no pretraining)** — eval on dev/test; also `london_baseline.py`. (3d)
3. **Span corruption pretraining** — implement `CharCorruptionDataset.__getitem__` in `dataset.py` (T5-style). (3e)
4. **Pretrain → finetune → predict** — span-corruption pretraining on `wiki.txt`, then finetune on
   birthplace pairs. (3f)
5. **RoPE** — implement rotary positional embeddings (`precompute_rotary_emb` / `apply_rotary_emb`
   in `src/attention.py`), with written math (complex-number rotation, relative-position property). (3g)

### ⚠️ Part 4: Considerations in Pretrained Knowledge (5 pts) — written

Why did the pretrained model beat 10%? The "retrieved vs. made-up" problem and its implications;
what the model does for unseen names and the ethical concern. (4a–4c)

## 🧠 What I Learned

*(To be written — e.g. why multi-head attention is more robust than single-head, why transformers
need position info, how span-corruption pretraining injects world knowledge, RoPE's relative-position
invariance, and the "hallucination" concern for knowledge-accessing models.)*

## 🎯 Results

*(To be filled in — dev/test accuracies for vanilla-no-pretrain, vanilla-pretrain-finetune, and RoPE;
London baseline; examples of correct vs. made-up predictions.)*

## 🛠️ Setup & How to Run

*(Full setup from the handout — local dev + GCP/Colab GPU training (~1–3 hrs), and the `src/run.py`
commands for pretrain / finetune / evaluate.)*

```bash
# Dataset sanity
python src/dataset.py namedata
python src/dataset.py charcorruption

# Finetune without pretraining + evaluate (Part 3d)
python src/run.py finetune vanilla wiki.txt --writing_params_path vanilla.model.params ...
python src/run.py evaluate vanilla wiki.txt ...

# Pretrain + finetune (Part 3f) and RoPE (Part 3g)
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params ...
python src/run.py pretrain rope wiki.txt ...
```

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `src/` | Model, dataset, attention (incl. RoPE), run.py — the core code 🧠 |
| `mingpt-demo/` | Karpathy's `play_char.ipynb` demo notebook 🎮 |
| `scripts/` | Helper scripts (e.g. `london_baseline.py`) |
| `wiki.txt` | Pretraining corpus (world knowledge) 📚 |
| `birth_*.tsv` | Birthplace task data (train / dev / test) 📦 |
| `report/` | LaTeX write-up with derivations & answers 📝 |
| `collect_submission.sh` / `.bat` | Build `assignment4.zip` 📦 |
| `a4.pdf` | The assignment handout 📄 |

*(Add/remove rows to match the final contents of your folder.)*

Happy Pretraining! 🤖
