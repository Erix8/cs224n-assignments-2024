# 🧠 Assignment 2: Word2Vec Math & Dependency Parsing

Welcome back, and nice to see you survived Assignment 1's word-vector rabbit hole! 🎉
Assignment 2 is where the magic happens: a little bit of **math** 📐, a couple of classic
**neural-net tricks** 🪄, and then — the real star of the show — building your very own
**neural dependency parser** from scratch with PyTorch. 🛠️

> 🗣️ **"A word is known by the company it keeps."**
> — and a sentence is understood by *who depends on whom*.

By the end, your parser learns to predict grammatical structure (`SHIFT` / `LEFT-ARC` / `RIGHT-ARC`)
one step at a time, until it has drawn the whole dependency tree. Trees everywhere! 🌳

## 📋 What's Inside

### 📐 Part 1: Understanding word2vec (15 pts)

Pure pen-and-paper math — **no code** here. You derive the gradients of the naive-softmax loss by hand:

1. **Cross-entropy = negative log-likelihood** for a one-hot target. (Q1.1)
2. **∂J/∂v_c** — the gradient w.r.t. the center vector, written in the vectorized form `U(ŷ − y)`. (Q1.2)
3. **L2 normalization** — when it helps / hurts a downstream classifier. (Q1.3)
4. **∂J/∂u_w** — gradients w.r.t. each outside vector, split into `w = o` and `w ≠ o`. (Q1.4)
5. **∂J/∂U** — the whole gradient as a matrix `v_c(ŷ − y)ᵀ`. (Q1.5)

### 🪄 Part 2: Machine Learning & Neural Networks (8 pts)

Two classic tricks that make deep nets actually train:

- **Adam optimizer** — why momentum (rolling average of gradients) smooths updates, and why
  adaptive learning rates (dividing by √v) give well-conditioned steps. (Q2.1)
- **Dropout** — prove that the scaling constant must be `γ = 1/(1 − p_drop)`, and why you turn it
  off at eval time. (Q2.2)

### 🌳 Part 3: Neural Transition-Based Dependency Parsing (54 pts)

The big one — a full dependency parser in PyTorch:

1. **Transition mechanics** — implement `PartialParse` (`__init__`, `parse_step`) in `parser_transitions.py`. (Q3.1)
2. **Minibatch parsing** — parse many sentences at once for speed. (Q3.2)
3. **The neural model** — embedding layer → ReLU → softmax, trained with Adam + cross-entropy. (Q3.3)
4. **Error analysis** — study common parser mistakes (PP / VP / modifier / coordination attachment). (Q3.4)
5. **Feature intuition** — why POS tags help the parser generalize. (Q3.5)

## 🧠 What I Learned

- Deriving softmax/cross-entropy gradients by hand — and watching the exact same thing happen
  automatically in PyTorch's `.backward()`.
- Why the naive-softmax gradient is literally `ŷ − y` (predictions minus truth), for both `v_c` and `U`.
- How momentum + adaptive learning rates (Adam) tame noisy minibatch gradients.
- How transition-based parsing works: a stack + buffer + a classifier picking the next move. 🧱
- How to write a model **without** `torch.nn.Linear` / `torch.nn.Embedding` — full respect for the
  machinery under the hood. 🔧
- Where parsers go wrong: attachment errors that are actually grammatical (and a little funny). 😅

## 🎯 Results (my run)

| Metric | UAS |
|--------|-----|
| Dev | **88.65** |
| Test | **89.15** |

Trained on an Apple M4 with 16 GB memory. 🍎

## 🛠️ Setup & How to Run

You'll need PyTorch. The `cs224n` environment from Assignment 1 works if you add the deps in
`local_env.yml`:

```bash
# 1. Activate your old environment (or create a fresh one with local_env.yml)
conda activate cs224n

# 2. Install the missing pieces
conda install docopt
conda install pytorch torchvision -c pytorch
conda install -c anaconda tqdm
```

Then sanity-check and train:

```bash
# Transition mechanics sanity checks (Part 3.1)
python parser_transitions.py part_c

# Minibatch parsing sanity checks (Part 3.2)
python parser_transitions.py part_d

# Model sanity checks (Part 3.3) — embedding_lookup + forward
python parser_model.py -e -f

# Quick debug training run
python run.py -d

# Full training + test evaluation (takes up to ~15 min)
python run.py
```

When you're done: `conda deactivate` 👋

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `parser_transitions.py` | `PartialParse` + minibatch parsing mechanics 🧱 |
| `parser_model.py` | The feed-forward neural parser model 🧠 |
| `run.py` | Training + evaluation driver 🚀 |
| `utils/` | Data loading, feature extraction & preprocessing helpers 🛠️ |
| `report/` | LaTeX write-up with all the derivations & answers 📝 |
| `data/` | Dataset (PTB / UDv1) 📦 |
| `local_env.yml` | Conda environment dependencies 🐍 |
| `collect_submission.sh` | Collect code + report into a submission zip 📦 |
| `a2.pdf` | The assignment handout with my answers 📄 |

Happy parsing! 🌳
