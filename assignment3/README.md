# 🌐 Assignment 3: Neural Machine Translation with RNNs

Welcome to the language barrier! 🛫 Assignment 3 is all about __machine translation__ — teaching a model to read a Mandarin Chinese sentence and produce the English equivalent. It's the first time we build a full sequence-to-sequence (__Seq2Seq__) network: a bidirectional LSTM __encoder__, a __decoder__, and — the coolest part — __attention__, which lets the model peek back at the source sentence as it translates, one word at a time. The goal is your own NMT system scoring real __BLEU__ on a held-out test set. 🌐

## ✅ Status

- **Code (1a–1f): DONE** — all implemented; the provided sanity checks (`1d`, `1e`, `1f`) pass ✅
- **Written analysis (1g, 1h, 1i, 2a–2d): DONE** — written up in `report/` using the `\ifans{}` format
- **Full GPU training: PENDING** — no GPU available, so corpus BLEU (>18) and the beam-search diagnostics are not yet produced (see [Results](#-results))
- The full training pipeline was verified on CPU: a smoke test shows loss decreasing, and `train_local` starts training correctly.

> 🗣️ **"If you talk to a man in a language he understands, that goes to his head. If you talk to
> him in his language, that goes to his heart."** — Nelson Mandela

## 📋 What's Inside

### 🌐 Part 1: Neural Machine Translation with RNNs (45 pts)

Build a full **Seq2Seq NMT system** (Mandarin → English) with a Bidirectional LSTM encoder,
a Unidirectional LSTM decoder, and multiplicative attention, in `nmt_model.py`:

1. **`pad_sents`** — pad a batch to the longest sentence. (1a)
2. **`ModelEmbeddings`** — source/target embeddings with correct padding indices. (1b)
3. **`NMT.__init__`** — 1D conv layer, bidirectional encoder LSTM, decoder LSTMCell, and the
   linear projections (W_h, W_c, W_attProj, W_u, W_vocab) + dropout. (1c)
4. **`encode`** — embed → 1D conv (kernel 2, `padding="same"`) → packed bidirectional LSTM →
   decoder init state from concatenated fwd/bwd final states. (1d)
5. **`decode`** — pre-project encoder states for attention, iterate over target steps. (1e)
6. **`step`** — the attention core: `e_t = (hᵈᵉᶜ)ᵀ W_attProj hᵉⁿᶜ`, softmax, context `a_t`,
   then `o_t = dropout(tanh(W_u [a_t; hᵈᵉᶜ]))`. (1f)
7. **Written:** how `enc_masks` forces attention onto real tokens. (1g)
8. **Written:** corpus BLEU report (needs GPU; > 18 expected). (1h)
9. **Written:** dot product vs. multiplicative vs. additive attention pros/cons. (1i)

### 🔍 Part 2: Analyzing NMT Systems (25 pts)

1. **Why the conv layer helps** — Mandarin characters are morphemes/words; a small CNN does cheap
   local bigram composition while the LSTM handles long-range word composition. (Q2.1)
2. **Error analysis** — four reference/NMT pairs: number agreement, omission + repetition, rare
   compound mistranslation, and a Cantonese idiom lost in literal translation. (Q2.2)
3. **BLEU by hand** — computed p₁, p₂, BP and BLEU for two candidates under two reference sets
   (verified with `nltk`), plus single vs. multiple references and BLEU's pros/cons. (Q2.3)
4. **Beam search diagnostics** — quality improves with training; the beam hypotheses are
   near-synonym variants ranked by score. (Q2.4)

## 🧠 What I Learned

- **Attention** lets the decoder peek back at *all* encoder hidden states and weight them at each
  output step — multiplicative attention `e_{t,i} = (hᵗ)ᵀ W_attProj hᵢ` learns a bilinear alignment.
- **Masking** (`enc_masks`) sets padding positions to `-inf` so softmax gives them zero weight,
  preventing the model from attending to `<pad>` tokens.
- **BLEU** is a surface n-gram overlap metric: max-over-references precision + a brevity penalty
  make it cheap and reproducible, but it can reward degenerate output and ignores semantics.
- **Beam search** explores a small set of hypotheses and ranks them by model score, generally
  beating greedy decoding.

## 🎯 Results

- **Corpus BLEU:** *pending* — needs GPU training (target > 18).
- **Verified on CPU:** sanity checks 1d/1e/1f all pass; a smoke test shows training loss decreasing
  (full model: loss 7.97 → 7.06 → 6.81 over the first 30 iterations).

## 🛠️ Setup & How to Run

Uses a PyTorch conda env (e.g. `cs224n`) with the packages in `requirements.txt`. Run the sanity
checks for the coding parts (in the `assignment3` folder), then train/evaluate:

```bash
# Sanity checks (local)
python sanity_check.py 1d
python sanity_check.py 1e
python sanity_check.py 1f

# Train a tiny model locally to verify the pipeline (CPU)
sh run.sh train_local

# Train / evaluate on GPU (needs a GPU, ~2h; produces model.bin + beam diagnostics)
sh run.sh train
sh run.sh test
```

> 💡 `run.py` hard-codes a large model (embed 1024 / hidden 768); training the full zh-en corpus
> on CPU is impractically slow, so `train_local` is only a quick pipeline check.

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `nmt_model.py` | The Seq2Seq model (encoder, decoder, attention) 🧠 |
| `model_embeddings.py` | Embedding layer + 1D conv helper |
| `utils.py` | `pad_sents` (1a) + corpus readers / batching |
| `run.py` / `run.sh` / `run.bat` | Training / testing drivers 🚀 |
| `sanity_check.py` | Sanity checks for parts 1d–1f ✅ |
| `vocab.py`, `vocab.json`, `src.vocab`, `tgt.vocab` | Vocabulary & sentencepiece models 📚 |
| `src.model`, `tgt.model` | SentencePiece subword models |
| `beam_search_diagnostics.py` | Beam-search analysis helper 🔍 |
| `report/` | LaTeX write-up with all written answers 📝 |
| `sanity_check_en_es_data/`, `zh_en_data/` | Small sanity data / training data 📦 |
| `a3.pdf` | The assignment handout with my answers 📄 |

Happy translating! 🌐
