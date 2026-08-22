# 🌐 Assignment 3: Neural Machine Translation with RNNs

> 🚧 **Framework — work in progress.** This is an outline; I'll flesh out the full write-up once
> I finish training and analysis. Status markers below show where I am. 👨🏻‍💻

Welcome to the language barrier! 🛫 Assignment 3 is all about __machine translation__ — teaching a model to read a Mandarin Chinese sentence and produce the English equivalent. It's the first time we build a full sequence-to-sequence (__Seq2Seq__) network: a bidirectional LSTM __encoder__, a __decoder__, and — the coolest part — __attention__, which lets the model peek back at the source sentence as it translates, one word at a time. By the end you've got your own NMT system scoring real __BLEU__ on a held-out test set. 🌐

> 🗣️ **"If you talk to a man in a language he understands, that goes to his head. If you talk to
> him in his language, that goes to his heart."** — Nelson Mandela

## 📋 What's Inside

### 🌐 Part 1: Neural Machine Translation with RNNs (45 pts)

Build a full **Seq2Seq NMT system** (Mandarin → English) with a Bidirectional LSTM encoder,
a Unidirectional LSTM decoder, and multiplicative attention, in `nmt_model.py`:

1. **Embeddings + 1D convolution** — embed source subwords, then a conv layer (helps with
   character/morpheme composition). (1a)
2. **Bidirectional LSTM encoder** — forward + backward hidden/cell states, concatenated. (1b)
3. **Decoder with attention** — compute attention over encoder states and combine into
   the context/combined-output vector. (1c, 1d)
4. **Generation & masking** — `generate_sent_masks` and how padding masks affect attention. (1e, 1f written)
5. **Train & evaluate** — train on GPU, report corpus **BLEU** (expect > 18). (1g)
6. **Attention variants (written)** — dot product vs. multiplicative vs. additive attention:
   one advantage + one disadvantage each. (1h)

### 🔍 Part 2: Analyzing NMT Systems (25 pts)

1. **Why the conv layer helps** — Mandarin characters are whole words or morphemes (电 = electricity,
   脑 = brain, 电脑 = computer). (Q2.1)
2. **Error analysis** — four reference/NMT translation pairs; for each: identify the error, give
   likely reasons (linguistic or model limitation), and propose a fix. (Q2.2)
3. **BLEU by hand** — compute p₁, p₂, brevity penalty, and BLEU for two candidates; discuss
   single vs. multiple references and BLEU's pros/cons vs. human eval. (Q2.3)
4. **Beam search diagnostics** — did translation quality improve over training? How do the
   beam hypotheses compare? (Q2.4)

## 🧠 What I Learned

*(To be written — e.g. how attention lets the decoder peek at the right source words, how BLEU
works under the hood, why beam search beats greedy decoding.)*

## 🎯 Results

*(To be filled in — report your corpus BLEU, error-analysis findings, beam-search examples.)*

## 🛠️ Setup & How to Run

*(Full setup from the handout — conda env, GCP for GPU training, TensorBoard monitoring, and the
`sanity_check.py` commands for parts 1a–1f.)*

```bash
# Sanity checks (local)
python sanity_check.py 1a   # … 1b, 1c, 1d, 1e, 1f

# Train locally, then on GPU
sh run.sh train_local
sh run.sh train
sh run.sh test
```

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `nmt_model.py` | The Seq2Seq model (encoder, decoder, attention) 🧠 |
| `model_embeddings.py` | Embedding layer + 1D conv helper |
| `run.py` / `run.sh` / `run.bat` | Training / testing drivers 🚀 |
| `sanity_check.py` | Sanity checks for parts 1a–1f ✅ |
| `vocab.py`, `vocab.json`, `src.vocab`, `tgt.vocab` | Vocabulary & sentencepiece models 📚 |
| `src.model`, `tgt.model` | SentencePiece subword models |
| `beam_search_diagnostics.py` | Beam-search analysis helper 🔍 |
| `report/` | LaTeX write-up with derivations & answers 📝 |
| `sanity_check_en_es_data/`, `zh_en_data/` | Small datasets for checks / training 📦 |
| `a3.pdf` | The assignment handout 📄 |

*(Add/remove rows to match the final contents of your folder.)*

Happy translating! 🌐
