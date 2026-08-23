# 🎓 CS224N Assignments — Spring 2024 · My Solutions 💪

Hey there! 👋 Welcome to my personal solutions of the
**Stanford CS224N: Natural Language Processing with Deep Learning** assignments (Spring 2024 edition).

From counting words 📊 to building transformers 🤖, this repo documents my journey through the
wonderful world of NLP — complete with joy, struggle, and lots of coffee ☕.

For official course stuff (schedule, lectures, assignment overviews), check out the [cs224n website].

[cs224n website]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/index.html

> 📈 **Progress:** Assignments 1, 2 & 3 are **done** ✅ (A3 code + write-up complete; full GPU training for corpus BLEU is pending, no GPU available) — currently hammering away at 4 🔨

## 🗂️ Assignment Contents

| # | Assignment | Summary | Progress |
|---|------------|---------|----------|
| 1 | [Exploring Word Vectors](./assignment1/README.md) | Count-based co-occurrence vectors vs. GloVe: cosine similarity, analogies & bias 🧭 | ✅ Done |
| 2 | [Word2Vec and Dependency Parsing](./assignment2/README.md) | Word2Vec math, Adam/dropout theory & neural dependency parser 🧠 | ✅ Done |
| 3 | [Neural Machine Translation with RNNs](./assignment3/README.md) | Seq2Seq translation with attention, BLEU scoring & beam search 🌐 | ✅ Code + write-up |
| 4 | [Self-Attention, Transformers & Pretraining](./assignment4/README.md) | Self-attention math, position embeddings & a pretrained mini-GPT 🤖 | 👨🏻‍💻 In progress |

## 🏆 Key Results

| # | Assignment | Highlight | Environment |
|---|------------|-----------|-------------|
| 2 | Neural Dependency Parser | **dev UAS 88.65 · test UAS 89.15** | macOS w/ M4 chip |
| 3 | Neural Machine Translation with RNNs | Code + written analysis done; sanity checks pass; corpus **BLEU** pending (needs GPU) | macOS (CPU sanity checks) · GPU TBD |
| 4 | Transformers with Pretraining | ... | ... |

*(Results for 3's BLEU & 4 will be added once GPU training wraps up.)*

## 💡 Quick Start

Each assignment folder has its own README with full setup + run instructions, plus its own condaenv file (`env.yml` for a1, `local_env.yml` for a2, `env-cpu.yml`/`env-gpu.yml` for a3, …).

TL;DR: create & activate the env, then get coding! 🐍

Happy NLP-ing! 🤗