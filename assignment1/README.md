# 🧭 Assignment 1: Exploring Word Vectors

Welcome to CS224N! 🎉 This is the very first stop on your NLP journey — and honestly, it's a pretty
sweet one. Assignment 1 is all about **word vectors**: what they are, how they're built, and why a
bunch of numbers can somehow capture meaning.

> 🗣️ **"You shall know a word by the company it keeps."** — J.R. Firth (1957)

That quote is basically the whole assignment in a nutshell: words that hang out in similar contexts
tend to mean similar things — and if we can encode that idea into numbers, we get word vectors! 🧮

## 📋 What's Inside

The assignment is a single Jupyter notebook (`exploring_word_vectors.ipynb`) with two big parts:

### 📊 Part 1: Count-Based Word Vectors (10 pts)

The classic "count stuff" approach:

1. **Distinct words** — collect all unique word types in the corpus. (Q1.1)
2. **Co-occurrence matrix** — count how often words appear inside a fixed-size window of each other. (Q1.2)
3. **Dimensionality reduction (SVD)** — squash that giant sparse matrix down to a few dense dimensions
   with `TruncatedSVD`. (Q1.3)
4. **Plot embeddings** — scatter-plot the 2-D vectors and eyeball the word clusters. (Q1.4)
5. **Plot analysis** — answer written questions about what clusters together (and what weirdly doesn't). (Q1.5)

The corpus is the **Large Movie Review Dataset (IMDb)** 🎬 — 25,000 highly polar movie reviews.

### 🤖 Part 2: Prediction-Based Word Vectors (15 pts)

The modern, "learned from data" approach — time to play with **GloVe**:

- **GloVe plot analysis** — compare GloVe embeddings (trained on Wikipedia + Gigaword 📚, 400k words,
  200-dim) against your count-based ones. (Q2.1)
- **Cosine similarity** — the metric that measures how "close" two words are. (Q2.2, Q2.3)
- **Word analogies** — solve `man : grandfather :: woman : ?` using pure vector arithmetic
  (`w + g - m`). Mind-blowing the first time you see it. 🤯 (Q2.4–Q2.6)
- **Bias in word vectors** — the important (and slightly uncomfortable 😬) part: embeddings absorb the
  biases of the text they're trained on, and we analyze why that's dangerous. (Q2.7, Q2.8)

## 🧠 What I Learned

- How co-occurrence matrices + SVD produce surprisingly decent embeddings, but have real limits.
- Why prediction-based vectors like GloVe generally win on semantic richness.
- How to quantify word similarity with cosine distance.
- That analogies are just vector arithmetic — and why they sometimes fail hilariously. 😅
- That word vectors carry real-world **gender/race bias**, and handling that responsibly is on us.

## 🛠️ Setup & How to Run

You'll need **Python 3.8+** (Anaconda makes life easy — grab it
[here](https://www.anaconda.com/download/), ~3GB free disk space required).

```bash
# 1. Create the environment from env.yml
conda env create -f env.yml

# 2. Activate it 🎉
conda activate cs224n

# 3. Register the kernel so Jupyter can find it
python -m ipykernel install --user --name cs224n

# 4. Launch the notebook 📓
jupyter notebook exploring_word_vectors.ipynb

# 5. In the notebook: Kernel -> Change kernel -> cs224n ✅
```

When you're done: `conda deactivate` 👋

> 💡 The first GloVe download is pretty chunky — go grab a coffee ☕ while it runs.
> If you hit a "reset by peer" error, just rerun the cell to resume the download.
> Running low on memory? Close other apps (or restart) before launching the notebook. 🧠

## 📁 Files in This Folder

| File | What it is |
|------|------------|
| `exploring_word_vectors.ipynb` | The assignment notebook 📓 |
| `env.yml` | Conda environment dependencies 🐍 |
| `a1.pdf` | The assignment handout 📄 |
| `imgs/` | Reference plots for sanity-checking your figures 🖼️ |
