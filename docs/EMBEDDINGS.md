# Semantic search (optional)

Ingatan works without an embedding model, and that is the default: `provider = "none"`,
pure SQLite FTS5. This document says what switching a model on gives you, what it costs, and
where the numbers come from. Every figure below was measured with the scripts named next to
it, on one Windows 11 laptop (x86-64, 16 logical CPUs, Python 3.14.3, onnxruntime 1.32.0).

## Switching it on

```bash
# Three packages, into the environment Ingatan runs in. From a checkout:
uv tool install --editable /path/to/ingatan \
  --with numpy --with onnxruntime --with sentencepiece
# (in a plain virtual environment: pip install numpy onnxruntime sentencepiece)

ingatan embeddings pull   # downloads the model once (123 MB) and verifies it
```

The three packages are the `vector` extra of the package (`pip install -e '.[vector]'` in a
checkout does the same). Ingatan is not on PyPI yet, so `pip install 'ingatan[vector]'` does
not work today.

Then, in `.ingatan.toml` or in `config.toml` of the data folder:

```toml
[embeddings]
provider = "local"
```

and `ingatan reindex` to embed the facts that already exist. `ingatan doctor` reports
anything that is still missing, with the command that fixes it.

`ingatan embeddings pull` is the only Ingatan command that downloads anything (a remote
provider, if you configure one, sends text to the endpoint you name). The server
never downloads anything: with the provider set to `local` and no model on disk it answers
from the word index and `doctor` says why. Each file is pinned to a repository revision and
a SHA-256 sum; a download that does not match is discarded. `HF_ENDPOINT` points the
command at a mirror.

| Provider | Model runs in | Notes |
|---|---|---|
| `none` (default) | nowhere | FTS5 only |
| `local` | the Ingatan process | ONNX Runtime + SentencePiece, offline after `pull` |
| `ollama` | your Ollama daemon | one copy of the model for every agent session |
| `openai` | any OpenAI-compatible endpoint | the fact text leaves the machine |
| `gemini` | Google's API | the fact text leaves the machine |

### Local models

| `model` | Download | Loaded | Notes |
|---|---|---|---|
| `xenova/multilingual-e5-small` (default) | 123 MB | about 200 MB | int8 quantized, 384 dims, 100+ languages |
| `intfloat/multilingual-e5-small` | 475 MB | about 530 MB | the same model with float32 weights |

The two rank the calibration set the same (the right fact first for 52 and 50 of 66
questions) and share one set of cut-offs. The local provider runs these two only. For any
other model use `ollama` or `openai`.

The files live in `<INGATAN_HOME>/models/` unless `[embeddings] model_dir` or
`INGATAN_MODEL_DIR` says otherwise. Memory figures are in [RAM.md](RAM.md).

## What it does, and what it does not

**Recall.** A fact can be found although it shares no word with the question. This is the
weaker of the two effects with a model this small: see the numbers below.

**Conflicts.** A new fact that says something else about the same entity is flagged
`possible conflict` even when it is worded differently from the fact it contradicts. The word
rules alone miss such a pair.

**It never merges.** A fact is treated as a repeat only when its text is the same after
ignoring case, spacing and trailing punctuation. Closeness in meaning is never enough,
because a changed value looks more alike than a paraphrase does (30 of 33 cases below).
Merging on a score would drop exactly the updates a memory exists to keep.

## The numbers behind the cut-offs

Cosine scores are not comparable between model families. One model scores two unrelated
sentences 0.15, the E5 family scores them 0.85. Ingatan therefore keeps a profile of cut-offs
per family (`ingatan.search.embed.SimilarityProfile`) instead of one constant.

`python scripts/calibrate_embeddings.py` embeds 33 facts, each with a paraphrase, a version
whose value changed and two questions, plus 24 questions about subjects no fact covers.
For `xenova/multilingual-e5-small`:

| Fact against fact | Pairs | Lowest | Median | Highest |
|---|---|---|---|---|
| Same fact, value changed | 33 | 0.893 | 0.964 | 0.984 |
| Same fact, other words | 33 | 0.874 | 0.932 | 0.962 |
| Same topic, different fact | 33 | 0.787 | 0.857 | 0.904 |
| Different topic | 495 | 0.735 | 0.841 | 0.899 |

| Question against fact | Questions | Lowest | Median | Highest |
|---|---|---|---|---|
| A question and its own fact | 66 | 0.804 | 0.851 | 0.921 |
| ... of those, sharing no word with the fact | 13 | 0.804 | 0.824 | 0.853 |
| A question and the best other fact | 66 | 0.787 | 0.832 | 0.873 |
| An off-topic question and the best fact | 24 | 0.740 | 0.805 | 0.831 |

### Recall floor: 0.83

A fact that matches a question only semantically must score at least this to be returned.

| Floor | Right fact still returned | Off-topic question answered with nothing |
|---|---|---|
| 0.80 | 100.0% | 33.3% |
| 0.81 | 95.5% | 54.2% |
| 0.82 | 90.9% | 75.0% |
| **0.83** | **83.3%** | **91.7%** |
| 0.84 | 74.2% | 100.0% |

No floor separates the two groups, and the store-relative scores that were tried (z-score,
distance from the store's mean) separate them worse than the raw cosine. 0.83 favours
silence: a store of a few thousand facts has more near misses than these 33, so a lower
floor returns unrelated facts more often than the table suggests.

The honest limit of this model is the second row of the question table. Of the 13 questions
that share no word with their fact, the ones only a model can answer, the right fact ranks
first for 5 and clears the floor for 4. Most of what semantic recall finds, the word index
finds too. Keywords written by the agent (synonyms, root words) remain the main way a fact
is found in other words.

### Conflict: 0.90, and only with a shared entity

| Cut-off | Value changed | Other words | Same topic | Different topic |
|---|---|---|---|---|
| 0.88 | 100.0% | 97.0% | 12.1% | 3.0% |
| **0.90** | **97.0%** | **93.9%** | **3.0%** | **0.0%** |
| 0.92 | 97.0% | 63.6% | 0.0% | 0.0% |

Among 26 pairs of different facts about one entity (five facts about a person, five about
a warehouse, four about a supplier) one scores 0.90 or more and four score 0.88 or more.

A score alone is never treated as a conflict for this family, and the reason is this
measurement of facts written on one template:

| Pair | Score |
|---|---|
| "Stok gula di gudang tinggal 20 karung" / "Stok tepung di gudang tinggal 20 karung" | 0.978 |
| "Ongkir ke Bandung Rp 15.000 ..." / "Ongkir ke Surabaya Rp 21.000 ..." | 0.954 |
| "Cabang timur buka pukul 08.00" / "Cabang barat buka pukul 10.00" | 0.939 |
| "Stok gula ... tinggal 20 karung" / "Stok gula ... tinggal 5 karung" (a real update) | 0.952 |
| "Toko tutup setiap hari Minggu" / "Toko tutup setiap hari Senin" (a real update) | 0.964 |

Two facts about different things score as high as an update of one fact (0.92 to 0.98
against 0.94 to 0.98, seven pairs each). Only the entities the agent supplies say
whether two facts are about one thing, so a semantic conflict needs a shared specific
entity. Two facts that name different external resources (two sheet ids, two URLs) are
never a conflict unless their wording is nearly identical.

### Tuning and other models

```toml
[embeddings]
recall_floor = 0.81                 # more recall, more unrelated results
conflict_similarity = 0.88
conflict_strict_similarity = 0.95   # allow a conflict on similarity alone (off for E5)
```

A model outside the E5 family gets generic cut-offs (0.35, 0.82, 0.88) that suit models
whose scores spread over most of the range. They are a starting point, not a measurement.
Run the script against your own model and set the three values from what it prints:

```bash
python scripts/calibrate_embeddings.py --provider ollama --model embeddinggemma
```

Whether a larger model does better on Indonesian paraphrases has not been measured here.

## How vectors are made and kept

- One text per forward pass. The quantized model computes its activation ranges over its
  whole input, so in a batch a text's vector depends on its neighbours (cosine 0.997 against
  the same text alone). Alone it is exact and repeatable, and for this model also faster.
- The text is embedded before any write lock is taken, once per `remember`.
- Vectors are L2-normalized float32 BLOBs in the same SQLite file as the fact, stored with
  the model name. `forget` removes them with the fact. Vectors of another model are ignored,
  and `ingatan doctor` counts the facts that still need `ingatan reindex`.
- Quantized arithmetic can differ slightly between CPU families. The cut-offs were measured
  on x86-64; `tests/search/test_real_model.py` and the calibration script show whether they
  hold on another machine.

## Checking it yourself

```bash
python scripts/calibrate_embeddings.py                 # the tables above
python scripts/measure_embed_ram.py                    # memory, stage by stage
python -m evals --embeddings local                     # every eval scenario with the model on
INGATAN_TEST_MODEL_DIR=~/.ingatan/models python -m pytest tests/search/test_real_model.py
```

`python -m evals --embeddings local` writes `evals/reports/latest-semantic.md`. Like the
default run it fails if anything opens a socket.
