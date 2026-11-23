# Memory use (NFR-4)

Two questions: what one stdio server process holds, and what the optional embedding model
adds to it. Resident set size (RSS) is the working set on Windows and VmRSS on Linux. The
threshold is OQ-19 and is the owner's call; these are measurements only.

## One stdio server process

Measured by `scripts/measure_ram.py`: N facts saved through the `remember` tool, then a few
`recall` and `brief` calls. "Peak" is the highest value the operating system recorded. With
a model, "loaded" includes it: the facts were just embedded and the model has not been idle
long enough to be unloaded (300 seconds by default).

| Date | Machine | Python | mcp | Embeddings | Facts | Idle MB | Loaded MB | Peak MB |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | Windows 11 (AMD64) | 3.11.16 | 2.3.0 | none | 500 | 70.2 | 73.6 | 73.6 |
| 2026-11-23 | Windows 11 (AMD64) | 3.14.3 | 2.3.0 | none | 500 | 75.9 | 79.5 | 79.5 |
| 2026-11-23 | Windows 11 (AMD64) | 3.14.3 | 2.3.0 | local (xenova/multilingual-e5-small) | 500 | 76.0 | 275.0 | 324.1 |
| 2026-11-23 | Windows 11 (AMD64) | 3.14.3 | 2.3.0 | local (intfloat/multilingual-e5-small) | 500 | 76.1 | 608.6 | 932.9 |

Without a model the server imports neither NumPy nor ONNX Runtime, whether or not they are
installed.

## What the local model costs, step by step

Measured by `scripts/measure_embed_ram.py`: one fresh process per model, memory read after
each step. "Private" is the Windows commit charge.

| Step | int8 model RSS MB | float32 model RSS MB |
|---|---|---|
| Python with Ingatan imported | 28.6 | 28.7 |
| Model loaded, first query answered | 225.4 | 558.9 |
| 32 facts embedded | 225.5 | 559.0 |
| 32 long notes embedded (512 tokens each) | 226.9 | 560.3 |
| Highest value at any point | 277.2 | 885.8 |
| Model unloaded | 71.4 | 70.8 |

| Timing | int8 | float32 |
|---|---|---|
| Load the model and answer the first query | 0.68 s | 1.75 s |
| One query, model warm | 2.0 ms | 7.3 ms |
| 32 facts | 0.14 s | 0.54 s |
| 32 long notes | 2.71 s | 13.61 s |

## Why it is built the way it is

Each line is a measurement from 2026-11-23 on the machine above. The "before" column is the
previous code on the same day and machine (its server figures were taken with 200 facts).

| Choice | Before | Now |
|---|---|---|
| float32 weights loaded through `fastembed`, the default | 843.7 MB loaded, 952.9 MB peak | not the default; `fastembed` is no longer used |
| int8 weights | 509.1 MB loaded (through `fastembed`) | 275.0 MB loaded |
| Tokenizer: Hugging Face `tokenizers` against SentencePiece | 250 MB for the 250,002-entry vocabulary | 35 MB |
| ONNX CPU memory arena | 32 long notes in one batch left 1.0 GB resident until unload | off: nothing stays |
| Texts per forward pass | 32 | 1: peak 277 MB, and each text gets the same vector every time |
| `import numpy` at start-up, 16-core machine | 490 MB of commit charge, also with embeddings off | not imported until a vector is scored; BLAS capped at one thread |

Two settings that turned out not to matter for memory: the number of ONNX threads (1 or 2:
same RSS, 1 is slower) and the graph optimization level.

The model is unloaded after `idle_unload_seconds` (default 300, 0 keeps it). On Windows the
process returned to 71 MB. On Linux the heap is trimmed with `malloc_trim` as well.

Every agent session that uses stdio runs its own server process and therefore its own copy
of the model. To share one copy between sessions, set `[embeddings] shared = true` (one
model process for the data folder, see [EMBEDDINGS.md](EMBEDDINGS.md)), or use HTTP mode
(one process for all tokens) or the `ollama` provider.

## How to run it

```
python scripts/measure_ram.py                      # 500 facts, appends a row to the table above
python scripts/measure_ram.py --facts 2000 --no-write
INGATAN_EMBED_PROVIDER=local python scripts/measure_ram.py
python scripts/measure_embed_ram.py                # the step-by-step table
python scripts/measure_embed_ram.py --sessions 3            # three sessions, a model copy each
python scripts/measure_embed_ram.py --sessions 3 --shared   # three sessions, one shared process
```

The last two add up the resident memory of processes that are alive at the same time
(`[embeddings] shared`, see [EMBEDDINGS.md](EMBEDDINGS.md)). They have not been run on a
machine with the model yet, so this page has no figure for them.

Both scripts start subprocesses in a temporary data folder and read memory with the
standard library: `K32GetProcessMemoryInfo` on Windows, `/proc/<pid>/status` on Linux,
`ps -o rss=` on macOS (no peak value there).

## Status

- Windows 11: measured, tables above. The first row is from Python 3.11.16 and the others
  from Python 3.14.3, so the two baselines (70 MB and 76 MB) are not directly comparable.
- Linux, including the 7.7 GB VPS: **not measured yet.** Run the two commands above there
  and commit the rows. OQ-19 (the threshold) stays open until then.
- macOS: not measured.
