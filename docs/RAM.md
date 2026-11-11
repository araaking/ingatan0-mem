# Memory use of one stdio server process (NFR-4)

Resident set size (RSS) of one `ingatan.mcp.server` process, measured by
`scripts/measure_ram.py`. "Loaded" means N facts saved through the `remember` tool, then a few
`recall` and `brief` calls. "Peak" is the highest value the operating system recorded.
The threshold is OQ-19 and is the owner's call; these are measurements only.

| Date | Machine | Python | mcp | Facts | Idle MB | Loaded MB | Peak MB | Command |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | Windows 11 (AMD64) | 3.11.16 | 2.3.0 | 500 | 70.2 | 73.6 | 73.6 | `python scripts/measure_ram.py` |

## How to run it

```
python scripts/measure_ram.py            # 500 facts, appends a row above
python scripts/measure_ram.py --facts 2000 --no-write
```

The script starts one server subprocess in a temporary data folder, uses it only through the
MCP tools, and reads RSS with the standard library: `K32GetProcessMemoryInfo` on Windows,
`/proc/<pid>/status` on Linux, `ps -o rss=` on macOS (no peak value there; peak equals loaded).

## Status

- Windows 11: measured, row above. The run used a Python 3.11.16 interpreter because the
  Python 3.14 environment on this machine could not load `pydantic_core` (a Windows
  Application Control policy blocks that file). Python 3.14 numbers may differ a little.
- VPS (7.7 GB, Linux): **not measured yet. The owner still has to run the command above on
  the VPS and commit the row.** OQ-19 (the threshold) stays open until then.
