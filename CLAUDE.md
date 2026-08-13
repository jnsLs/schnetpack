# schnetpack_fork

## Environment

Use the `schnetpack_fork` conda environment for everything in this project —
running tests, `spktrain`, scripts, regenerating test data:

```bash
conda activate schnetpack_fork
# or, non-interactively:
/home/jonas/miniforge3/envs/schnetpack_fork/bin/python -m pytest tests/ -q
```

It is the only environment whose editable install points at this checkout
(`/home/jonas/Documents/schnetpack_fork/src`). The similarly named `schnetpack`
environment points at the **sibling** repo `/home/jonas/Documents/schnetpack`
and ships a different torch build, so using it here silently tests the wrong
source tree and produces different float32 results.

## Golden regression tests

`tests/testdata/newton_step_golden.pt` records historical values. **Do not
regenerate it to make a failing test pass.** Regenerating overwrites the
evidence with whatever the code currently does, which is precisely the
regression the test exists to catch, and `generate_golden.py` writes in place
with no dry-run.

When the golden test fails, first classify the failure:

- **Key-name mismatch** (`missing from actual` / `unexpected in actual`) — a
  rename, not a numeric change. Fix it *without* re-running the model: update
  `OUTPUT_LABELS`/`KEYS` in `tests/surrogate/golden.py`, which exist so that
  renaming a source key does not invalidate recorded values, or migrate the
  keys of the stored file in place. Values must be carried over untouched.
- **Value mismatch** — treat as a real regression until proven otherwise.
  Investigate the arithmetic. Only re-record once you can explain exactly which
  change moved the number and why the new value is correct, and say so in the
  commit message.

Comparison is bit-exact (`rtol=0, atol=0`). Keep it that way; a value that
drifts is information. Record only in the `schnetpack_fork` environment — a
different torch build shifts float32 results by ~1e-8 and silently invalidates
the file.
