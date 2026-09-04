# Patch notes

## Bug found and fixed

**File:** `scripts/run_pipeline.py`

**Symptom:** the staged execution options `--from-stage` / `--to-stage` (documented
in the CLI as a way to split the pipeline across separate runs, e.g. one job per
cluster submission) were broken for any `--to-stage` earlier than `evaluation`.

- `python scripts/run_pipeline.py --config ... --to-stage data` always crashed with
  `FileNotFoundError: Missing clean checkpoint: .../checkpoints/clean/best.pt`,
  even though the user explicitly asked the run to stop after the data stage.
- `python scripts/run_pipeline.py --config ... --to-stage clean` always crashed with
  `FileNotFoundError: Missing artifact: .../artifacts/feature_scale.pt`, even though
  the user explicitly asked the run to stop after the clean-classifier stage.

**Root cause:** two sanity checks —

```python
clean_best = root / "checkpoints" / "clean" / "best.pt"
if not clean_best.exists():
    raise FileNotFoundError(...)
...
for required in (scale_path, graph_path, codebook_path):
    if not required.exists():
        raise FileNotFoundError(...)
```

ran unconditionally after the `clean` and `artifacts` stage blocks, regardless of
whether `--to-stage` had asked the pipeline to stop before those files were ever
supposed to exist. They were meant only to guard the case where a later stage is
about to run without having produced (or been given) its prerequisites — e.g.
`--from-stage artifacts` on a machine that never trained the clean classifier.

**Fix:** gated each check on whether a later stage that actually needs the file is
in scope (`end >= 2` for the clean checkpoint, `end >= 3` for the artifact trio),
mirroring the existing `start <= N <= end` stage-gating pattern used everywhere
else in the file:

```python
if end >= 2 and not clean_best.exists():
    raise FileNotFoundError(...)
...
if end >= 3:
    for required in (scale_path, graph_path, codebook_path):
        if not required.exists():
            raise FileNotFoundError(...)
```

**Verification performed:**
- Ran `scripts/smoke_test.py` before and after the fix — identical results
  (`F1=0.9327/1.0/0.8611/1.0/1.0/1.0`), confirming no regression to the default
  full pipeline run.
- Manually staged a run through all five stages one at a time
  (`--to-stage data`, `--from-stage clean --to-stage clean`,
  `--from-stage artifacts --to-stage artifacts`,
  `--from-stage semcom --to-stage semcom`,
  `--from-stage evaluation --to-stage evaluation`); all five completed without
  error and produced a valid `summary_aggregated.csv`.
- Verified the SIGTERM interruption/resume path in `train_clean_classifier`
  (emergency checkpoint saved mid-epoch with correct `batch_in_epoch` and loader
  RNG state; resume picked up at the exact epoch/batch) — this part of the code
  was already correct and needed no change.
- Ran `scripts/verify_checkpoints.py` on the staged run: all 76 `.pt` files
  readable.

No other correctness issues were found in `src/semcom/` (channel, policy, losses,
metrics, checkpoint, data, models, engine) or in the remaining scripts
(`evaluate.py`, `build_artifacts.py`, `train_clean.py`, `train_semcom.py`,
`prepare_dataset.py`, `analyze_artifacts.py`, `plot_results.py`, `_common.py`)
during this audit.
