# Verification report

The package was checked in the supplied execution environment with Python 3.13, PyTorch 2.10 CPU, and torchvision 0.25.

## Smoke test

`python scripts/smoke_test.py` completed the following stages on a generated three-class ImageFolder dataset:

1. deterministic manifest creation;
2. clean classifier training;
3. fixed feature-scale calibration;
4. SNR-conditioned confusion graph construction;
5. pairwise codebook construction;
6. training of proposed and full-feature schemes;
7. cell-wise evaluation and aggregated CSV generation.

The second execution skipped completed checkpoints and result cells.

## Interruption test

A clean-training process was terminated with SIGTERM after checkpoints had been created. The process wrote `emergency.pt`. Re-running the same command reported:

`[resume] ... at epoch 1, batch 4`

This verifies restoration inside an incomplete epoch rather than incorrectly advancing to the next epoch.

## Scope limitation

This release implements the main analog feature-transmission path and its direct ablations. JPEG/BPG plus coded modulation, reconstruction-oriented DeepJSCC, OFDM/TDL, and LDPC are not included in the executable pipeline and must not be claimed as completed baselines.
