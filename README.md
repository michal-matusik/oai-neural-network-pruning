# Reconstructed Neural Network Pruning - Polish AI Olympiad I

This repository is a reconstructed reference solution for the Neural Network Pruning task in the Polish Artificial Intelligence Olympiad.
It is not the author's original competition submission.

The task fixed a 128-1024-10 sigmoid regression architecture and rewarded sparsity while penalizing validation MSE.
The reconstruction identifies the sparse linear structure in the official training targets and encodes it with paired sigmoid units inside the required architecture.

![Neural-network pruning illustration](assets/task-pruning.png)

*Task illustration embedded in the official Polish AI Olympiad I notebook.*

## Quick start

```bash
python scripts/download_data.py
python -m src.evaluate data
python -m unittest discover -s tests -v
```

## Validation

On the 2,000 official public validation examples, the model reaches 1.004804 MSE, 99.6740% sparsity, and a 0.993615 combined metric.
That exceeds the notebook's 0.95 full-credit threshold and corresponds to an estimated 100% validation result.
No hidden-test or leaderboard score is claimed.
See `SOLUTION.md` for the derivation and evaluation details.

## Provenance

The original Polish task notebook is retained as `notebooks/original_submission.ipynb`.
`docs/task_en.md` is an English task summary.
