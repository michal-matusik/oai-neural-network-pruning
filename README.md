# Reconstructed Neural Network Pruning - Polish AI Olympiad I

This repository is a reconstructed reference solution for the Neural Network Pruning task in the Polish Artificial Intelligence Olympiad.
It is not the author's original competition submission.

The task fixed a 128-1024-10 sigmoid regression architecture and rewarded sparsity while penalizing validation MSE.
The reference approach uses iterative global magnitude pruning with validation-controlled threshold selection, avoiding architecture changes as required by the task.

![Neural-network pruning illustration](assets/task-pruning.png)

*Task illustration embedded in the official Polish AI Olympiad I notebook.*

## Quick start

`python -m unittest discover -s tests -v`

## Validation

The smoke test verifies exact global magnitude sparsity and mask behavior on NumPy parameters.
It does not claim an official validation or leaderboard score.

## Provenance

The original Polish task notebook is retained as `notebooks/original_submission.ipynb`.
`docs/task_en.md` is an English task summary.
