# Reconstructed Neural Network Pruning - Polish AI Olympiad I

This repository is a reconstructed reference solution for the Neural Network Pruning task in the Polish Artificial Intelligence Olympiad.


The task fixed a 128-1024-10 sigmoid regression architecture and rewarded sparsity while penalizing validation MSE.
The reconstruction identifies the sparse linear structure in the official training targets and encodes it with paired sigmoid units inside the required architecture.

![Neural-network pruning illustration](assets/task-pruning.png)

*Task illustration embedded in the official Polish AI Olympiad I notebook.*

## Method

The evaluator fixes the architecture (`128 -> 1024(sigmoid) -> 10`, no bias omitted), so
"pruning" here means constructing a set of mostly-zero weights that still realizes an accurate
function rather than iteratively removing weights from a trained dense network. The construction
rests on one identity: `sigmoid(z) - sigmoid(-z) = tanh(z/2)` (a half-angle identity, verified
numerically and derivable directly from the sigmoid/hyperbolic-tangent definitions). For small
`z`, `tanh(z/2) ~ z/2`, so a *pair* of sigmoid units fed `+alpha*x` and `-alpha*x`
(`alpha = 0.003` here) and differenced with equal-and-opposite output weights realizes
`~alpha*x/2` — an (almost) linear function of `x` — despite the layer's activation being a fixed
nonlinearity. This is the same small-signal-linearization trick used to linearize a diode or
transistor's I-V curve around an operating point in circuit analysis.

Concretely: a `LinearRegression` is fit on the official training split; coefficients with
`|coef| <= 0.1` are zeroed. Each of the 128 input features gets exactly one `+alpha`/`-alpha`
pair of first-layer units (`2 x 128 = 256` nonzero first-layer weights, first-layer bias left at
zero); each of the 99 regression coefficients that survive thresholding gets one
`+coefficient`/`-coefficient` pair of second-layer weights (`2 x 99 = 198` nonzero); the 10
second-layer biases hold the regression intercepts. That accounts for every nonzero parameter:
`256 + 198 + 10 = 464` of `142,346` total, i.e. `1 - 464/142346 = 99.674%` exact zeros — matching
the measured sparsity to four decimal places, since the construction *is* the sparsity pattern
rather than an approximation of it.

`src/pruning.py` also ships a generic `global_magnitude_mask` (rank-and-zero-the-smallest-weights
pruning, applied post-hoc to an already-trained dense network) for comparison/reuse, but it is not
what the validated solution uses — the official score,
`(1 - min(mse, 1000)/1000)^1.5 * sparsity^1.5` (linearly rescaled between 0.085 and 0.95 to 0-100
points), rewards near-total sparsity so heavily, and tolerates MSE so loosely (up to 1000), that
an exact analytic sparse construction dominates an iterative magnitude-pruning search here.

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
