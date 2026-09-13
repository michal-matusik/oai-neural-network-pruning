# Solution notes

This repository contains a retrospective reconstruction and not the original competition submission.

The official evaluator fixes a 128-1024-10 sigmoid MLP and rewards low mean squared error together with exact zero-valued parameters.
Inspection of the official training split shows that the ten targets are almost linear and that only 99 regression coefficients exceed an absolute magnitude of 0.1.

The reconstruction fits a multivariate linear regression on the training split.
For every input feature, a pair of hidden units encodes `sigmoid(alpha*x) - sigmoid(-alpha*x)`.
For small `alpha`, this is a highly accurate scaled linear function.
The output weights rescale the paired activations to reproduce the retained regression coefficients while all unused parameters remain exactly zero.

On the 2,000 official public validation examples, the model has 1.004804 MSE and 99.6740% parameter sparsity.
The official combined metric is 0.993615, above the 0.95 full-credit threshold, for an estimated validation result of 100%.
Training and validation take under two seconds on this Apple Silicon CPU.
No hidden-test or leaderboard score is claimed.
