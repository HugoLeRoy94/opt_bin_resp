"""Evaluation-time measurements: the valid names and the counting helpers.

These are what a test script records into test_<label>.json. They are
separate from the training
objective: the loss optimizes one bound, these record several, so a run can be
judged by an estimator it was not trained on.

Adding a measurement means a MEASUREMENTS entry and its branch in
train/runner.py::Runner._eval_stats.

Acronyms: MI = mutual information. KT = Kolchinsky-Tracey. MM = Miller-Madow
(a bias correction for plug-in entropy, +(K-1)/(2 B ln2) with K distinct
observed symbols and B samples).
"""
import math

import torch

from estimate.counting import entropy_from_counts


def miller_madow_entropy(activity):
    """Plug-in and MM entropy of the HARD codewords (activity thresholded at 0.5)."""
    B, R = activity.shape
    codes = (activity > 0.5).long()
    _, counts = torch.unique(codes, dim=0, return_counts=True)
    H_plugin, H_MM, K_hat, _, missing = entropy_from_counts(counts)
    return H_plugin, H_MM, K_hat, math.log2(B), math.ldexp(float(K_hat), -R), missing


def codeword_entropy(activity):
    """Hard-codeword plug-in and MM entropy; activity binarised at 0.5.

    K_frac = K_hat / 2^R is the fraction of the alphabet ever observed, and
    missing_mass is the Good-Turing estimate of the probability never sampled.
    Both say whether the plug-in number is trustworthy.
    """
    H_plugin, H_MM, K_hat, log2_B, K_frac, missing = miller_madow_entropy(activity.detach())
    return {
        'codeword_entropy_plugin': H_plugin,
        'codeword_entropy_mm':     H_MM,
        'codeword_entropy_log2B':  log2_B,
        'codeword_entropy_K_hat':  float(K_hat),
        'codeword_entropy_K_frac': K_frac,
        'codeword_entropy_missing_mass': missing,
    }


# Every name a test script or Config.measurement_fns may request. The runner
# computes them all in Runner._eval_stats, accumulated over the whole budget.
MEASUREMENTS = frozenset({
    "entropy_kt",                    # KT lower bound on H(Y)
    "entropy_kt_upper",              # KT upper bound on H(Y)
    "conditional_entropy_response",  # H(Y|X)
    "mutual_information_kt",         # KT lower bound on I(Y;X)
    "mutual_information_kt_upper",   # KT upper bound on I(Y;X)
    "codeword_entropy",              # thresholded outputs, counted
    "grouped_information",           # exact count enumeration, cells
    "grouped_counting",              # sampled group counts, cells
})
