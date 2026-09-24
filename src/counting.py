"""Frequency-based entropy for arbitrary discrete vector symbols, on CPU."""
import math

import torch


def entropy_from_counts(counts):
    """Plug-in and Miller-Madow estimates, plus the Good-Turing missing mass.

    Returns (plugin, miller_madow, n_unique, log2_n, missing_mass).

    `missing_mass` is f1/n, where f1 counts symbols seen EXACTLY ONCE. It is the
    Good-Turing estimate of the probability mass that was never sampled, and it is
    the quantity that says whether n was large enough for these frequencies to mean
    anything. A plug-in entropy computed where it is large measures the sample
    size, not the distribution.

    Prefer it to n_unique/n: that counts every distinct symbol rather than the
    once-only ones, so it overstates how badly a sample is covered and has no
    corresponding theory.

    Neither correction is a population bound. Miller-Madow adjusts by the symbols
    SEEN, which is itself an underestimate when most of the alphabet is missing.
    """
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Counting requires at least one observation.")
    p = counts.double() / n
    plugin = -(p * p.log2()).sum().item()
    unique = counts.numel()
    singletons = int((counts == 1).sum())
    return (plugin, plugin + (unique - 1) / (2 * n * math.log(2)),
            unique, math.log2(n), singletons / n)


class SymbolCounter:
    """Merge observed integer rows, retaining U unique symbols and frequencies.

    No mixed-radix integer encoding (which could overflow), no alphabet-sized
    allocation, and no thresholding of count values greater than one.
    """
    def __init__(self):
        self.symbols = None
        self.counts = None
        self.n_samples = 0

    def update(self, symbols):
        if symbols.ndim != 2 or symbols.shape[0] == 0:
            raise ValueError("Expected a nonempty matrix of discrete symbols.")
        symbols = symbols.detach().to(device="cpu", dtype=torch.int64)
        rows, counts = torch.unique(symbols, dim=0, return_counts=True)
        if self.symbols is not None:
            if rows.shape[1] != self.symbols.shape[1]:
                raise ValueError("Symbol dimension changed between chunks.")
            merged = torch.cat((self.symbols, rows))
            weights = torch.cat((self.counts, counts))
            rows, inverse = torch.unique(merged, dim=0, return_inverse=True)
            counts = torch.zeros(rows.shape[0], dtype=torch.int64).scatter_add_(0, inverse, weights)
        self.symbols, self.counts = rows, counts
        self.n_samples += symbols.shape[0]

    def entropy(self):
        if self.counts is None:
            raise ValueError("Counting requires at least one observation.")
        return entropy_from_counts(self.counts)
