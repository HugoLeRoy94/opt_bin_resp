"""Frequency-based entropy for arbitrary discrete vector symbols, on CPU."""
import math

import torch


def entropy_from_counts(counts):
    """Plug-in and Miller–Madow estimates; no claim of a population bound."""
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Counting requires at least one observation.")
    p = counts.double() / n
    plugin = -(p * p.log2()).sum().item()
    unique = counts.numel()
    return plugin, plugin + (unique - 1) / (2 * n * math.log(2)), unique, math.log2(n)


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
