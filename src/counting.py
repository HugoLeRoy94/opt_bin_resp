"""Frequency-based entropy for arbitrary discrete vector symbols, on CPU."""
import math
from typing import Optional, Sequence

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


_MAX_KEY = 1 << 62      # headroom below int64 overflow


class SymbolCounter:
    """Merge observed integer rows, retaining U unique symbols and frequencies.

    Two storage paths, chosen once:

    `radices` given and prod(radices) < 2^62
        Each row is packed into ONE int64, most significant digit first, so sorting
        keys reproduces the lexicographic row order that torch.unique(dim=0) gives.
        Keys are accumulated raw and reduced once, in `entropy`. Cost is a single
        O(B log B) sort and 8 bytes per observation.

    otherwise
        Rows are kept as a (U, J) table and re-reduced on every update. Correct for
        any alphabet, but every chunk re-sorts the whole accumulated table, so the
        total cost is quadratic in the number of observations. Only take this path
        when the alphabet genuinely cannot be packed.

    No thresholding of count values greater than one, and no alphabet-sized
    allocation on either path.
    """

    def __init__(self, radices: Optional[Sequence[int]] = None):
        self._strides = self._radices = None
        if radices is not None:
            values = [int(r) for r in radices]
            if min(values, default=1) < 1:
                raise ValueError("Radices must be positive.")
            total = 1
            for r in values:
                total *= r
            if total < _MAX_KEY:
                strides, step = [0] * len(values), 1
                for j in range(len(values) - 1, -1, -1):
                    strides[j] = step               # column 0 ends up most significant
                    step *= values[j]
                self._strides = torch.tensor(strides, dtype=torch.int64)
                self._radices = torch.tensor(values, dtype=torch.int64)
        self._pending = []       # packed path: raw keys, reduced once in _reduce
        self._unique_keys = None
        self._symbols = None
        self._counts = None
        self.n_samples = 0

    # --- accumulation -----------------------------------------------------

    def update(self, symbols):
        if symbols.ndim != 2 or symbols.shape[0] == 0:
            raise ValueError("Expected a nonempty matrix of discrete symbols.")
        symbols = symbols.detach().to(device="cpu", dtype=torch.int64)
        if self._strides is not None:
            if symbols.shape[1] != self._strides.numel():
                raise ValueError("Symbol dimension changed between chunks.")
            if (symbols < 0).any() or (symbols >= self._radices).any():
                raise ValueError("Symbol value outside the declared radices.")
            self._pending.append(symbols @ self._strides)
            self._counts = self._symbols = self._unique_keys = None   # invalidate
        else:
            rows, counts = torch.unique(symbols, dim=0, return_counts=True)
            if self._symbols is not None:
                if rows.shape[1] != self._symbols.shape[1]:
                    raise ValueError("Symbol dimension changed between chunks.")
                merged = torch.cat((self._symbols, rows))
                weights = torch.cat((self._counts, counts))
                rows, inverse = torch.unique(merged, dim=0, return_inverse=True)
                counts = torch.zeros(rows.shape[0], dtype=torch.int64).scatter_add_(
                    0, inverse, weights)
            self._symbols, self._counts = rows, counts
        self.n_samples += symbols.shape[0]

    def _reduce(self):
        """Collapse the pending packed keys into (unique keys, counts). Idempotent.

        Only the counts are needed for an entropy, and they cost 8 bytes per unique
        symbol. The (U, J) row table is reconstructed lazily by `symbols`, because
        materialising it would undo the whole point of packing.
        """
        if self._strides is None or self._counts is not None:
            return
        if not self._pending:
            raise ValueError("Counting requires at least one observation.")
        self._unique_keys, self._counts = torch.unique(
            torch.cat(self._pending), return_counts=True)

    # --- results ----------------------------------------------------------

    @property
    def symbols(self):
        self._reduce()
        if self._symbols is None:
            self._symbols = ((self._unique_keys[:, None] // self._strides[None, :])
                             % self._radices[None, :])
        return self._symbols

    @property
    def counts(self):
        self._reduce()
        return self._counts

    def entropy(self):
        if self._strides is None and self._counts is None:
            raise ValueError("Counting requires at least one observation.")
        self._reduce()
        return entropy_from_counts(self._counts)
