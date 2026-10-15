"""Projected bias of sampled grouped counting, before you spend the cluster time.

Sampled counting estimates H(K) from B draws, so it cannot exceed log2(B) while
H(K|X) is computed analytically and has no such cap. When H(K|X) approaches log2(B)
the reported MI = H(K)hat - H(K|X) collapses, and past it the MI goes NEGATIVE. That
is not a subtle bias, it is arithmetic, and it has already happened twice in this
project (fig2 at C=50, and the whole dilution task).

The fix is to find out before training, not after. Call projected_counting_bias on
the design and refuse to launch if it is too large.

Calibration: the exponent comes from estimator_crosscheck in the archive, which
measured sampled counting against exact enumeration over 12 points of the G=5, C=30
arrays and fitted

    bias [bits] ~= 3.34 * (B / 2^H(K)) ** -0.92

so it is the ratio of budget to EFFECTIVE ALPHABET 2^H(K) that matters, not the
budget alone. Bias falls below 0.1 bits at a ratio near 46.
"""
import math
from collections import Counter

__all__ = ["projected_counting_bias", "counting_budget_report"]

# Measured on this project's cell arrays. FIRING_P is the per-cell firing rate that
# reproduces the observed H(Y|X) per cell (~0.4 bits once a cell holds >1 receptor);
# TYPICAL_MI is the largest KT MI seen across the 224 fig2 runs (7.86), rounded up.
FIRING_P = 0.0905
TYPICAL_MI = 8.0


def _binomial_entropy(n, p):
    terms = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    return -sum(q * math.log2(q) for q in terms if q > 0)


def projected_counting_bias(gene_sets, batch_size, firing_p=FIRING_P,
                            typical_mi=TYPICAL_MI):
    """(bias in bits, H(K|X), H(K)) for one design point.

    gene_sets is one array: cells with an IDENTICAL set are one group, and a group of
    n contributes H(Binom(n, p)) rather than n*H(p) — which is exactly why grouping
    changes whether counting is usable at all.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    sizes = Counter(tuple(gs) for gs in gene_sets).values()
    h_kx = sum(_binomial_entropy(n, firing_p) for n in sizes)
    h_k = typical_mi + h_kx
    return 3.34 * (batch_size / 2 ** h_k) ** -0.92, h_kx, h_k


def counting_budget_report(rows, batch_size, key="gene_sets", limit=0.2,
                           firing_p=FIRING_P, typical_mi=TYPICAL_MI):
    """Print the projection per design point and RAISE if the worst exceeds `limit`.

    Raising matters more than printing: above ~0.2 bits the bias differs between
    design points, so it distorts the SHAPE of a curve rather than shifting it, and
    no amount of staring at the figure afterwards will reveal that.
    """
    print(f"\n{'point':>5} {'cells':>6} {'groups':>7} {'H(K|X)':>8} {'H(K)~':>7} "
          f"{'bias':>7} {'B for 0.1b':>12}")
    worst = 0.0
    for index, row in enumerate(rows):
        bias, h_kx, h_k = projected_counting_bias(row[key], batch_size,
                                                  firing_p, typical_mi)
        worst = max(worst, bias)
        print(f"{index:>5} {len(row[key]):>6} {len(set(map(tuple, row[key]))):>7} "
              f"{h_kx:>8.1f} {h_k:>7.1f} {bias:>6.2f}b {46 * 2 ** h_k:>12,.0f}")
    print(f"\nbudget {batch_size:,} = 2^{math.log2(batch_size):.0f}; "
          f"worst projected bias {worst:.2f} bits")
    if worst > limit:
        raise ValueError(
            f"Projected counting bias {worst:.2f} bits exceeds {limit} bits. Above "
            f"this the bias differs BETWEEN design points, so it distorts the curve "
            f"shape. Raise the batch size, or cut n_cells: H(K|X) is a sum over "
            f"cells, so 2^H(K) grows close to exponentially in C while an affordable "
            f"budget grows linearly.")
    return worst
