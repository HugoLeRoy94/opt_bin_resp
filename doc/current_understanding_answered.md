# Current understanding — with answers

Copy of `current_understanding.md` with your bracketed comments answered in place.
Your original text is untouched. Your questions are kept as `> Q:` blockquotes so
you can see what was asked, and each is followed by an **A.** block. Merge what you
want and delete the rest.

Numbers are measured from the runs on disk unless stated. Where I am guessing, or
where I got something wrong before, I say so.

---

# receptors array

# cells array

The response of an array of cells is non-deterministic. If $Y \in \mathbb{R}^C$ is
the response of a cell for a given input $X \in \mathbb{R}^L$ (the concentration of
the $L$ ligands in a sniff, 0 if it's not there). We write:

$$MI(Y;X) = H(Y) - H(Y \mid X)$$

## Cell diversity

Given $G$ possible genes, there are $\binom{G}{g}$ ways to make a cell with $g$
genes. In a system composed of $C$ cells, there is at most

$$U = \min\left(C, \sum_{g=1}^G \binom{G}{g}\right) = \min(C,\,2^{G}-1)$$

unique genetic patterns, called cell diversity.

> **Q:** "Measured at G = 5, C = 30, flat size law at mean 2.5: about 18 of the 30
> reachable sets appear." How is that computed?

**A.** Two ways, and they agree.

*Empirically*: draw the array with the sampler and count distinct gene sets,
`len(set(cell_gene_sets))`, averaged over seeds.

*Analytically*: each cell independently lands on gene set $s$ with probability
$q_s$. Under a size law $\pi$ with uniform choice of which genes,

$$q_s \;=\; \frac{\pi(|s|)}{\binom{G}{|s|}}$$

because a size is drawn first, then one of the $\binom{G}{|s|}$ sets of that size
uniformly. Cell $c$ misses $s$ with probability $1-q_s$, and cells are independent,
so all $C$ miss it with probability $(1-q_s)^C$. Therefore

$$\boxed{\;\mathbb{E}[U] \;=\; \sum_{s}\left(1 - (1-q_s)^{C}\right)\;}$$

Checked against simulation, $G=5$, flat size law at mean 2.5 (so sizes 1 to 4, and
30 reachable sets rather than 31):

| $C$ | formula | measured, 10 seeds |
|---|---|---|
| 10 | 8.49 | 8.20 |
| 30 | **18.50** | **18.20** |
| 60 | 25.16 | 25.10 |
| 120 | 29.02 | 29.40 |

Note $U$ here is the *realised* diversity, a random variable. Your $\min(C, 2^G-1)$
is the maximum it can take, attained only by the deterministic designs.

---

Cell diversity should increase the entropy of $Y$.

> **Q:** [the $U_0 \to U_1$ argument] can you reformulate this paragraph, I like my
> argument, I would like to make it more streamlined.

**A.** Suggested rewrite, same argument, tightened:

> Let $\mathcal{P}_U$ be the set of distinct firing probabilities realised by an
> array of diversity $U$. Add one cell of a new genetic type, giving diversity
> $U+1$. Two cases. If the new type's response is a duplicate of an existing one,
> its output is an extra draw from a distribution already present, and it adds no
> new coordinate to the response alphabet: $H_{U+1}(Y) = H_U(Y)$ up to the
> multiplicity term. If its response differs, the joint response gains a coordinate
> that is not a deterministic function of the others, and $H(Y)$ strictly increases.
> So diversity is monotone in $H(Y)$: it can only help, and it helps exactly when
> the new type is genuinely distinguishable.

One caveat worth adding: "duplicate" must mean *identical firing probability for
every input*, not merely identical at one input. That is exactly the condition
`CellGrouping` tests for, which is why the grouped estimator can merge such cells
without approximation.

---

Now, looking at the conditional entropy, given a sniff, cells trigger
independently, which means that

$$H(Y \mid X) \;=\; \mathbb{E}_x\left[\sum_{c=1}^{C} h\big(p_c(x)\big)\right],
\qquad h(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

> **Q:** I don't understand this formula well. Can we demonstrate it? You said, if
> the cells trigger independently, the entropy of the array is the sum of individual
> entropy? Now, you say because there is no coupling, then diversity doesn't enter.
> Which I understand. But I didn't understand well the idea behind this sentence:
> "Adding a new cell type at fixed $g$ changes which $p_c$ occur, but not the typical
> magnitude of $h(p_c)$." Then I understand how to move $h(p_c)$. You called this the
> promiscuity?

**A. The demonstration.** Three steps.

*Step 1, conditional independence.* Fix a sniff $x$. Each cell $c$ computes a drive
$S_c(x)$ from the receptor occupancies, and then fires as an independent Bernoulli
with $P(Y_c = 1 \mid x) = p_c(x)$. The randomness is the cell's own output noise,
drawn separately per cell. Nothing couples two cells once $x$ is fixed. So

$$P(y_1,\dots,y_C \mid x) \;=\; \prod_{c=1}^{C} P(y_c \mid x)$$

*Step 2, entropy of a product is a sum.* For any product distribution,

$$H(Y \mid X{=}x) = -\sum_{y}\prod_c P(y_c|x)\log_2\prod_c P(y_c|x)
= \sum_{c} H(Y_c \mid X{=}x) = \sum_c h(p_c(x))$$

using $\log$ of a product is a sum of logs, and each marginal sums to one.

*Step 3, average over sniffs.* $H(Y\mid X) = \mathbb{E}_x[H(Y\mid X{=}x)]$ by
definition, giving the boxed formula. It is implemented literally in
`bin_loss.compute_response_conditional_entropy`.

**Yes**, conditional on the input, the array entropy is the sum of the individual
entropies. Unconditionally it is *not*: $H(Y) \le \sum_c H(Y_c)$ with strict
inequality whenever cells are correlated through $x$, and that gap is precisely the
mutual information you are trying to maximise.

**A. What the sentence meant, restated.** $H(Y\mid X)$ is a sum of $h$ evaluated at
each cell's probability. It therefore depends only on the *multiset* of values
$\{p_1(x),\dots,p_C(x)\}$, not on which cell has which, nor on how many distinct
types there are. Swap two cells' gene sets and $H(Y\mid X)$ is unchanged. Replace a
duplicated type by a fresh type whose probabilities happen to have the same
distribution and $H(Y\mid X)$ is again unchanged, while $H(Y)$ rises.

So: diversity changes *which* $p_c$ occur, and that changes $H(Y)$. It does not
systematically change the *typical size* of $h(p_c)$, which is what $H(Y\mid X)$
sums. What does change that typical size is $g$, because a cell expressing more
genes averages its drive over more receptors, pulling $p_c$ toward $1/2$ where $h$
is maximal.

**A. Terminology.** I used "promiscuity" loosely for $g$, the number of genes a
cell expresses. It is your word from the notes. Pick one and define it once, since
$g$, "promiscuity" and "expression level" are all in circulation.

**A. On your reading that the two mechanisms differ in nature.** I agree and think
it is worth stating explicitly:

| mechanism | enters through | analogue in the receptor-only model |
|---|---|---|
| cell diversity | $H(Y)$ | direct: more receptors, more response patterns |
| promiscuity $g$ | $H(Y\mid X)$ | **none**, receptors are not mixed |

The second has no receptor-model counterpart, which is exactly why the cell picture
has a maximum and the receptor picture does not.

Measured per cell, $G=5$ heteromers:

| $g$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $H(Y\mid X)/C$ | 0.023 | 0.336 | 0.441 | **0.509** | 0.446 |

{Now, that also increases $H(Y|X)$. Is it true, or is it that increasing $g$ (the
average number of genes in a cell) increases the conditional entropy. Thus raising
$g$ means increasing the cell diversity, but also making the response more "mild".}

**A.** Your second guess is the right one, per the argument above. "Mild" is a good
word: the response becomes less committed, $p_c \to 1/2$.

## Receptor diversity

Increasing $g$ increases the cell diversity the same way in the homomers and
heteromers case. So cell diversity alone cannot explain why heteromers are better
than homomers. So we need to account for receptor diversity, to understand this
raise.

> **Q:** can we now write how increasing receptor diversity can have a different
> impact on $h(p_c)$ in the heteromers than in the homomer case? Is that where the
> difference is from, or does it also come into play in the $H(Y)$ term?

**A. It enters both terms, and the decisive one is $h(p_c)$.** Measured, $G=5$,
$C=30$, complete coverage:

| $g$ | $H(Y)$ het | $H(Y)$ hom | $H(Y\mid X)$ het | $H(Y\mid X)$ hom | $h$/cell het | $h$/cell hom | MI het | MI hom |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.06 | 6.13 | 0.69 | 0.70 | 0.023 | 0.023 | 5.37 | 5.43 |
| 2 | 17.11 | **19.22** | 10.07 | **14.18** | 0.336 | **0.473** | **7.03** | 5.04 |
| 3 | 18.91 | 22.41 | 13.23 | 18.12 | 0.441 | 0.604 | 5.67 | 4.29 |
| 4 | 19.52 | 20.32 | 15.27 | 17.03 | 0.509 | 0.568 | 4.25 | 3.29 |
| 5 | 15.73 | 16.30 | 13.38 | 14.04 | 0.446 | 0.468 | 2.35 | 2.26 |

Read the $g=2$ row carefully, because it is counter-intuitive: **homomers have the
higher $H(Y)$** (19.22 against 17.11) and still lose by 2 bits. Their response
entropy is larger because their cells are noisier, and noise inflates $H(Y)$ without
carrying information. The decisive column is $h$/cell: 0.473 against 0.336.

**Mechanism.** With the `mean` readout the drive is a convex combination
$S_c = \sum_r w_{cr}\,p_r$ of the tuning curves of the receptors the cell carries. {A
convex combination is never sharper than the sharpest curve entering it.}
[can we demonstrate this ? Is the choice of a mean vs threshold or any other mechanism so important ? Or, what should be the lowest mathematical requirement for this statement ( on the blurring of the homomers by adding multiple receptors) to be true.]

- A **homomer** cell expressing $g$ genes can only average the $g$ homomer curves
  that already existed at $g=1$. Averaging can only blur, so $p_c$ moves toward
  $1/2$ and $h(p_c)$ rises steeply. There is no compensating gain.
- A **heteromer** cell expressing $g$ genes acquires receptor species that did not
  exist in the array at $g=1$: the pool goes 5, 65, 353, 605, 629 against a constant
  5 for homomers. Those new species can be sharper or differently centred, so the
  convex combination need not be flat, and $h(p_c)$ rises more slowly.

So the honest one-line statement is: **receptor diversity buys a slower growth of
$H(Y\mid X)$, not a faster growth of $H(Y)$.** That is the opposite of what one
would guess, and it is the part of the story I would put in the paper.

## Genes sampling

### complete coverage

We started with a "complete coverage" sampling which enumerates (within the number
of available cells) all the genetic combinations possible. If I remember correctly,
given an average number of genes $g$, it takes all the combinations with exactly
$g$ genes. Now, an alternative could be enumerating all the combinations with
$1,2,\dots,g$ genes.

> **Q:** It's important to keep in mind that the mean comes as a consequence of the
> enumeration. Can we write a demonstration of how to compute this mean?

**A.** Yes, and it is short. In the cumulative design at level $m$, the array
contains every gene set of size $k \le m$, each appearing the same number of times
(round robin). There are $\binom{G}{k}$ sets of size $k$, and each contributes $k$
genes. So the mean genes per cell is the size-weighted average

$$\bar g(m) \;=\; \frac{\displaystyle\sum_{k=1}^{m} k\binom{G}{k}}
{\displaystyle\sum_{k=1}^{m}\binom{G}{k}}$$

Using $k\binom{G}{k} = G\binom{G-1}{k-1}$ the numerator is $G\sum_{k=0}^{m-1}\binom{G-1}{k}$,
so equivalently

$$\bar g(m) \;=\; G\,\frac{\sum_{k=0}^{m-1}\binom{G-1}{k}}{\sum_{k=1}^{m}\binom{G}{k}}$$

and at $m=G$ this collapses to $\bar g(G) = G\,2^{G-1}/(2^{G}-1) \to G/2$ for large
$G$. Values:

| $G$ | $\bar g(1)$ | $\bar g(2)$ | $\bar g(3)$ | $\bar g(4)$ | $\bar g(5)$ |
|---|---|---|---|---|---|
| 4 | 1 | 1.60 | 2.00 | 2.13 | — |
| 5 | 1 | 1.67 | 2.20 | 2.50 | 2.58 |

The range is narrow because the high-order sets are few. Worth stating as a
limitation: the cumulative design cannot probe $\bar g$ beyond about $G/2$, so it
cannot reach the collapse at $g = G$ that the fixed design shows.

Two caveats for the implementation. When $C$ is not a multiple of the number of
sets the surplus cells land on the first sets, which are the *small* ones, biasing
$\bar g$ down; the code rotates the round-robin start by replicate so this averages
out. And the realised $\bar g$ is reported per run rather than assumed.

Is this method the best sampling strategy to maximize MI?

> **Q:** ok, we cannot answer that through simulations, but can we come up with an
> intuitive reason why it's probably better than any independent sampling?

**A.** Here is the argument I would use. It is an intuition, not a proof.

Fix $C$ cells and a target mean $\bar g$. You are choosing a *multiset* of gene
types. Every choice trades off two things:

- **Coverage.** A type that is absent contributes nothing, and whatever it could
  have discriminated is lost. No amount of replication elsewhere recovers it,
  because the missing discrimination is not a function of the types you kept.
- **Replication.** A type present $n$ times yields a count $K \sim
  \mathrm{Binomial}(n, p)$ rather than a single bit. That is worth something, it
  averages the output noise, but it is subject to diminishing returns: {the
  information about $p$ in $n$ Bernoulli draws grows like $\tfrac{1}{2}\log_2 n$,
  not like $n$.}[How do you compute that ?]

So the marginal value of the $n$-th copy of a type falls off logarithmically, while
the marginal value of a *first* copy of a missing type is finite and does not fall
off at all. A strategy that spends cells on duplicates before covering the support
is therefore paying a logarithmic return where a constant one was available.

Complete coverage spends exactly in that order: cover first, replicate with what is
left. Independent sampling cannot, because it has no memory of what the other cells
drew, so it duplicates and omits at random.

**Important qualifier.** This argument only bites while the array is the
bottleneck. Once the array saturates the environment (see the environment section)
every strategy that reaches the ceiling ties, and "best" becomes meaningless. In the
current runs the fixed design is already at the ceiling, so the observed 1-bit gap
over the independent laws is a statement about *efficiency of reaching a ceiling*,
not about optimality.

The biological drawback to this method is that cells need to talk to each other to
achieve that, which seems unrealistic.

### independant cell samplings

More realistic cell sampling would be that each cell draws genes independently from
one another. We can either draw genes directly according to a given probability, or
use a given probability distribution for $g$ the number of genes expressed in a
cell. We start with 2 comparisons: 1) uniform genes distribution. 2) uniform genes
distribution with exponential distribution of $g$.

> **Q:** explain the exact independent cell samplings, and the issue with estimating
> the mean with the 0 truncated poisson distrib.

**A. The two laws, precisely.**

*Law 1, "each gene independently"* (`--design bernoulli`,
`sample_gene_sets_bernoulli`). Gene $u$ is expressed in a cell with probability
$p_u$, independently of the other genes and of the other cells. With uniform
$p_u = p$ the number of expressed genes is $\mathcal{B}(G,p)$. **The size
distribution is induced, never chosen.** This is the law with no explicit
dependence on $g$ that you asked for.

*Law 2, "size then identity"* (`--design size_law`, `sample_gene_sets_by_size`).
Draw the number of genes $g \sim \pi$, then draw $g$ genes without replacement with
weights $\{p_u\}$. The two laws $\pi$ and $\{p_u\}$ are independent and either can
be varied with the other fixed.

**A. The truncation issue, and a correction to your wording.** It is a zero-truncated
**Binomial**, not Poisson. (The zero-truncated Poisson does appear in this project,
but in the *environment*: it is how many ligands a sniff contains. Different object,
easy to confuse, worth distinguishing in the notes.)

A cell expressing no gene is not a cell, so `sample_gene_sets_bernoulli` rejects and
redraws it. That conditioning changes the mean. Unconditionally $\mathbb{E}[g] = Gp$,
but conditioned on $g \ge 1$,

$$\mathbb{E}[g \mid g \ge 1] \;=\; \frac{Gp}{1-(1-p)^{G}}$$

since $P(g\ge 1) = 1-(1-p)^G$ and the numerator is unchanged because the $g=0$ term
contributes nothing to the sum. The denominator is less than one, so the realised
mean is always **above** $Gp$. At $G=5$, setting $p = 1/G = 0.2$ in the hope of one
gene per cell actually gives $1/(1-0.8^5) = 1.49$ genes per cell, a 49% overshoot.

The code therefore solves the equation above for $p$ by bisection rather than
setting $p = \bar g/G$, so this law sits on the same mean axis as the others. Note
$\bar g = 1$ exactly is unreachable: it needs $p \to 0$, where almost every draw is
empty and the rejection loop does not terminate.

{my expectation is that independant cell sampling is always smaller than complete
coverage.}

> **Q:** Ok, let's keep it as an intuition supported by the argument you gave.

**A.** Agreed. The coverage-beats-replication argument above is the support, and the
measured $G=5$ figures are the evidence: 7.03 fixed, 5.96 flat law, 5.47 exponential
law. State it as an empirical regularity with a mechanism, not as a theorem.

{Second expectation is that increasing the number of cells $C$ to infinity will lead
independant cell sampling to converge toward [complete coverage].}

> **Q:** Ok, half right, but for the support, I understand (I like the phrasing of
> support vs multiplicities, but I would like to explicitly define it). I understand
> the difference, but my expectation is that multiplicity matters way less than
> coverage of the support. [...] Maybe a more exact statement would be that entropy
> of the independent law grows while remaining below the complete coverage.

**A. Definitions.** Describe an array by the histogram of its cell types. Let
$\mathcal{S} = \{s : 2^G-1 \text{ non-empty gene sets}\}$ and let $n_s \ge 0$ be the
number of cells carrying set $s$, with $\sum_s n_s = C$.

- **Support**: $\mathrm{supp} = \{s : n_s > 0\}$, which types are present at all.
- **Multiplicity profile**: the normalised histogram $\{n_s / C\}$, in what
  proportion.

Complete coverage at level $m$ has $\mathrm{supp} = \{s : |s| \le m\}$ and a flat
profile on it. An independent law with type probabilities $q_s$ has, by the law of
large numbers, $n_s/C \to q_s$, so its profile converges to $q$, which is flat only
if $q$ is.

**A. Your refined statement is the right one, and I think it is correct.** As
$C\to\infty$ the support of the independent law converges to the full set (every $s$
with $q_s>0$ appears), so the *structural* difference from complete coverage
disappears and only the profile differs. If, as you say, multiplicity matters much
less than support, the independent law should approach complete coverage from below
without attaining it. I would phrase it as:

> Under an independent law the realised support converges to $\{s : q_s > 0\}$ as
> $C \to \infty$, while the multiplicity profile converges to $q$ rather than to the
> flat profile of complete coverage. If the information is carried mainly by which
> types are present rather than in what proportion, then $MI$ under an independent
> law increases with $C$ toward, but never reaching, the complete-coverage value.

That is a testable prediction and it is to-do item 2. One honest caveat: the premise
"multiplicity matters less than support" is an assumption, not something measured.
The cheap test is to take one array and redistribute the *same* support over
different multiplicities, at fixed $C$, and see how much $MI$ moves. That isolates
the multiplicity effect without any new sampling machinery.

{There are $2^G-1$ possible combinations of cells, which means that we need roughly
[Ncell] to sample every combination at least once. How do we compute that? what are
the assumptions.}

> **Q:** Ok, I'm not sure to understand what is $E[T]$: the expected diversity? or
> the expected number of cells required to have every combination? How you compute
> it in the first case, and also how do you get to that integral in the second case?

**A. They are two different questions and I should have separated them.**

*Question 1, expected diversity at a given $C$.* Answered at the top of this
document: $\mathbb{E}[U] = \sum_s (1 - (1-q_s)^C)$. Each term is the probability
that type $s$ appears at least once, and expectation is linear so you may sum them
even though the events are dependent.

*Question 2, expected number of cells to see every type.* This is $\mathbb{E}[T]$,
$T$ being the **number of cells drawn until the last missing type appears**. It is
the coupon-collector problem.

*Uniform case.* If all $S = 2^G-1$ types are equally likely, then once $k$ distinct
types are held, the wait for a new one is geometric with success probability
$(S-k)/S$, so its mean is $S/(S-k)$. Summing,

$$\mathbb{E}[T] \;=\; \sum_{k=0}^{S-1}\frac{S}{S-k} \;=\; S\,H_S \;\approx\;
S(\ln S + \gamma), \qquad \gamma \approx 0.5772$$

*Non-uniform case*, which is what you actually have. The geometric argument breaks,
because the wait depends on *which* types are missing, not just how many. The
standard trick is **Poissonisation**. Let cells arrive as a Poisson process of rate
1, so that by time $t$ the number of cells is $\mathrm{Poisson}(t)$ and, crucially,
the arrivals of the different types become **independent** Poisson processes of
rates $q_s$. Then

$$P(\text{type } s \text{ unseen by } t) = e^{-q_s t}
\quad\Longrightarrow\quad
P(T > t) = 1 - \prod_{s}\left(1 - e^{-q_s t}\right)$$

using independence for the product. Finally $\mathbb{E}[T] = \int_0^\infty P(T>t)\,dt$
for a non-negative variable, giving

$$\boxed{\;\mathbb{E}[T] \;=\; \int_0^\infty\left(1 - \prod_{s}\left(1 - e^{-q_s t}\right)\right)dt\;}$$

Because the process has rate 1, expected time equals expected number of cells, which
is what makes the Poissonised answer usable directly. Computed:

| $G$ | types $S$ | uniform $S(\ln S+\gamma)$ | flat size law | size law at mean 1.6 |
|---|---|---|---|---|
| 3 | 7 | 18 | 22 | 21 |
| 4 | 15 | 49 | 66 | 90 |
| 5 | 31 | 124 | 181 | **389** |

*Assumptions*: cells independent and identically distributed, type probabilities
fixed and known, and every type has $q_s>0$ (otherwise $\mathbb{E}[T]=\infty$). That
last one bites: a size law with $\bar g = 1.6$ at $G=5$ makes the 5-gene set nearly
unreachable, which is why 389 is so much worse than 124.

---

# computational scaling

We use a pair of KT entropy for training (which is differentiable), and counting
entropy to measure the resulting MI.

> **Q:** Ok, let us replace $U$ by $J$. How do you compute the actual value or
> expected value of $J$ as a function of $C$? What does exact alphabet mean?

**A.** $J$ and $U$ are the same quantity, the number of distinct cell types, so the
formula is the one derived above:

$$\mathbb{E}[J](C) \;=\; \sum_{s}\left(1 - (1-q_s)^{C}\right), \qquad J \le 2^{G}-1$$

The cap is the single most useful fact in this section: **$J$ saturates and does not
grow with $C$.** Measured, $G=5$, flat law at mean 2.5: $J$ = 8.5, 18.5, 25.2, 29.0,
30.0 at $C$ = 10, 30, 60, 120, 250.

**"Exact alphabet"** is my sloppy phrase for the size of the response alphabet the
*exact* estimator must enumerate. Grouping replaces the labelled response
$Y \in \{0,1\}^C$ by the count vector $K = (K_1,\dots,K_J)$, where $K_j$ is how many
of the $n_j$ cells in group $j$ fired. Since $K_j$ takes $n_j+1$ values,

$$|\mathcal{K}| \;=\; \prod_{j=1}^{J}\left(n_j+1\right)$$

That is the number of joint count states, which the exact estimator enumerates
explicitly and stores a probability for. With $\sum_j n_j = C$ it is maximised when
the groups are equal, giving $\approx (C/J + 1)^J$: polynomial in $C$, but with
exponent $J$ up to $2^G-1$. At $G=5$, $C=30$ it is about $2^{24}$; at $C=120$ about
$2^{64}$. This is why exact enumeration dies early, and it is a *memory and
enumeration* limit, distinct from the statistical one below.

## KT entropy

> **Q:** I prefer you to give formula, and explain the main crux to get. Like the
> mixture idea. Then only the scaling, but I also want you to explain the scalings
> based on the formula. We need to add though, that as the value of the entropy
> increases I will need bigger batch size for the training.

**A. The object.** Over a training batch of $B$ sniffs, the response distribution is
a **mixture**:

$$P(y) \;=\; \frac{1}{B}\sum_{i=1}^{B} P_i(y), \qquad
P_i(y) = \prod_{r} A_{ir}^{\,y_r}(1-A_{ir})^{1-y_r}$$

each component $P_i$ being the product-Bernoulli response to sniff $i$, with
$A_{ir} = p_r(x_i)$. The entropy of a mixture of product distributions has no closed
form, and that is the entire difficulty.

**The crux.** Kolchinsky and Tracey bound the mixture entropy using pairwise
overlaps rather than the intractable sum. With the Bhattacharyya coefficient

$$BC(i,j) \;=\; \prod_{r}\left[\sqrt{A_{ir}A_{jr}} + \sqrt{(1-A_{ir})(1-A_{jr})}\right]$$

the lower bound is

$$\boxed{\;H(Y) \;\ge\; \underbrace{\frac{1}{B}\sum_i \sum_r h(A_{ir})}_{H(Y\mid X)}
\;-\;\frac{1}{B}\sum_i \log_2\!\left(\frac{1}{B}\sum_j BC(i,j)\right)\;}$$

and since the first term is exactly $H(Y\mid X)$, the second term alone is a lower
bound on the mutual information:

$$MI_{\rm KT} \;=\; -\frac{1}{B}\sum_{i}\log_2\!\left(\frac{1}{B}\sum_{j}BC(i,j)\right)$$

This is what `compute_kt_entropy(..., return_mi=True)` returns and what training
maximises. Reading it: $BC(i,j)$ measures how *confusable* sniffs $i$ and $j$ are at
the output. If every pair is perfectly distinguishable the inner sum collapses to
the $j=i$ term and the bound is large; if all sniffs look alike, $BC \to 1$ and the
bound goes to zero. Maximising it means pushing the $B$ conditional response
distributions apart.

Identical cells enter with a multiplicity: a group of $n_j$ identical cells
contributes $BC_j^{\,n_j}$, i.e. $n_j \log BC_j$ in the exponent, which is why the
cost depends on $J$ rather than $C$.

**The ceiling, which answers your batch-size point.** Note $BC(i,i) = \prod_r [A_{ir}
+ (1-A_{ir})] = 1$. So the inner sum is at least $1$, hence
$\tfrac{1}{B}\sum_j BC(i,j) \ge 1/B$, hence

$$MI_{\rm KT} \;\le\; -\log_2(1/B) \;=\; \log_2 B$$

**The objective cannot express more than $\log_2 B$ bits, however good the array
is.** At $B = 4096$ that is 12 bits. So yes: as the information to be captured
grows, the training batch must grow with it, exponentially in bits. A useful rule is
to keep $B \ge 2^{MI+2}$ or so, leaving a couple of bits of headroom. In the current
runs $MI \approx 7$ against a ceiling of 12, which is comfortable, and the
crosscheck confirmed the optimisation loses nothing (0.004 bits).

**Scaling, read off the formula.**

- *Time*: the double sum over $i,j$ with a product over $J$ group columns is
  $O(B^2 J)$. Independent of $C$, because $J \le 2^G-1$.
- *Memory*: the tile computes an $(m, n, J)$ array for chunk edge $m$, so peak is
  $O(m^2 J)$ with chunking, or $O(B^2 J)$ without. With
  `recompute_backward=True` the retained graph drops to $O(BJ)$ and the $O(B^2J)$
  cost becomes compute-only.

## counting entropy

> **Q:** what is $K_j$? Why is it drawn from a binomial distrib? [...] I draw
> ligands, look at the response of the array, if it's a new one I store it as such,
> if it's not new, I add +1 to the histogram? Then I use this histogram to sample
> P(A)? [...] it would take as much memory as enumeration if I want the computation
> to be exact. [...] You wrote that the bias scales as $B/2^{H(K)}$, what is $K$?
> How do you get that result?

**A. What $K_j$ is, and why binomial.** Group $j$ contains $n_j$ cells with
*identical* gene sets, hence identical firing probability $p_j(x)$. Given the sniff,
they fire independently. The number that fire,

$$K_j \;=\; \sum_{c \in \text{group } j} Y_c \;\sim\; \mathrm{Binomial}\big(n_j,\, p_j(x)\big)$$

is a sum of $n_j$ i.i.d. Bernoulli variables, which is the definition of a binomial.
$K = (K_1,\dots,K_J)$ is the count vector, and $H(K)$ is its entropy.

Using $K$ instead of the labelled $Y$ loses nothing: given $K$, which particular
cells fired is uniform over the $\prod_j \binom{n_j}{K_j}$ arrangements *regardless
of $x$*, so that part carries no information and $I(X;K) = I(X;Y)$ exactly. That is
the whole point of grouping.

**A. Your description of the algorithm is exactly right.** Draw a sniff, compute the
array response, sample the output bits, reduce to the count vector $K$, and
increment its tally. Then $\hat P(K) = \text{count}/B$ and
$\hat H(K) = -\sum \hat P \log_2 \hat P$. That is the plug-in estimator.

**A. And your memory worry is the right worry, but it does not bite.** You store
only the symbols you have **actually observed**, not the alphabet. At most $B$
distinct ones, and usually far fewer. The alphabet $\prod_j(n_j+1)$ can be $10^7$
while the table holds $10^4$ rows. Concretely the implementation packs each count
vector into a single `int64` by mixed radix and reduces once, costing 8 bytes per
observation: 128 MiB for $B = 1.7\times10^7$. So the estimator is $O(B)$ in memory
and $O(B\log B)$ in time, never $O(\text{alphabet})$.

The price of not enumerating is that the estimate is **biased**, which is the next
point.

**A. What $K$ is in "$B/2^{H(K)}$", and where the law comes from.** $K$ is the count
vector above and $H(K)$ its true entropy, so $2^{H(K)}$ is the **effective alphabet
size**: the number of symbols that carry appreciable probability, which is far
smaller than $\prod_j(n_j+1)$.

The classical result is that the plug-in entropy is biased downward by approximately
$(m-1)/(2B\ln 2)$ where $m$ is the number of symbols with non-negligible mass. You
cannot apply it directly because $m$ is unknown, and Miller-Madow's substitution of
the *observed* count under-corrects exactly when most of the alphabet is unseen. It
recovered only about a quarter of the measured bias here.

So the law I quoted is **empirical, not a theorem**. It was obtained by measuring
the bias directly: `estimator_crosscheck.py` re-evaluates the same trained
checkpoints with both the exact enumerating estimator (unbiased, available at $C=30$)
and sampled counting, over 5 expression levels and 3 budgets. Fitting those 12
points,

$$\text{bias [bits]} \;\approx\; 3.34\left(\frac{B}{2^{H(K)}}\right)^{-0.92}$$

The useful content is the *ratio*: what matters is budget divided by effective
alphabet, not budget alone. Under 0.1 bits needs a ratio near 46. Treat the
constants as a calibration for this family of arrays, not a universal law.

For a diagnostic that needs no ground truth, use the **Good-Turing missing mass**
$f_1/B$, the fraction of observations that are the only sighting of their symbol.
It estimates the probability mass never sampled, and it is reported as
`grouped_counting_missing_mass`.

## scaling of the computation time

> **Q:** you have mixed up time and memory. [...] you said "this is the memory
> bottleneck of the whole pipeline" which I don't understand, as you were talking
> about computation time a second ago. Then you mention a KT sweep, but it used the
> old evaluation [...] I want to know what is the hard ceiling of measuring.

**A. Fair criticism, I did conflate them. Disentangled:**

**Computation time.** Dominated by the physics forward pass, which is
$O(B \cdot L \cdot R_{\rm pool} \cdot k_{\rm sub})$ per batch: linear in the number
of sniffs, the number of ligands, and the size of the deduplicated receptor pool.
Training adds a backward pass, roughly a factor 3. On top sits the estimator:
$O(B^2 J)$ for KT, $O(B \log B)$ for counting, $O(B \cdot \prod_j (n_j+1))$ for
exact enumeration.

**Memory.** A different quantity with a different bottleneck. The forward pass
materialises an energy tensor of shape $(B, L, R_{\rm pool}, k_{\rm sub})$, which is
what forces chunking and what I meant by "$R_{\rm pool}$ is the bottleneck": it is
the dimension you cannot chunk away, because the whole pool must be reachable by
every cell. $R_{\rm pool}$ grows fast in $g$, as
$\frac{1}{k}\sum_{d\mid k}\varphi(d)g^{k/d}$, giving 1, 8, 51, 208, 629 for
$g=1\ldots5$ at $k_{\rm sub}=5$. The estimators then add: $O(m^2 J)$ for a KT tile,
$O(B)$ for counting, $O(B \cdot |\mathcal{K}|)$ for exact.

**You are also right about the stale 21% figure.** That number came from the
calibration before the counter was fixed and before repeats were removed. It should
be recomputed rather than quoted.

**The hard ceiling of measuring, which is the real question.** There are two
different ceilings and only one of them is about speed.

1. *Computational ceiling*: how long the counting takes. **This one I removed.** It
   used to be quadratic in $B$ (every chunk re-sorted the whole accumulated table);
   it is now $O(B\log B)$ with 8 bytes per observation. 8.4M responses take 2.3
   seconds. A C++ or CUDA rewrite would gain very little, because this is no longer
   the binding constraint.
2. *Statistical ceiling*: how many samples are needed for the estimate to be
   unbiased. $B$ must scale with $2^{H(K)}$, and
   $H(K\mid X) \approx (J/2)\log_2(C/J)$ once $J$ has saturated, so
   $$B \;\sim\; C^{\,J/2}$$
   polynomial in $C$ with exponent $J/2 \approx 15$ at $G=5$. **This ceiling cannot
   be moved by faster code.** It is information, not implementation.

That distinction is the answer to your to-do item 2 confusion below: I fixed the
first and the second is untouched, because it is not fixable.

## scaling of the memory

**A.** Covered above; the three contributions are the energy tensor
$O(B_{\rm chunk} L R_{\rm pool} k_{\rm sub})$, the estimator, and the symbol table
$O(B)$ for counting or $O(|\mathcal{K}|)$ for exact enumeration.

---

# impact of the environment

> **Q:** when you say "the task is 100-way classification plus a concentration
> readout", can we really add up entropy that way? If yes, why? and what is the
> limit to that? Because as we see the concentration channel can add quite a lot of
> entropy. Except that it doesn't really. I don't understand why.

**A. Yes, and it is exact, not an approximation.** It is the **chain rule for mutual
information**. Write the sniff as $X = (\text{identity}, \text{concentration})$.
Then

$$I(X;Y) \;=\; I(\text{identity};Y) \;+\; I(\text{concentration};Y \mid \text{identity})$$

which follows from $I(A,B;Y) = I(A;Y) + I(B;Y|A)$, itself just the chain rule for
entropy rearranged. No independence assumption is needed, and the decomposition is
exact for any joint distribution. Both terms are non-negative.

*What limits each.* The first is at most $H(\text{identity}) = 6.637$ bits, measured,
a hard combinatorial cap since exactly one of 100 ligands is present. The second is
at most $H(Y \mid \text{identity}) \le C$ bits, but in practice far less.

**Why concentration contributes so little, and the honest answer.** Measured, the
total peaks at 7.03 while identity alone caps at 6.64, so concentration contributes
about 0.4 bits. Two candidate reasons, and I have **not** measured which dominates:

- The readout may simply be insensitive: with `cell_readout='mean'` the firing
  probability is the weighted open fraction, and receptor occupancy saturates in
  concentration, so a large dynamic range in $c$ maps to a small range in $p$.
- Or the array is spending its capacity on identity and has little left. At the
  peak, $MI \approx H(\text{identity})$, so it has essentially solved identity, and
  the residual response capacity is what is left over.

This is directly measurable and not yet measured. `identity_channel`,
`concentration_channel`, `mutual_information_ligand` and
`mutual_information_concentration` all exist in the measurement registry and were
simply not requested in these sweeps. Adding them to `final_measurement_fns` on a
re-measurement of saved checkpoints would settle it cheaply. **I would do this before
writing the section**, because "concentration adds only 0.4 bits" is a claim the
paper would have to defend.

> **Q:** I don't understand this sentence: "distance 0.136 within a cloud of radius
> 0.244, while the genes sit 1.6 times further from the centre than that radius. The
> receptors are free to place themselves wherever they like." I understand the
> numbers, but I don't understand how it leads to that conclusion.

**A. It does not, and I overreached. Retract it.**

What the numbers actually support is narrow: the learned gene positions are not
confined inside the ligand cloud, so the optimiser is not being forced to place
receptors in a crowded region. That is all. It says nothing about whether the
*latent dimension* is sufficient to separate 100 ligands, which is the question that
matters, and it is not answerable by comparing two radii.

The real test is an experiment, not a measurement of the existing geometry: lower
`latent_dim` from 6 to 3 and then 2, holding everything else fixed, and see whether
the peak $MI$ falls. If it is flat, the geometry was slack. If it falls, you have
located the constraint. That is cheap and I would run it before claiming anything
about geometry.

## In the receptor cases

The optimal behavior of an array of binary receptors is easy to compute (1 bit per
receptor). In that case, we have designed a type of environment that was "complex
enough" to generate all these bits.

{Is it a matter of comparing the entropy of the environment with the entropy of the
array? However, the entropy of the environment is continuous due to the
concentration channel. How can we compare the entropy of the array that is discrete,
to the entropy of the environment that is continuous?}

> **Q:** So, I like your decomposition: array side / environment side. I like your
> decomposition of the MI with $I(\text{identity};Y) + I(\text{concentration};Y\mid\text{identity})$.
> Now the cap of the first term is clear, but the second term could be also
> bracketed, no?

**A.** Yes. Collecting the brackets on the second term, from loosest to tightest:

$$0 \;\le\; I(\text{conc};Y\mid \text{identity}) \;\le\; H(Y \mid \text{identity})
\;\le\; H(Y) \;\le\; C \ \text{bits}$$

and a tighter, more useful upper bound using what you already measure:

$$I(\text{conc};Y\mid\text{identity}) \;=\; MI_{\rm total} - I(\text{identity};Y)
\;\ge\; MI_{\rm total} - H(\text{identity})$$

which gives a *lower* bound of $7.03 - 6.64 = 0.39$ bits from measured quantities
alone, with no extra runs. Combined:

$$0.39 \ \text{bits} \;\le\; I(\text{conc};Y\mid\text{identity}) \;\le\; 30 \ \text{bits}$$

The lower end is tight and the upper end is useless, which is exactly why measuring
`concentration_channel` directly is worth doing.

The general point, which I think is the one for the paper: **you never need the
differential entropy of the environment.** $MI = H(Y) - H(Y\mid X)$ is a difference
of two *discrete* entropies, finite and well defined for a continuous $X$. The
continuum is automatically truncated by the array's own discreteness, so the
discrete-versus-continuous comparison you were worried about simply does not arise.

Additionally, this is just a lower bound, then we have classifier limits (cover
dimension) on top of the environment's intrinsic entropy.

> **Q:** I think you are WRONG on the cover bound in the cell case. You said "A cell
> is a threshold on a weighted sum of receptor responses followed by a sigmoid", no,
> there is no "followed by a sigmoid" right now it's just a weighted sum. So I don't
> think that the same counting theorem applies with $R_{\rm pool}$. Because most of
> the receptors are melted inside a single cell. But it's fine, I don't expect an
> analytic equivalent statement to Cover, but just mentioning that the same geometric
> issue exists.

**A. You are right and I was wrong.** Checked in `cells.py::CellReadout.finalize`:
with `cell_readout='mean'`, which is what every one of these runs used, the output is

```python
return acc.clamp(0.0, 1.0)
```

a clamped weighted sum. No sigmoid, no threshold, no learned $\theta$. (The sigmoid
exists only in `mode='threshold'`, which these sweeps do not use.) So the cell is not
a classifier and Cover's dichotomy count does not apply to it. My claim was wrong on
the facts.

Your second objection is also right and is the deeper one: even with a threshold, the
relevant dimension would not be $R_{\rm pool}$, because a single cell does not see
the pool as $R_{\rm pool}$ free coordinates. Its weights $w_{cr}$ are fixed by its
gene set through the multinomial stoichiometry, not free parameters, and most
receptors have $w_{cr} = 0$ for any given cell. The free dimension is at most the
number of receptors that cell actually carries, and even those are tied together.

Taking your suggestion, the statement I would put in the notes is qualitative:

> The same geometric limitation as in the receptor case is present, in a weaker
> form. A cell responds to a fixed, non-negative, normalised combination of receptor
> occupancies, each of which is a monotone function of a distance in the latent
> space. The set of ligand subsets a cell can separate is therefore restricted by
> the latent geometry, exactly as for a single receptor, but without a clean
> dichotomy count because the combination weights are not free parameters. We do not
> have an analytic analogue of Cover's bound for the cell case.

On top of this, there are some computational limits to the complexity of the
environment.

## The cell case

I imagine that it should be roughly the same: the entropy of the environment
argument should work. Then, the cover bound should also work, and the computational
limits are the same. Now the 1 gene per cell limit is the receptor case so all good.
But the case with several genes per cell may introduce additional environmental
variation.

**A.** The entropy argument transfers cleanly. The cover bound does not, per the
above. The $g=1$ check is worth keeping and it passes empirically: measured $MI$ at
one gene per cell is 5.37 heteromers against 5.43 homomers, a paired difference of
$-0.06 \pm 0.03$ bits, consistent with zero as it must be, since a one-gene cell can
only build the homopentamer.

---

# To do list

- **Redo the measurement of the trained array of the receptor case for the figure of
  the article.**

  > **Q:** Right, so what's the batch size measurement limit here?

  **A.** For an array of $R$ binary receptors the response alphabet is $2^R$, so
  *exact* enumeration is available whenever $2^R$ fits in memory: $R \le 15$ gives
  32768 states, $R = 20$ gives $10^6$, which is the practical limit at
  `eval_chunk_size` of a few hundred. Use exact and the question of batch size
  disappears, since it is unbiased at any $B$ (only the variance shrinks with $B$).

  If $R$ is too large for that, the counting rule is the same as everywhere else:
  $B \gtrsim 46 \cdot 2^{H(Y)}$ for a bias under 0.1 bits, and $H(Y) \le R$. So
  $R=20$ with $H(Y)\approx 14$ needs $B \approx 7.5\times10^5$, which is cheap. The
  receptor case is far easier than the cell case because there is no grouping, no
  promiscuity, and $H(Y)$ is bounded by $R$ rather than growing with $C$.

- **Test the cell $\to\infty$ prediction.**

  > **Q:** So, I'm confused: you say counting bounds to $C\approx 60$, but previously
  > you said that measurement wasn't really bounded. I think you are thinking of the
  > previous way of measuring. If not, I would like you to think about ways to improve
  > the computation time of the measurement, including writing a C++ or even CUDA
  > subroutine. Then you proposed to validate against exact measurement, no but I have
  > the [Good-]Turing criterion. It's enough.

  **A. Both statements were true and about different things, which I should have
  separated.** See the computation-time section above:

  - The **computational** limit was quadratic and is now $O(B\log B)$. That is the
    one I said "is no longer a constraint", and it is not.
  - The **statistical** limit, $B \sim C^{J/2}$, is untouched and is what bounds you
    near $C \approx 60$ at $G=5$.

  **A C++ or CUDA rewrite would not help**, because the binding constraint is how
  many samples the estimator needs, not how fast each sample is processed. Counting
  8.4M responses already takes 2.3 seconds; the problem is that at $C=120$ you would
  need $10^{12}$ of them.

  The way through is the **KT bound**, whose cost is $O(B^2 J)$ with $J \le 2^G-1$
  and therefore **independent of $C$**. It is the only estimator that survives the
  limit you want to probe. Its cost is a roughly constant offset (measured at 0.79 to
  0.81 bits below exact at two expression levels and two budgets), which cancels in
  comparisons.

  **A. And you are right that the missing mass is enough**, I withdraw the suggestion
  to validate against exact enumeration. $f_1/B$ is a self-contained criterion that
  needs no ground truth, which is precisely why it was worth adding. Use it, and use
  KT beyond where it says counting has failed.

- **Can we assess that the complete coverage strategy is THE BEST strategy?**

  > **Q:** right, let's just keep this one qualitative then.

  **A.** Agreed. The coverage-beats-replication argument is in the gene-sampling
  section above. Add the qualifier that it only has content below saturation.

- **Test on what depends this maximum? And what is the best expression law.**

  **A.** In the current setup the maximum depends on **the environment**, not on the
  array: $G=5$ peaks at 7.034 and $G=8$ at 6.978, the same value, against an identity
  entropy of 6.637, while $G=3$ peaks at 4.851 because it cannot reach the ceiling.
  Until the ceiling is lifted, "the best expression law" cannot be asked, because
  every law that reaches it ties. The two experiments that would separate
  environment-limited from array-limited are mixtures
  (`mu_ligands_per_source: 1e-6 -> 2.0`) and more ligands ($100 \to 1000$).
