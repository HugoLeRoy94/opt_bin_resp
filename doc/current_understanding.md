


# receptors array

# cells array

The response of an array of cells is non-deterministic. If $Y \in \mathbb{R}^C$ is the response of a cell for a given input $X \in \mathbb{R}^L$ (the concentration of the $L$ ligands in a sniff, 0 if it's not there). We write: 
$$
MI(Y;X) = H(Y) - H(Y | X)
$$
## Cell diversity
Given $G$ possible genes, there are C(G,g) ways to make a cell with $g$ genes. In a system composed of $C$ cells, there is at most:
$$
 J = \min(C,\sum_{g=1}^G C(G,g)) = \min(C,\,2^{G}-1)
$$
unique genetic pattern, called cell diversity.
In practice, the average number of unique pattern expressed in a set of $C$ cells is given by:
$$\boxed{\;\mathbb{E}[J] \;=\; \sum_{s}\left(1 - (1-q_s)^{C}\right)\;}$$
where $q_s$ is the probability of expressing a given genetic pattern $s\in{0,1}^G$

For instance, for genes drawn uniformly, with a size law $\pi$.Writing $q_s$ the probability that a cell lands on a set of genes $s$:

$$q_s \;=\; \frac{\pi(|s|)}{\binom{G}{|s|}}$$

Cell diversity always increases the entropy of response $H(Y)$ at minimum, and by 0 if the cell is redundant.

Now, looking at the conditional entropy, given a sniff, cell triggers independantly. which means that the entropy of the array is the sum of the entropy of individual cell:

$$H(Y \mid X) \;=\; \mathbb{E}_x\left[\sum_{c=1}^{C} h\big(p_c(x)\big)\right],
\qquad h(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

As a result, $H(Y\mid X)$ is a sum of $h$ evaluated at each cell's probability. It therefore depends only on the *multiset* of values $\{p_1(x),\dots,p_C(x)\}$, not on which cell has which, nor on how many distinct types there are.

## Receptor diversity

Increasing $g$ increases the cell diversity the same way in the homomers and heteromers case. So cell diversity alone cannot explain why heteromers are better than homomers. So we need to account for receptor diversity, to understand this raise.

Looking at the data, we see that with increasing average genes expressed $g$, H(Y) grows in both heteromers and homomers. However, it also increases individual cell entropy: $h_c$ faster for the homomers than the heteromer case. So the mechanism through which heteromers are better than homomers is by making the response of individual cell less "mild".


With the `mean` readout the drive is a convex combination
$S_c = \sum_r w_{cr}\,p_r$ of the tuning curves of the receptors the cell carries. {A convex combination is never sharper than the sharpest curve entering it.}
[can we demonstrate this ? Is the choice of a mean vs threshold or any other mechanism so important ? Or, what should be the lowest mathematical requirement for this statement ( on the blurring of the homomers by adding multiple receptors) to be true.]

**receptor diversity buys a slower growth of $H(Y\mid X)$, not a faster growth of $H(Y)$.**
[Can we demonstrate this, or illustrate it with a toy model.]

## Genes sampling

### complete covrage

We started with a "complete coverage" sampling which enumerate (within the number of available cells) all the genetic combination possible. Given a cumulative coverage up to $m$, it takes all the combinations up to $m$ genes.
This leads to an average number of genes expressed :
$$\bar g(m) \;=\; \frac{\displaystyle\sum_{k=1}^{m} k\binom{G}{k}}
{\displaystyle\sum_{k=1}^{m}\binom{G}{k}}$$
Where,
$$\bar g(G) \lim_{G \rightarrow \infty} G/2$$
Intuitively, this could be the best way to generate diversity at fixed number of cells. The logic is that for $C$ fixed cells and a target mean $\bar g$. Every choice trades off two things:

- **Coverage.** A type that is absent contributes nothing, and whatever it could
  have discriminated is lost. No amount of replication elsewhere recovers it,
  because the missing discrimination is not a function of the types you kept.
- **Replication.** A type present $n$ times yields a count $K \sim
  \mathrm{Binomial}(n, p)$ rather than a single bit. That is worth something, it
  averages the output noise, but it is subject to diminishing returns: {the
  information about $p$ in $n$ Bernoulli draws grows like $\tfrac{1}{2}\log_2 n$,
  not like $n$.}[How do you compute that ?]

So the marginal value of the $n$-th copy of a type falls off logarithmically, while the marginal value of a *first* copy of a missing type is finite and does not fall off at all. A strategy that spends cells on duplicates before covering the support is therefore paying a logarithmic return where a constant one was available.
[TO BE TESTED ! with the experiment : The cheap test is to take one array and redistribute the *same* support over different multiplicities, at fixed $C$, and see how much $MI$ moves. That isolates the multiplicity effect without any new sampling machinery.]

The biological drawback to this method is that cells need to talk to each other to achieve that, which seems unrealistic.

### independant cell samplings

More realistic cell sampling would be that each cell draw a genes independant from one another. We can either draw genes directly according to a given probability. Or use a size probability distribution $\pi$ for $g$ the number of genes expressed in a cell. We start with 2 comparisons :
1) uniform genes distribution. 2) uniform genes distribution with exponential distribution of $g$.
For the case 1), we can compute the average number of genes expressed in a cell (given that we reject drawing where 0 genes are sampled in a cell):
$$\mathbb{E}[g \mid g \ge 1] \;=\; \frac{Gp}{1-(1-p)^{G}}$$
since $P(g\ge 1) = 1-(1-p)^G$ and the numerator is unchanged because the $g=0$ term contributes nothing to the sum.

Let us describe an array by the histogram of its cell types. Let $\mathcal{S} = \{s : 2^G-1 \text{ non-empty gene sets}\}$ and let $n_s \ge 0$ be the
number of cells carrying set $s$, with $\sum_s n_s = C$.

- **Support**: $\mathrm{supp} = \{s : n_s > 0\}$, which types are present at all.
- **Multiplicity profile**: the normalised histogram $\{n_s / C\}$, in what
  proportion.

Complete coverage at level $m$ has $\mathrm{supp} = \{s : |s| \le m\}$ and a flat profile on it. An independent law with type probabilities $q_s$ has, by the law of large numbers, $n_s/C \to q_s$, so its profile converges to $q$, which is flat only if $q$ is.

Under an independent law the realised support converges to $\{s : q_s > 0\}$ as $C \to \infty$, while the multiplicity profile converges to $q$ rather than to the flat profile of complete coverage. If the information is carried mainly by which types are present rather than in what proportion, then $MI$ under an independent law increases with $C$ toward, but never reaching, the complete-coverage value.

There are $2^G-1$ possible combination of cells. But for a given number of cells $C$, there are in average $\mathbb{E}[J] = \sum_s (1 - (1-q_s)^C)$ different cells. On the other hand, if all $S = 2^G-1$ types are equally likely, then once $k$ distinct types are held, the wait for a new one is geometric with success probability $(S-k)/S$, so its mean is $S/(S-k)$. Summing,

$$\mathbb{E}[T] \;=\; \sum_{k=0}^{S-1}\frac{S}{S-k} \;=\; S\,H_S \;\approx\;
S(\ln S + \gamma), \qquad \gamma \approx 0.5772$$
Average number of cells required to achieve complete coverage.

*Non-uniform case*, 
The standard trick is **Poissonisation**. Let cells arrive as a Poisson process of rate 1, so that by time $t$ the number of cells is $\mathrm{Poisson}(t)$ and, crucially, the arrivals of the different types become **independent** Poisson processes of rates $q_s$. Then

$$P(\text{type } s \text{ unseen by } t) = e^{-q_s t}
\quad\Longrightarrow\quad
P(T > t) = 1 - \prod_{s}\left(1 - e^{-q_s t}\right)$$

using independence for the product. Finally $\mathbb{E}[T] = \int_0^\infty P(T>t)\,dt$ for a non-negative variable, giving

$$\boxed{\;\mathbb{E}[T] \;=\; \int_0^\infty\left(1 - \prod_{s}\left(1 - e^{-q_s t}\right)\right)dt\;}$$

Because the process has rate 1, expected time equals expected number of cells.

# computational scaling

We use a pair of KT entropy for training (which is differentiable), and counting entropy to measure the resulting MI.


## KT entropy
to be completed
[Ok, I don't like your way of presenting it. I prefer you to give formula, and explain the main cruks/crucs/kruks to get. Like the mixture idea. Then only the scaling, but I also want you to explain the scalings based on the formula We need to add though, that as the value of the entropy increase I will need bigger batch size for the training.]

## counting entropy
to be completed
[what is $K_j$ ? Why is it drawn from a binomial distrib ? The first sentence was actually what I want you to explain. I draw ligands, look at the response of the array, if it's a new one I store it as such, if it's not new, I add +1 to the histogram ? Then I use this histogram to sample P(A) ? and then compute entropy from there ? No, that seems silly, because it would take as much memory as enumeration if I want the computation to be exact. Can you explain it more. You wrote that the biais scales as B/2^H(K), what is K ? How do you get that result ?]

## scaling of the computation time.

[ I don't understnad in your explanation, you have mixed-up time and memory... I would like to disentangle the two in the explanation. Now I understand that computation time is linear in the number of receptors. Then you said "this is the memory bottleneck of the whole pipeline" which I don't understand ,as you were talking about computation time a second ago. Then you mention a KT sweep, but it used the old evalution which were taking 21% of the computation time, whereas it barely cost anything now. It's barely anything now, but if we jump to larger value of C, keep training with KT and evaluation with counting, then, it might become bigger. I want o know what is the hard ceiling of measuring.]

## scaling of the memory.


# impact of the environment

[So when you say "So the task is 100-way classification plus a concentration readout" can we really add up entropy that way ? If yes, why ? and what is the limit to that ? Because as we see the concentration channel can add quite a lot of entropy. Except that it doesn't really. I don't understand why.]
[I don't understand this sentence : "distance 0.136 within a cloud of radius 0.244, while the genes sit 1.6 times further from the centre than that radius. The receptors are free to place themselves wherever they like." I mean, I understand the numbers, what they mean, but I don't understand how it leads to that conclusion.]

## In the receptor cases
The optimal behavior of an array of binary receptor is easy to compute (1 bit per receptors). In that case, we have designed a type of environment that was "complex enough" to generate all these bits. 
{Is it a matter of comparing the entropy of the environment, with the entropy of the array ? However, the entropy of the environment is continuous due to the concentration channel. How can we compare the entropy of the array that is discrete, to the entropy of the environment jthat is continous ?}
[So, I like you decomposition : array side / environment side I like you decomposition of the MI with I(identity;Y) + I(concentration;Y|identity). Now the cap of the first term is clear, but the second term could be also bracketed, no ?]
Additionally, this is just a lower bound, then we have classifier limit (type cover dimension) on top of the environment intrinsict entropy. I already have some notes about this aspect on the receptor case.
[I think you are WRONG on the cover bound in the cell case. You said "A **cell** is a threshold on a *weighted sum of receptor responses*, $S_{bc} = \sum_r w_{cr} p_{br}$ followed by a sigmoid.", no, there is no "followed by a sigmoid" right now it's just a weighted sum. So I don't think that the same counting theorem applies With Rpool. Because most of the receptors are melted inside a single cell. But it's fine, I don't expect an analytic equivalent statement than cover, but just mentionning that the same geometric issue exits. ]

On top of this, there are some computational limits to the complexity of the environment.

## The cell case

I Imagine that it should be roughly the same: the entropy of the environment argument should work. Then, the cover bound should also work, and the computational limit are the same. Now the 1 genes per cell limit is the receptor case so all good. But the case with several genes per cell may introduce additional environmental variation.

# To do list
- Redo the measurement of the trained array of the receptor case for the figure of the article. [Right, so what's the batch size measurement limit here ?]
- test the cell->\infty prediction. [So, I'm confused you say counting bounds to C\approx 60, but previously you said that measurement wasn't really bounded, I think you are thinking of the previous way of measuring. If not, I would like you to think about way to improve the computation time of the measurement. Including writing a c++ or even cuda subroutine. Then you proposed to validate against exact measurement, no but I have the turing-correct (or idk the name already f1/[i don't mremember the ratio]) criteria. It's enough.]
- can we assess that the complete coverage strategy is THE BEST strategy ? [right, let's just keep this one qualitative then.]
- Test on what depends this maximum ? And what is the best expression law.
