# Spectral normalization for $G$enerative Adversarial Networks

## T. Miyato, T. Kataoka, M. Koyama, Y. Yoshida



---



## Abstract



---



## I - Introduction

* **Generative Adversarial Networks (GANs)** have proven tremendously successful
recently in **unsupervised generative modeling** (learning to mimic a target
distribution). They roughly work as follows: train a generator $G$ and a
discriminator $D$, consecutively with one step for each in turn,
where $D$ tries to discriminate real from fake data, and $G$
tries to fool $D$ by producing realistic fake data; after training, drop $D$
and keep only the generator $G$.

* GANs are able to impressively **learn highly structured probability
distributions**, but the discriminator is interesting in its own right
(estimating a density ratio between target and generated distributions).

<br>

* However, GANs are notoriously **hard to train** and sensitive
**mode collapse** - in high dimensional spaces, the density ratio
learned by the discriminator is often inaccurate and very unstable, and the
generator resorts to a single mode of the true distribution, failing to cover
the multimodal structure (and producing only a very small family of samples).

* Moreover, when the supports of the target and generated distributions are
disjoint, a **perfect discriminator is reached**, and density ratio yields no
useful gradient: training stops. This motivates the restriction to smaller
classes of discriminators, a.k.a. regularization.

<br>

* In this paper:
  * **Spectral normalization**, a weight normalization method that can stabilize
  the training of the discriminator (requiring little implementation, tuning and
  computational cost)
  * Explanation of the effectiveness of Spectral Normalization (SN) against
  other regularization techniques.



---



## II - Method

* A neural discriminator, with input $x$, takes the form $D(x, \theta) =
\mathcal{A}(f(x, \theta))$ with $\theta$ the parameters to learn. **The standard
GAN formulation is given by:**

<strong>

\[
  \min_G \max_D V(G,D) \\

  \text{with}\ \ \ \ \ \  V(G,D) = \mathbb{E}_{x \sim q_{data}}[\log{D(x)}] +
                           \mathbb{E}_{x' \sim p_G}[1 - \log{D(x')}]
\]

</strong>

<br>

* Recent work stresses the impact of the allowed discriminator class on the
GAN performance, and **encourage the restriction to Lipschitz continuous
discriminators.** For example it is known that, for a fixed $G$, the standard
formulation leads to an optimal discriminator:

\[
  D^{* }_G(x) = \frac{q_{data}(x)}{q_{data}(x) + p_G(x)}
               = \text{sigmoid}(\log{{q_{data}(x)} - \log{p_G(x)}})
               = \text{sigmoid}(f^{* }(x)) \\
               \ \\
               \\
               \\
  \text{and its derivative} \ \ \ \ \
  \nabla_x f^{* }(x) = \frac{1}{q_{data}(x)} \nabla_x q_{data}(x) -
                       \frac{1}{p_G(x)} \nabla_x p_G(x) \\

  \text{can be unbounded or even uncomputable}
\]

<br>

* Some regularization would help tackling this issue. We therefore look for
the discriminator $D = \text{arg } \max_{{||f||}_{Lip} \leq K} V(G,D)$.


<br>


## II.1 - Spectral normalization

* Input-based regularizations are easily computed, but cannot impose
regularization outside the supports of the generated and target distributions
without some heuristics.

* Noting the **spectral norm $\sigma(A) = \max_{{||h||}_2 \leq 1}$** (equivalent
to the largest singular value), and given a linear layer $g(h) = Wh$, we have
${||g||}_{Lip} = \sigma(W)$. The composition rule for $L$ layers
therefore gives:

\[
  {||f||}_{Lip} \leq \product_{l=1}^{L+1} \sigma(W^l)
\]

* **Spectral normalization controls the Lipschitz constant of the discriminator
function $f$ by constraining the spectral norm of each layer to be equal to 1**,
effectively ensuring that the Lipschitz constant of $f$ is bounded above by 1:

<strong>

\[
  \bar{W}_{SN}(W) := W / \sigma(W)
\]

</strong>


<br>


## II.2 - Fast approximation of the spectral norm $\sigma(W)$

* It turns out that we can use the power iteration method to estimate the
spectral norm with very small additional computational time.


<br>


## II.3 - Gradient analysis of the spectrally normalized weights

* The derivation of the gradient (see formula in the paper)
