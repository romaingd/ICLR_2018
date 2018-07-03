<h1>Certifying some distributional robustness with principled
adversarial training</h1>

## A. Sinha, H. Namkoong, J. Duchi



---



## Abstract



---



## I - Introduction

* Classical supervised learning framework:
  * Minimize an expected loss $\mathbb{E}_{P_0}[l(\theta ; Z)]$
  * over a parameter $\theta \in \Theta$
  * where $Z \sim P_0$, $P_0$ probability distribution over the space
  $\mathcal{Z}$, $l$ loss function

<br>

* **Robustness to changes in the data-generating distribution $P_0$ is often
highly desirable**, be they naturally occurring (e.g. covariate shift) or
adversarial attacks.

* Neural networks are vulnerable to adversarial attacks (e.g. can lead to
misclassification). Defense mechanisms are at early stages only, the classes of
attacks against which they are effective are not well known, and the formal
verification of a network is generally NP-hard.

<br>

### In this paper

* Efficient procedures for **distributionally robust optimization** with
rigorous guarantees for small to moderate amounts of robustness.

* For a class $\mathcal{P}$ of probability distributions around $P_0$:
<strong>
\[
\min_{\theta \in \Theta} \sup_{P \in \mathcal{P}} \mathbb{E}_P[l(\theta ; Z)]
 \ \ \ \ \ \ \ \ (1)
\]
</strong>

* The authors propose:
  * Robustness sets $P$ with computationally efficient relaxations
  * An adversarial training procedure with nice convergence guarantees,
  that generalizes defense to the test dataset

<br>

### Approach overview

* We consider an adversarial cost $c(z,z_0)$ of perturbing $z_0$ to $z$, and a
robustness region $\mathcal{P} = \{P : W_c(P,P_0) \leq \rho \}$ as a Wasserstein
ball around $P_0$.

* Problem (1) is generally intractable for deep networks and arbitrary $\rho$.
We consider a **Lagrangian relaxation** of the problem:
\[
\min_{\theta} \{F(\theta) :=
  \sup_P{}\{\mathbb{E}_P[l(\theta ; Z)] - \gamma W_c(P,P_0)\}\}
  \ \ \ \ \ \ \ \ (2a)  
\]

* It happens that this relaxation can be reformulated as:
<strong>
\[
\min_{\theta} \{F(\theta) :=
  \mathbb{E}_{P_0} [\phi_\gamma(\theta ; Z)] \}
  \ \ \ \ \ \ \ \ (2b)  \\
  \text{where } \phi_\gamma(\theta ; z_0) :=
  \sup_z \{l(\theta ; z) - \gamma c(z,z_0)\}
\]
</strong>

<br>

* For smooth losses $l$, and large enough penalty $\gamma$ (i.e. small enough
robustness $\rho$), $\phi_\gamma$ (**robust surrogate of $l$ that allows
adversarial perturbations with a budget constrained by $\gamma$**) computation
is simply strongly concave optimization, which is easy, providing nice
convergence guarantees.

<br>

### Related work

#### Robust optimization and adversarial training

* Standard robust optimization approach: minimize $\sup_{u \in \mathcal{U}}
l(\theta;z+u)$ for some uncertainty set $\mathcal{U}$. This is intractable
except for specially structured losses.

* Heuristic perturbations of the training data (e.g. fast gradient sign method)
yield models with unclear convergence and adversarial coverage.

<br>

#### Distributionally robust optimization

* The choice of $\mathcal{P}$ yields a **trade-off between richness of the
uncertainty and tractability of the computation**.

* Previous approaches:
  * parametric classes
  * $f$-divergence balls (fixed support)
  * Wasserstein balls (varying support) (studied in tractable but limited cases)



---



## II - Proposed approach

* Main idea: assume $l(\theta; \cdot)$ is *smooth*, i.e. $\nabla_z
l(\theta;\cdot)$ is $L$-Lipschitz for some $L$. Then a Taylor expansion shows
that **$(l(\theta;\cdot) - \gamma c(\cdot, z_0))$ is $(\gamma - L)$-strongly
concave**.

* Hence, for moderate enough robustness $\rho$ requirements (i.e. by duality
large enough penalty $\gamma$), $\phi_\gamma(\theta ; z_0) =
\sup_z \{l(\theta ; z) - \gamma c(z,z_0)\}$ is easy to compute. That is, **with
essentially no additional computational cost, we can train on the worst possible
perturbation instead of the raw training data**.

* The Wasserstein metric is a natural way to compare two probability
distributions, especially when one is derived from the other by small,
non-uniform perturbations. Viewing a probability distribution as a unit
amount of 'dirt' piled on $\mathcal{Z}$, the Wasserstein distance is
intuitively the minimal cost of turning one pile into the other
(amount moved times the distance it is moved). The authors show that using a
Wasserstein ball yields the equivalence between $(2. a)$ and $(2. b)$.


<br>


### II.1 - Optimizing the robust loss by Stochastic Gradient Descent
