# On the convergence of ADAM and beyond

## S. Reddi, S. Kale, S. Kumar


---


## Abstract

ADAM has become completely dominant in deep learning over the past few years.
This stochastic optimization algorithm is a variant of SGD, in the same vein
as ADAGRAD, insofar that it adjusts the learning rate on a per-feature basis
based on past gradients. However, ADAM specifically uses an exponential moving
average of past gradients, limiting the reliance to the past few gradients.

The authors show that this sort-term memory makes the convergence proof of ADAM
flawed. Intuitively, informative gradients that happen rarely are killed out
too fast by the short-term memory. The authors provide a simple counterexample
where ADAM converges to the worst solution possible.

Although it is of high importance, this result doesn't change the fact that
ADAM tends to work very well in deep learning, where data distribution
is heuristically quite smooth. It seems that the variant provided by the authors
(AMSGRAD) doesn't translate into much better performance.


---


## I - Introduction

* **SGD (Stochastic Gradient Descent)** : Update the parameters of a model by
a step in the direction of the negative gradient of the loss evaluated
on a *minibatch*.

* Due to the large amount of data involved, SGD is the dominant method to
train deep neural networks (DNN) today.


<br>


* **ADAGRAD (Adapative Gradient)** : Variant of SGD that scales coordinates
of the gradient by square roots of some form of *averaging of past values*.

* SGD variants in ADAGRAD's vein are popular and successful because they
**adjust the learning rate on a per-feature basis**.


<br>


* ADAGRAD's performance falls in non-convex and dense settings: using *all* past
gradients makes the learning rate decay too quick, and early updates lose
too much impact.

* **ADAM (Adaptive Moment estimation)** : ADAGRAD idea + mitigate the rapid
learning rate decay using an exponential moving average of squared past
gradients (limit reliance to only the *past few gradients*)

* This is also the idea of RMSPROP, ADADELTA, NADAM, ...

* Empirically, **ADAM fails to converge when important information happens
rarely, because it is forgotten too fast**
("*when some minibatches provide large
gradients but only quite rarely; although they are quite informative, their
influence dies out rather quickly due to exponential moving average*")


<br>


* In this paper :
  * **Counterexample of a convex optimization problem where ADAM
  does not converge**
  * Localization of the **error in ADAM's convergence proof**
  * ADAM variants with **long-term gradient memory**



---



## II - Preliminaries

### Optimization setup

* Online optimization problem in the full information feedback setting
(minimize regret)

* At each time step $t$ :
  * Pick a point $x_t \in \mathcal{F}$
    **(parameters of the model, e.g. weights)**
  * Gain access to a loss function $f_t$
    **(loss of the model on the next minibatch)**
  * Incur loss $f_t(x_t)$

<br>

* **SGD** : $x_{t+1} = \Pi_{\mathcal{F}} (x_t - \alpha_t g_t)$ with
$g_t = \nabla f_t(x_t)$
(gradient descent projected onto the set of feasible points)


<br>


### Generic adaptive methods

<br>Access

<center>

![Generic Adaptive Method Setup](pictures/01-generic_adaptive.png)

**Generic framework for adaptive algorithms** </center>

<br>

* Additional notes :
  * $\alpha_t$ step size
  * $\alpha_t V_t^{-1/2}$ learning rate
  * Restriction to $V_t = \text{diag}(v_t)$
  * Decreasing step size is required for convergence

<br>

#### SGD

<center>

![SGD phi and psi](pictures/01-SGD.png)

**SGD - Specifications in the framework** </center>

<br>

* Aggressive learning rate decay $\alpha / \sqrt{t}$

<br>

#### ADAGRAD

<center>

![ADAGRAD phi and psi](pictures/01-ADAGRAD.png)

**ADAGRAD - Specifications in the framework** </center>

<br>

* Modest learning rate decay $\alpha / \sqrt{\sum_i g_{i,j}^2}$

<br>

#### ADAM

<center>

![ADAM phi and psi](pictures/01-ADAM.png)

**ADAM - Specifications in the framework (main formulation)** </center>

<center>

![ADAM moment formulation](pictures/01-ADAM_moment.png)

**ADAM - Moment update formulation (alternative formulation)** </center>
<br>

* ADAGRAD with momentum (exponential moving average)

* The momentum term with $\beta_1 > 0$ significantly boosts the performance,
especially in deep learning.



---



## III - The non-convergence of ADAM

* **The proof of convergence of ADAM (given in Kingma & Ba, 2015) is wrong.**

<center>

![ADAM error](pictures/01-ADAM_error.png)

**This quantity should be semi-definite positive.** <br>
**It is for SGD and ADAGRAD,
but not for ADAM.** </center>
<br>

* The error essentially lies in the fact that the learning rate does not
always decay.

<br>

<center>

![ADAM counterexample](pictures/01-ADAM_counterexample.png)

**Simple counterexample where ADAM converges to the worst solution.** <br>
<strong> $\mathcal{F} = [-1,1]$, $\ \ C>2$, $\ \ \beta_1=0$,
$\ \ \beta_2=1/(1+C^2)$ </strong>
</center>
<br>

* **There is an online convex optimization problem where ADAM fails.**

* One could think that adding a small constant $\epsilon$ to $v_t$ would solve
the problem. Actually, although it helps, **for any $\epsilon > 0$, there is
an online convex optimization problem where ADAM fails.**

* One could think that using a large $\beta_2$ would solve the problem.
Actually, although it helps, **for any $\beta_1, \beta_2 \in [0,1)$ such that
$\beta_1 < \sqrt{\beta_2}$, there exists an online convex optimization problem
where ADAM fails.** (Note that this condition is typically satisfied with
suggested parameters)

* One could think that this problem is specific to online optimization problems,
and that stochastic optimization problems would not be affected by this issue.
Actually, although stochastic optimization is typically easier, **for any
$\beta_1, \beta_2 \in [0,1)$ such that $\beta_1 < \sqrt{\beta_2}$,
there exists an online convex optimization problem where ADAM fails.**

<br>

* This means one has to use "problem-dependent" update hyperparameters ;
in high-dimensional settings, this typically means using different
hyperparameters for each dimension, which defeats the purpose of adaptive
algorithms.



---



## IV - A new exponential moving average variant : AMSGRAD

<center>

![AMSGRAD](pictures/01-AMSGRAD.png)

</center>

<br>

* The key difference between AMSGRAD and ADAM is that the former maintains
the maximum of all previously seen $v_t$, and uses it to normalize the running
average of the gradient instead of $v_t$ (ADAM).

* In other words, **to ensure a non-increasing learning rate, AMSGRAD uses
the maximum value of the normalization term, instead of its current value.**

* Proof of convergence is provided, with a data-dependent ensured regret of
**$O(\sqrt{T})$**, which can be considerably **better than $O(\sqrt{dT})$ of
SGD.**
