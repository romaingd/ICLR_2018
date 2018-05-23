# On the convergence of ADAM and beyond

## S. Reddi, S. Kale, S. Kumar


---


## I - Introduction

* **SGD (Stochastic Gradient Descent)** : Update the parameters of a model by
moving them in the direction of
the negative gradient of the loss evaluated on a minibatch

* SGD is the dominant method to train deep networks (DNN) today


<br>


* **ADAGRAD (Adapative Gradient)** : Variant of SGD that scales coordnates
of the gradient by square roots of some form of averaging

* SGD variants in ADAGRAD's vein
have found success in **adjusting the learning rate on a per-feature basis**


<br>


* ADAGRAD's performance falls in nonconvex and dense settings, since the
entirety of past gradients are used in the updates.

* **ADAM (Adaptive Moment estimation)** : ADAGRAD idea where the rapid learning
rate decay is mitigated using the exponential moving average of squared past
gradients (limit reliance to only the past few gradients)

* This is also the idea of RMSPROP, ADADELTA, NADAM, ...

* Empirically, **ADAM fails to converge when important information happens rarely, because it is
forgotten too fast**
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
  * Access a loss function $f_t$
    **(loss of the model on the next minibatch)**
  * Incur loss $f_t(x_t)$

<br>

* **SGD** : $x_{t+1} = \Pi_{\mathcal{F}} (x_t - \alpha_t g_t)$ with
$g_t = \nabla f_t(x_t)$
(gradient descent projected onto the set of feasible points)


<br>


### Generic adaptive methods

<br>

![Generic Adaptive Method Setup](pictures/01-generic_adaptive.png)
<center> **Generic framework for adaptive algorithms** </center>

<br>

* Additional notes :
  * $\alpha_t$ step size
  * $\alpha_t V_t^{-1/2}$ learning rate
  * Restriction to $V_t = \text{diag}(v_t)$
  * Decreasing step size required for convergence

<br>

#### SGD

<center>
![SGD phi and psi](pictures/01-SGD.png)
**SGD : Specifications in the framework** </center>

<br>

* Aggressive learning rate decay $\alpha / \sqrt(t)$

<br>

#### ADAGRAD
<center>
![ADAGRAD phi and psi](pictures/01-ADAGRAD.png)
**ADAGRAD : Specifications in the framework** </center>

<br>

* Modest learning rate decay $\alpha / \sqrt{\sum_i g_{i,j}^2}$

<br>

#### ADAM

<center>
![ADAM phi and psi](pictures/01-ADAM.png)
**ADAM : Specifications in the framework (main formulation)** </center>

<center>
![ADAM moment formulation](pictures/01-ADAM_moment.png)
**ADAM : Moment update formulation (alternative formulation)** </center>
<br>

* ADAGRAD with momentum (exponential moving average)

<br>
