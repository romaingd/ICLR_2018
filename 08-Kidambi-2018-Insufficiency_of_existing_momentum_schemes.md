<h1> On the insufficiency of existing momentum schemes for Stochastic
Optimization </h1>

## R. Kidambi, P. Netrapalli, P. Jain, S. M. Kakade



---



## Abstract



---



## I - Introduction

* **First order optimization methods** (FOOM) rely solely on the gradient of the
function (or an unbiased approximation thereof), that is the first order
derivative. They are thus distinct from second order optimization methods
such as Newton's algorithm, which rely on the Hessian of the function.

* The simplest FOOM is **Gradient Descent**, but it is known to be
**suboptimal** for smooth convex problems among others.

* **Fast gradient, or momentum-based methods** achieve optimal convergence
rates:
  * **Heavy ball method** (HB)
  * **Nesterov's accelerated gradient descent** (NAG)

<br>

* Deep learning and large datasets have favored the development of
**momentum Stochastic Gradient Descent** (SGD) algorithms, cheap and efficient.

* However, classical momentum methods require **exact gradients** (on the full
dataset), while they are implemented in practice with **stochastic gradients**
(on a mini-batch).

<br>

<center>

*Are momentum methods optimal even in the <strong> stochastic first
order oracle (SFO) </strong>, where we access stochastic gradients computed on
small-constant-sized mini-batches (including batches of size 1)?*

</center>

<br>

*
