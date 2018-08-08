# Breaking the softmax bottleneck: <br> A high-rank RNN language model

## z. Yang, Z. Dai, R. Salakhutdinov, W.W. Cohen



---



## Abstract



---



## I - Introduction

* **Statistical language modeling** has greatly improved over the past decade,
from N-grams to neural networks. Despite their diversity, most models still rely
on **an *autoregressive* factorization of the joint probability** as follows:
given a corpus $X = (X_1, ..., X_T)$, we have **$P(X) = \prod_t P(X_t | C_t)$**
where **$C_t = X_{<t}$ is the *context***. Each approach then models
$P(X_t|C_t)$ differently.

* **Recurrent Neural Networks (RNNs)** use this factorization. They typically
work as follows, at a given step:
  1. Encode the context $C_t$ into a fixed size vector
  2. Compute the **logits** as the dot products between the context and each
     word embedding
  3. Feed the logits to the **softmax** function to produce a categorical
     probability distribution over the words.

<br>

* It remains unclear **whether the combination of dot product and softmax is
powerful and expressive enough** to model the conditional probabilities
$P(X_t | C_t)$, which heavily depend on $C_t$ and can dramatically vary with any
change in context.

<br>

* In this work:
  * Learning a standard softmax-based recurrent language model is equivalent
    to solving a **high-rank matrix factorization** problem.
  * **Softmax bottleneck** - the expressiveness of such models with output word
    embeddings is too low to correctly model natural language.
  * **Mixture of Softmaxes** do not show limited expressiveness, and lead to
    substantial performance improvements.



---



## II - Language modeling as matrix factorization

* With autoregressive factorization, language modeling can be reduced to
modeling $P(X|c)$. Consider thus a natural language as $\mathcal{L} =
\{(c_1, P^{* }(X|c_1)), ..., (c_N, P^{* }(X|c_N))\}$, and $M$ possible tokens.

<br>

<center>

**The objective of a language model is to learn a model distribution
$P_\theta(X|C)$ parametrized by $\theta$ to match the true data distribution
$P^* (X|C)$.**

</center>


<br>


### II.1 - Softmax

* Most parametric language models use **context vectors (hidden states) $h_c$
and word embeddings $w_x$** to compute **logits $h_c \cdot w_x$** and model the
distribution with a **softmax** function:

<strong>

\[
  P_\theta (x|c) = \frac{\exp(h_c \cdot w_x)}{\sum_{x'} \exp(h_c \cdot w_{x'})}
\]

</strong>

<br>

* Consider three matrices:
  * $H_\theta \in \mathbb{R}^{N\times d}$ with
    **${[H_\theta]}_{i, j} = {[h_{c_i}]}_j$**, each row being a context vector
    (typically implemented as a RNN)
  * $W_\theta \in \mathbb{R}^{M\times d}$ with
    **${[W_\theta]}_{i, j} = {[w_{x_i}]}_j$**, each row being an embedded word
  * $A \in \mathbb{R}^{N\times M}$ with
    **${[A]}_{i,j} = {\log P(x_j | c_i)}$**, each row being the probability
    distribution over the tokens given a context.

<br>

* Define furthermore the set **$F(A)$ of row-wise shifts of $A$**, i.e. the set
of matrices obtained by adding an arbitrary real number to each row of $A$
independently. It turns out that:
  1. $A' \in F(A) \iff \text{Softmax}(A') = P^{* }$, i.e. $F(A)$ is the set of
     all possible logits that yield $P^{* }$
  2. All matrices in $F(A)$ have similar ranks, with a maximum
     rank difference of 1.

<br>
