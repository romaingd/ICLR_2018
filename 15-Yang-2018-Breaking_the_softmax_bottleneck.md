# Breaking the softmax bottleneck: <br> A high-rank RNN language model

## z. Yang, Z. Dai, R. Salakhutdinov, W.W. Cohen



---



## Abstract



---



## I - Introduction

* Despite some recent and spectacular progress (from N-grams to neural
networks), **statistical language modeling** is still mostly based on the
following universal factorization: given a corpus of tokens $X =
(X_1, ...,X_T)$, the **joint probability factorizes as $P(x) =
\prod_t P(X_t|C_t)$** where $C_t = X_{<t}$ is the **context**. Different
approaches model $P(X_t|C_t)$ differently.

* **Recurrent Neural Networks (RNN)** achieve impressive results on various
tasks. RNNs typically work as follows (based on the factorization):
  1. Encode the context into a fixed size vector
  2. Compute the **logits** as the dot product between the encoding and the
     word embeddings
  3.
