<h1> Asking the right questions: <br>
Active question reformulation with reinforcement learning </h1>

<h2> C. Buck, J. Bulian, M. Ciaramita, W. Gajewski,
A. Gesmundo, N. Houlsby, W. Hang </h2>



---



## Abstract



---



## I - Introduction

* In the face of complex information needs, humans have the ability to
**reformulate in order to ask the right question.**

* Reproduce this behavior with an agent **AQA (active question answering)**
that sits the user and a backend QA system (environment).

<br>

* AQA aims to maximize the chance of getting the correct answer by sending a
reformulated question to the environment.

* Two components of the AQA:
  * Sequence-to-sequence model trained with **Reinforcement learning** using a
  reward based on the answer returned by the environment (black-box probing
  using only question strings)
  * Convolutional neural network to combine the evidences returned by the
  environment to **select an answer**

<br>

* AQA is a machine-machine communication instance, **able to learn non-trivial
reformulations** such as stemming, tf-idf, term re-weighting.



---



## II - Related work

* Previous work:
  * Paraphrasing to augment the training of a semantic parser
  * Pivoting through auxiliary languages (in MT)

* In contrast, this method a **direct neural paraphrasing system that generates
full question reformulations** while optimizing directly end-to-end.

<br>

* Policy gradient methods allow sequence-level reward functions and prevent
**exposure bias** (in word-level training, the model is trained on ground
truth pre-sequences, while it is tested on its own generated pre-sequences,
hence a discrepancy between training and test sets).

<br>

* Previous work:
  * Query reduction networks with policy gradient
  * Improvement of document retrieval by reformulating queries with added
  terms from documents retrieved from a search engine for the original query



---



## III - Active Question Answering model

<br>

<center>

![AQA](pictures/07-AQA.png)
**AQA agent-environment setup**
</center>

<br>

* An episode starts with an original question $q_0$. The agent then **generates
a set of reformulations** $\{q_i\}_{i=1..N}$. The environment returns answers
$\{a_i\}_{i=1..N}$. The selection model then **picks the best candidate**.


<br>


### III.1 - Question-answering evironment

* **BiDAF (BiDirectional Attention Flow)** - extractive QA system (selects an
answer from contiguous spans of a given document) (Seo et al., 2017)

* Quality metric chosen is the **token-level F1 score**.


<br>


### III.2 - Reformulation model

* Sequence-to-sequence model, trained on multilingual translation, then adapted
to monolingual paraphrasing (little high quality monolingual training data).


<br>


### III.3 - Answer selection model

* Selects the best answer by predicting the difference of the F1 score to the
average F1 score of all variants.

* Pre-trained embeddings for the tokens of query, rewrite and answer are each
passed through a 1-dimensional CNN and max-pooling, then concatenated
and passed through a feed-forward network which produces the output.
