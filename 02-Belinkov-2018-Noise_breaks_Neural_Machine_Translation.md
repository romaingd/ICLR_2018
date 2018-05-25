# Synthetic and natural noise both break Neural Machine Translation

## Y. Belinkov, Y. Bisk


---


## Abstract


---


## I - Introduction

* Human language processing is surprisingly robust, easily overcoming typos,
misspellings, omission of letters, etc. A famous example is the 'Cambridge meme'
below.

<br>

<center>

![Cambridge meme](pictures/02-Cambridge_meme.png)

**Cambridge meme** </center>

<br>

* In contrast, **Neural Machine Translation (NMT) systems are very brittle**
when faced to such noise, despite their careful training and
their widespread use.

<br>

<center>

![Cambridge meme Google](pictures/02-Cambridge_meme_Google.png)

**Google Translation of Cambridge meme (German -> English)** </center>

<br>

* NMT systems are rarely explicitly trained to address noise, and mainly
rely on the hope that the relevant noise will naturally occur frequently
in the training data.

* It is important to move to **character-based NMT**, or other sub-word units.
It helps with out-of-vocabulary words, reduces the computational cost of
large word embedding matrices.

* Character-based NMT models are able to extract stem and morphological
information to generalize to unseen situations.

* Unfortunately, **training on clean data makes models brittle.** Even small
amounts of noise lead to substantial drops in performance.

<br>

* In this paper :
  * **Investigation of the effects of noise on NMT**
  * Strategies for increasing model robustness:
    * **Using structure-invariant representation**
    * **Robust training on noisy data** (adversarial training)
  * **Character CNN trained on an ensemble of noise types** robust to all kinds
  of noise



---



## II - Adversarial examples

* **Adversarial attacks** are becoming common in computer vision, and
amazingly efficient in changing the output of a model without altering the human
perception of the input.

* Adversarial attacks can be categorized into **white box (access to the model
parameters) or black box attacks**.

* Adversarial attacks have begun receiving increasing attention from the NLP
community.

<br>

* **Adversarial training (including adversarial examples in the training data)**
can improve a model's ability to cope with such examples at test time.

* Robustness is further increased by **adversarial ensemble training.**



---



## III - MT systems

* **End-to-end** models in NMT are getting more and more thoroughly studied,
especially their ability to learn various linguistic properties. The use of
**sub-word units** is an important component of these systems.

* This paper studies 3 different NMT systems with **access to character
information at different levels** :
  * `char2char` - sequence-to-sequence model with attention, trained on
  characters to characters; complex encoder with convolutional, highway, and
  recurrent layers; standard recurrent decoder.
  * `Nematus` - sequence-to-sequence model operating on sub-word units using
  Byte-Pair Encoding
  * ` charCNN` - sequence-to-sequence model with a word representation based
  on a character convolutional neural network; 2 LSTM layers in the encoder
  and decoder; a CNN over characters in each word replaces the word embeddings
  in the encoder



---



## IV - Data

### IV.2 - Noise : natural and artificial

#### IV.2.1 - Natural noise

* **Naturally occurring errors** (e.g. typos, misspellings) are harvested from
available corpora of edits to build a look-up table of possible lexical
replacements.

* Every word in the base corpus is then replaced with an error if one has
been registered.

<br>

#### IV.2.2 - Synthetic noise

* **Swap** : `noise -> nosie`

* **Middle Random** : `noise -> nisoe`

* **Fully Random** : `noise -> iones`

* **Keyboard Typo** : `noise -> noide`



---



## V - Failures to translate noisy texts

<br>

<center>

![BLEU Vanilla](pictures/02-BLEU_Vanilla.png)

**BLEU scores on different test sets when trained on clean (Vanilla) texts**
</center>

<br>


<br>

<center>

![Cambridge Vanilla](pictures/02-Cambridge_Vanilla.png)

**Translation of the Cambridge meme when trained on Vanilla texts** </center>

<br>

* The more noise in the text, the worse the translation quality, with random
scrambling producing the lowest BLEU scores.

* Correcting the input with **spell checkers** roughly adds +5 to the BLEU
scores in both French and German, but actually worsens the result in Czech.



---



## VI - Dealing with noise

### VI.1 - Structure invariant representations

* All three NMT models presented are **sensitive to word structure** (either
because of convolutional layers, or BPE sub-word units) and **character order**,
hence very sensitive to character scrambling noise.

* `meanChar` - generate a word representation by averaging character embeddings,
then proceed with a word-level encoder similar to `charCNN`.

* By construction, **`meanChar` is insensitive to scrambling**, although still
sensitive to typos and natural noise.

<br>

<center>

![meanChar](pictures/02-meanChar.png)
 </center>

<br>


### VI.2 - Black-box adversarial training

* The model is presented with adversarial examples that are generated without
direct access to the model. The original training set is replaced with a noisy
training set, one for each type of noise.

<br>

<center>

![charCNN](pictures/02-charCNN.png)
 </center>

<br>

* The robust training is sensitive to the kind of noise. Among the scrambling
methods, more noise helps in training.

* **The three classes of noise are not mutually beneficial.** In particular,
only models trained on natural noise can reasonably translate natural noise
at test time.

* This indicates an **important difference between computational models and
human performance**, since humans are able to decipher random letter orderings
without any explicit training of this form.

<br>

* **Training on mixed noise datasets makes models robust to the specific types
of noise they were trained on.** In particular, the model trained on all
types of noise achieves the best average result, despite not being the best
on any one kind of noise.

<br>

<center>

![All noises training charCNN](pictures/02-charCNN_all_noises_training.png)
**Translation of the Cambridge meme by a charCNN model trained on all types of
noise**
 </center>

<br>



---



## VII - Analysis

### VII.1 - Learning multiple kinds of noise in charCNN