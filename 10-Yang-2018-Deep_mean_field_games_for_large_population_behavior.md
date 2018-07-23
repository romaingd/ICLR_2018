<h1>Learning deep mean field games for modeling
large population behavior</h1>

## J. Yang, X. Ye, R. Trivedi, H. Xu, H. Zha



---



## Abstract



---



## I - Introduction

* There is significant interest for modeling macroscopic population behavior,
first of all in physics, but also in social contexts, such as social media.
One of the major intuitions in such situations is that individual behavior
may be optimal with respect to some objective, yielding population dynamics.
It is also very likely that the current population state, when observable,
influences the individual actions (e.g. discussing already popular topics).

* This motivates the development of a **model of population behavior learnable
from real data** and satisfying some criteria:
  1. The model captures the **dependency between population distribution and
  their actions**.
  2. **Individual behaviors are modeled as optimal for some implicit reward**.
  3. The model enables **prediction of future population distribution**, given
  measurements at previous times.

<br>

* Those criteria are addressed using a **mean field game (MFG)** approach.
Stemming from game theory and economics, MFG consider the limit of $N$-player
games when $N$ tends to infinity.

* In MFG, the agent population is represented via their distribution over a
state space, and the individual reward depends on the population distribution
and aggregate actions.

* MFG are characterized by stochastic differential equations, and can be used
to model a wide range of phenomena (Mexican waves [Ola], where people put their
towel on the beach, etc).

<br>

* The problem is linked to **Markov Decision Processes (MDP)** and
**Reinforcement Learning (RL)** by framing it as **MFG model inference via
MDP policy optimization**. In other words, the MFG model is used to describe
the natural behavior of the population and is fitted by solving an associated
MDP, without imposing any control on the system.

* This is in some way the best of both worlds:
  * The MFG framework allows the tractable adaptation of **inverse reinforcement
  learning (IRL)** methods, to learn complex reward functions able to explain
  the behavior of arbitrarily large populations.
  * RL inspires a data-driven method to solve an MFG model of a real-world
  system for temporal prediction.

<br>

* In this paper:
  * **Data-driven fitting of an MFG model along with its reward function**
  * Reduction of a discrete time graph-state MFG (derived from the general case)
  to an MDP, which means that solving the MDP (policy and reward) is equivalent
  to inference of the MFG.



---



## II - Related work

* Prior work made strong assumptions on the cost function in order to attain
analytical solutions, hence the importance of **learning the cost function
directly from data**.

* Earlier work imposed Gaussian assumptions on the distributions, or relied
on numerical finite-difference methods for solving continuous MFG, limiting the
applications to **toy models** given the inherent computational challenges.

* In RL, multi-agent generalizations of the MDP are not easily used to represent
the interaction of thousands of agents, the trajectory of each being optimal,
hence the importance of the **macroscopic point of view** of MFG.

* In unknown MDP estimation and IRL, he **maximum entropy IRL framework**
has proved successful at learning reward functions from expert demonstrations.
It can be augmented by the use of deep neural networks to learn complex
reward functions.

* The presented method is an extension of **Guided Cost Learning** (Finn et al.,
  2016) to the problem of **learning a reward function under which a large
population's behavior is optimal**.



---



## III - Mean Field Games

### III.1 - Mean Field Games on graphs
