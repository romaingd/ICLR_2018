# Training and inference with integers in Deep Neural Networks

## s. Wu, G. Li, F. Chen, L. Shi



---



## Abstract



---



## I - Introduction

* Deep Neural Networks (DNN), although widely used, need **energy-intensive
devices (CPU, GPU, TPU, ...) during training** to achieve a good performance
in reasonable time.

* DNN also typically consist in millions of parameters (138M for VGG-16),
and are therefore **very heavy to store**, notwithstanding the increased
overfitting opportunities.

* Therefore much attention has been brought to **reducing the size of the
networks, as well as deploying dedicated hardware.**

<br>

* **Training typically requires higher precision (gradient descent).** Most
papers hence study the compression of standard networks trained with full
precision.

* In this paper :
  * **How to process both training and inference of DNNs with low-bitwidth
    integers**
    * How to quantize all operands and operations
    * How many bits are needed for SGD computation and accumulation
  * **WAGE framework** (constrains Weights, Activations, Gradients, Errors)



---



## II - Related work

* **Focus on reducing precision of operands and operations in both training
and inference.**

* Orthogonal complexity reduction techniques like network compression, pruning,
compact architectures.

<br>

#### Weights and activations

* BNN : binary weights and activations, but still real-valued variables for
gradients accumulation.

* XNOR-Net : filter-wise scaling factor for weights, efficient implementation
of convolutions.

* TWN, TTQ : ternary-valued weights with two symmetric thresholds.

<br>

#### Gradient computation and accumulation

* DoReFa-Net : quantized gradients to low-bitwidth floats with discrete states
in the backward pass.

* TernGrad : ternary-valued gradients to reduce the overhead of gradient
synchronization in distributed training.

* However both works use float32 weights during training, and ignore the
quantization of batch normalization.



---



## III - WAGE quantization

* Notation : errors $e^i = \frac{\partial L}{\partial a^i}$,
gradients $g^i = \frac{\partial L}{\partial W^i}$. MAC stands for
multiply-accumulate ($x + y \times z$).

* Main idea : **quantize weights $W$ and activations $a$ during inference,
errors $e$ and gradients $g$ during backpropagation.**

<br>

<center>

![WAGE](pictures/04-WAGE.png)

</center>

<br>

#### Forward propagation

* Weights are stored with $k_G$-bit integers.

* Find a **good quantization function $Q_W(\cdot)$ that maps higher precision
weights to their $k_W$-bit reflections.**

* Although weights are accumulated with high precision, the deployment of
reflections in dedicated hardware is much more efficient after training.

* Quantize activations to compensate for the increased bitwidth caused by MACs.

<br>

#### Backward propagation

* Quantize gradients and errors to compensate for the increased bitwidth caused
by MACs.


<br>


### III.1 Shift-based linear mapping and stochastic rounding

* Notation : $\sigma(k) = 2^{1-k}, k \in \mathbb{N}_+$

* Quantization function :
**$Q(x,k) = Clip \{\ \sigma(k) \cdot round[\frac{x}{\sigma(k)}],\
-1+\sigma(k),\ 1-\sigma(k)\}$**

* To avoid permanent saturation, use a scaling factor
**$Shift(x) = 2^{round(\log_2 x)}$**

<br>

<center>

![WAGE](pictures/04-WAGE_quantization.png)

</center>

<br>


### III.3 Quantization details

* Use a **layer-wise shift based scaling factor** to attenuate the amplification
effect (variances of the weights are scaled by quantization, which would make
the network's outputs explode).

* Avoid average pooling, given that they need more precision

* **Batch normalization is replaced by the scaling factor mentioned** (assumes
batches have zero-mean).

<br>

* It seems that **it is the orientations rather than the orders of magnitude
in errors that guide previous layers to converge**; hence we scale the errors
distribution into a fixed interval.

* Since only relative values of errors are preserved after shifting, gradients
are shifted accordingly.

* There only remains directions for weights to change, and the **step sizes are
integer multiples of minimum step $\sigma$.**

<br>

* **Drop memory-consuming techniques** such as momentum, L2 regularization
(quantization and randomness added amount for empirically sufficient
regularization), avoid softmax and cross-entropy (use Sum of Squared Error).
