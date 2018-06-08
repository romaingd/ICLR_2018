# Spherical CNNs

## T. S. Cohen, M. Geiger, J. Köhler, M. Welling



---



## Abstract



---



## I - Introduction

* CNNs are able to detect local patterns regardless of their position in the
image, that is **regardless of translations** (by translation equivariance).

* We would like to similarly have a **spherical CNN** that would detect local
patterns **regardless of 3D rotations.**

* However, there is **no good way to use translational convolution
(or cross-correlation) to analyze spherical signals** (spherical rotations
cannot be emulated by translation of the signal's planar projection)

<br>

* Major difference :
  * **The plane acts on itself *via* the isomorphism to translations**, which
  form a subgroup of the symmetry group.
  * The sphere cannot act on itself (it is not a group), hence we must consider
  **the whole of SO(3), which is not isomorphic to the sphere.**

<br>

* Hence the result of a spherical correlation (the output feature map) is to be
considered a **signal on SO(3), not on the sphere $S^2$.** A spherical CNN
would involve **SO(3) group correlation in deeper layers.**

<br>

* In this paper :
  * **Theory of spherical CNNs**, including
  non-commutative harmonic analysis to define $S^2$ and SO(3) correlations
  (first use of cross-correlations on a continuous group inside a deep neural
  network)
  * **Efficient implementation using generalized Fast Fourier Transform**
  * Empirical support for the utility of spherical CNNs for rotation-invariant
  learning problems.



---



## II - Related work

* The power of CNNs stems in large part from their ability to exploit
translational symmetries (using weight sharing, translational equivariance).

* Natural **generalization to larger groups of symmetries.**

* Previous work :
  * Discrete groups (except SO(2)-steerable networks)
  * Analysis of spherical images, but architecture not equivariant

<br>

* In comparison, this work achieves equivariance to a continuous,
non-commutative group (SO(3)), and uses generalized Fourier transform for
fast group correlation.



---



## III - Correlation on the sphere and rotation group

* **Planar correlation** - The value of the output feature map at translation
$x \in \mathbb{Z}^2$ is computed as an inner product between the input feature
map and a filter, shifted by $x$.

* **Spherical correlation** - The value of the output feature map evaluated at
rotation $R \in$ SO(3) is computed as an inner product between the input feature
map and a filter, rotated by $R$.

* Noting, for $R \in$ SO(3), $L_R$ the rotation operator such that
$[L_R f](x) = f(R^{-1}x)$, we define the spherical correlation for spherical
signals $f$ and $\psi$ (over K channels) as :

<strong>

\[  
  [\psi * f] = \left< L_R \psi, f \right>
    = \int_{S^2} \sum_{i=1}^K \psi_k (R^{-1}x) f_k(x) dx
\]

</strong>

<br>

* **Rotation group correlation** - Similarly, naturally extending the rotation
operator to signals on SO(3), we define the rotation group correlation for
signals $f$ and $\psi$ on SO(3) as :

<strong>

\[  
  [\psi * f] = \left< L_R \psi, f \right>
    = \int_{\text{SO(3)}} \sum_{i=1}^K \psi_k (R^{-1}Q) f_k(Q) dQ
\]

</strong>

<br>

* **Equivariance** - A layer $\Phi$ is equivariant if
$\Phi \circ L_R = T_R \circ \Phi$ for some operator $T_R$. We show that
planar, spherical and rotation group correlations are equivariant.



---



## IV - Fast spherical correlation with G-FFT

* Correlations and convolutions can be computed efficiently using the
**Fast Fourier Transform** in $O(n \log n)$, faster than the naive $O(n^2)$
implementation.

* For functions on the sphere and rotation group, there is an analogous
transform, the **generalized Fourier Transform**, and a corresponding fast
algorithm **G-FFT**.

* 
