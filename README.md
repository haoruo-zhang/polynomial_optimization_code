# polynomial_optimization_code
## Optimization Problem (B.1)

Minimize:

$$
\min_{\mu \in \mathbb{R}^{(d+1) \times D \times L}} \sum_{n \in \text{supp}(p)} p_n \phi_n(\mu),
$$

subject to

$$
\mathcal{M}_d(\mu_i^{(l)}) = R_i^{(l)} (R_i^{(l)})^T,
$$

$$
\mathcal{M}_{d-1}(\mu_i^{(l)}; 1 - x_i^2) = S_i^{(l)} (S_i^{(l)})^T, \quad i = 1, \dots, D, \ l = 1, \dots, L,
$$

$$
R_i^{(l)} \in \mathbb{R}^{(d+1) \times (d+1)}, \quad S_i^{(l)} \in \mathbb{R}^{d \times d},
$$

$$
\mu_{i,0}^{(l)} \geq 0,
$$

$$
\mu_{i,0}^{(l)} = 1, \quad i = 2, \dots, D,
$$

$$
\phi_n(\mu) = \sum_{l=1}^{L} \prod_{i=1}^{D} \mu_{i,n_i}^{(l)}.
$$


