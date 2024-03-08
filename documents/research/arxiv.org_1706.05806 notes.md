# SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability

8 Nov 2017

detect over-parameterization

input dataset: $X = \{ x_1, ..., x_m \}$

specific neuron: $i$

specific layer: $l$

output vector: $z_i^l = (z_i^l(x_1), ..., z_i^l(x_m))$

$l_1 = \{z_1^{l_1}, ..., z_{m_1}^{l_1} \}$

$l_2 = \{z_1^{l_2}, ..., z_{m_2}^{l_2} \}$

singular value decomposition (SVD) to get 99%

Calculate Canonical Correlation similarity of $l_1'$ and $l_2'$

fourier transformation for large convolutional networks

$\rho_i$ describes how well the layers align

$\overline{\rho} = \frac{1}{\min(m_1, m_2)} \sum_i \rho_i$

graphics shows bottom up learning

"freeze training"
