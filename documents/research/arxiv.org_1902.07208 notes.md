# Transfusion: Understanding Transfer Learning for Medical Imaging

29 Oct 2019

ImageNet de-facto method

feature reuse vs over-parametrization

medical imaging

- region of interest is small
- small variations
- high image resolution
- not as much images
- fewer classes

model size as confounder

over-parameterized: number of parameters is suited for 1000 classes

models

- ResNet50
- Inception-v3
- CBR (convolution, batch normalization, relu)

random vs pretrained weights

Retina results (AUC): similar results

cheXpert results (AUC): similar results

large models with small datasets:
gap is a bit higher with the largest of the three models

CCA similarity scores

(SV)CCA = (Singular Vector) Canonical Correlation Analysis

Faster Convergence with pretrained weights

Reusing a subset of the pretrained weight

- Faster Convergence
- less steps to reach 91% AUC
- transfering lower layers has a higher impact than upper layers

Mean Var init

1. Calculating $\tilde{\mu}$ and $\tilde{\sigma}^2$ from pretrained weights
2. Init model with weights using $\mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)$
3. Results are between transfer and random init

Hybrid approaches

- weights of 2 first blocks combined with slimmed remainder
- synthetic Gabor filters combined with random
