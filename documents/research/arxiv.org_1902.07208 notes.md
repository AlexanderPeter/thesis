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
