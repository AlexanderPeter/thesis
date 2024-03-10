# How Useful is Self-Supervised Pretraining for Visual Tasks?

31 Mar 2020

self-supervised models now produce features that are comparable to or outperform those produced by ImageNet pretraining

experiments show: the pretrained model converges to baseline performance before accuracy plateaus with more and more labels

possiblilities:

1. self-supervision achieves a better accuracy than the baseline
2. self-supervision achieves the same accuracy but with fewer labeled examples
3. self-supervision achieves the same accuracy with the same number of labeled examples

number of labeled examples: $n$

accuracy of a model trained from scratch: $a(n)$

accuracy of a finetuned model : $a_{ft}(n)$

balance: $a(\hat{n}) = a_{ft}(n)$

Utility at $n$: $U(n) = \frac{\hat{n}}{n} - 1$

It is the ratio of additional labels to match the same accuracy as the finetuned model

this utility tends to decrease with ample labels

self-supervision is more helpful when applied to larger models and to more difficult versions of the data

visual self-supervision

- autoencoder
- inferring missing parts: inpainting, colorization
- predicting spatial position or applied transformations
- contrastive embeddings

Influencing factors:

- Data: amount of complexity (variation)
- Model
- Self-supervision algorithm

tasks on artificial images

- Object classification
- Object pose estimation
- Semantic segmentation
- Depth estimation

Pretraining Methods

1. Variational autoencoder (VAE)
2. Rotation
3. Contrastive Multiview Coding (CMC)
4. Augmented Multiscale Deep InfoMax (AMDIM)

Models

- ResNet9
- ResNet50

As more labeled data is included, the utility tends toward zero

"This suggests that the utility of self-supervised pretraining comes mainly from better regularization that reduces overfitting, not better optimization that reduces underfitting—otherwise we should expect self-supervision to have non-negligible utility even with large numbers of labeled samples."

"With few labeled samples the performance of the ResNet50 model is worse when trained from scratch, but when pretrained is better than the pretrained ResNet9 suggesting the importance of pretraining large models when working with less data."

Different numbers of layer frozen: Performance suffers as more of the model is frozen.
