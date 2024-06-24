# What Makes Transfer Learning Work For Medical Images: Feature Reuse & Other Factors

09 Jun 2022

"The lack of large public datasets has led to the widespread adoption of transfer
learning from IMAGENET \[10\] to improve performance on
medical tasks [27, 28, 39]."

the benefits from
transfer learning increase with:
• reduced data size
• smaller distance between the source and target
• models with fewer inductive biases
• models with more capacity, to a lesser extent

"...a large dataset, distant
from IMAGENET. In this scenario, transfer learning yields
only marginal benefits which can largely be attributed to the
weight statistics"

"Frechet Inception Distance (FID) \[17\] between IMAGENET and the datasets listed above to measure
similarity to the source domain...Although it may not be a perfect measure [6, 26], it gives a
reasonable indication of relative distances between datasets."

"re-initialization robustness"

"ViTs appear to benefit far more from feature reuse than CNNs."
