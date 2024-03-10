# How Transferable are Self-supervised Features in Medical Image Classification Tasks?

30 Nov 2021

self-supervised ImageNet models

- SimCLR
- SwAV
- DINO

Kappa score

Approach: Dynamic Visual Meta-Embedding (DVME)

transfer learning from supervised pretraining on a large labeled dataset such as ImageNet is widely studied (Raghu et al., 2019; Ke et al., 2021)

DINO consistently outperforms other self-supervised techniques

Model Genesis:
autoencoder reconstructing images from four transformations

Sowrirajan et al. (2021)

- self-supervised on CheXpert
- finetunes on Shenzhen Hospital X-ray dataset

"Transfer learning with ImageNet pretrained features still incites debates over its actual benefits for downstream medical tasks (Raghu et al., 2019; Ke et al., 2021; He et al., 2019). In a large data regime, Raghu et al. (2019) show that lightweight models with random initialization can perform on par with large architectures pretrained on ImageNet such as ResNet-50 (He et al., 2016) and Inception-v3 (Szegedy et al., 2015). On the contrary, Ke et al. (2021) argue that ImageNet pretraining can significantly boost the performance with newer architectures such as DenseNet (Huang et al., 2017) and EfficientNet (Tan and Le, 2019)."

contrastive learning

- SimCLR, SwAV

momentum encoder

- BYOL, DINO

"In comparison to ImageNet supervised pretrained features, we observe that self-supervised features improve the performance across all downstream tasks. "

t-SNE visualization of features
