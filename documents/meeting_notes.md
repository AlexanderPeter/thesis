The following sections include selected notes from multiple meetings with the supervisors. These statements have been summed up, rephrased and sometimes translated.

### Weekly meeting 11.03.2024

**Q: SVCCA was used to compare the layers of a model before and after training. In another setting the layers of a single model were compared with each other. What is the reason for this comparison?**

**A:** Comparing Layers with SVCCA is an approach to get insights on what layers change the most during learning. Frozen evaluation is a more common approach, to compare the first layers and to verify the possibility that some pretrained weights do not contain useful features, but are just lucky.

**Q: What dermatology datasets are suited for this work? Are there any public ones or is it possible to use one from the University Hospital Basel?**

**A:** The dermatology datasets, pretrained weights and tasks can be provided when needed. It makes sense to start with the plant disease datasets and some common models like ResNet50 and Tiny-ViT. These two models have a similar number of parameters.

| Weights           | Model                  | Plant disease tasks | Dermatology tasks |
| ----------------- | ---------------------- | ------------------- | ----------------- |
| Random            | ResNet50 <br> Tiny-ViT | ❹                   | ❹                 |
| ImageNet (SL)     | ResNet50 <br> Tiny-ViT | ❸                   | ❸                 |
| Plant (SSL / SL)  | ResNet50 <br> Tiny-ViT | ❶                   | ❷                 |
| Dermatology (SSL) | ResNet50 <br> Tiny-ViT | ❷                   | ❶                 |

It is recommended to use about 4 tasks for plant diseases and dermatology each, differing in number of classes, proximity(close up photo with neutral background vs. full plant picture with natural background), etc.
For the tasks it is recommended to use mainly classification rather than segmentation. Segmentation would give an unfair advantage for some model architectures.

**Q: ?**

**A:**
