The following sections include selected notes from multiple meetings with the supervisors. These statements have been summed up, rephrased and sometimes translated.

### Meeting 11.03.2024

**Q: SVCCA was used to compare the layers of a model before and after training. In another setting the layers of a single model were compared with each other. What is the reason for this comparison?**

**A:** Comparing Layers with SVCCA is an approach to get insights on what layers change the most during learning. Frozen evaluation is a more common approach, to compare the first layers and to verify the possibility that some pretrained weights do not contain useful features, but are just lucky.

**Q: What dermatology datasets are suited for this work? Are there any public ones or is it possible to use one from the University Hospital Basel?**

**A:** The dermatology datasets, pretrained weights and tasks can be provided when needed. It makes sense to start with the plant disease datasets and some common models like ResNet50 and Tiny-ViT. These two models have a similar number of parameters.
It is recommended to use about 4 tasks for plant diseases and dermatology each, differing in number of classes, natural/lab-controlled background, etc.
For the tasks it is recommended to use mainly classification rather than segmentation. Segmentation would give an unfair advantage for some model architectures.

**Q: PDDD does not provide weights for a tiny ViT model which would have a similar size as the ResNet50 model. Would the weights of the tiny Swin be a suitable substitute?**

**A:** The pretrained dermatology models from the University Hospital Basel are also tiny ViT models. Still, for this work the base ViT would be a better choice than the Swin model.

### Meeting 18.03.2024

**Q: What is the difference between adding a linear layer to the frozen model to train and using logistic regression on the features calculated by the frozen model?**

**A:** Logistic regression leads to a global minimum, which is not granted with a linear layer. The approach with the linear layer uses batches, which introduces randomness into the system. KNN uses the whole embedding and therefore leads to a global minimum as well.

### Meeting 30.04.2024

**Q: Which dermatology tasks are suited, but still distinct from the models of the University Hospital Basel?**

**A:** The provided datasets are DDI, Fitzpatrick17k, HAM10000 and PAD-UFES-20. These sets were already used in previous work e.g. the MICCAI paper.

### Meeting 13.05.2024

**Q: Does your example diagram use the average, minimal and maximal values or the median?**

**A:** It uses the median and the standard error as interval.

### Meeting 13.05.2024

**Q: Some of the logistic regression tasks do not fully converge even after 10'000 iterations. Is it worth to increase the maximal amount of iteration even further?**

**A:** No, 10'000 iterations are already sufficient to cover most tasks.

### Meeting 27.05.2024

**Q: The results with 1000 repetitions are very similar to the ones with 100 repetitions. Is is okay to use only 100 repetitions for all datasets?**

**A:** Yes, in this case 100 repetitions are enough.

### Meeting 24.06.2024

**Q: The default ImageNet weights for the tiny ViT model from the timm-library is the AugReg (pretrained on the 21k ImageNet and finetuned on the 1k ImageNet). Why should I use the Winkawaks from the transformer-library, which has slightly different architecture?**

**A:** The Winkawaks is trained only with the 1k ImageNet like the provided SSL weigths and therefore better suited for the comparison.

### Meeting 15.07.2024

**Q: Some of my dermatology results greatly differ from the ones in your previous work. Could this be caused by a different split or are there other major differences in our procedures?**

**A:** The following points are recommended to adjust:

- The macro-f1-score should be used instead of the weighted f1-score, except for PAD-UFES-20, where the binary-f1-score is correct.
- The last four layers of the ViT should be concatenated.
- The KNN from scikit uses the euclidean distance as default. It is better to explicitly use the cosine metric.
- If there are not enough samples per class it is better to skip the whole evaluation instead of allowing repeated sampling.

### Meeting 30.09.2024

**Q: The task desription also mentions segmentation models. The given datasets are for classifications only. Should I omit segmentation or try out datasets like the ISIC 2018 task 1?**

**A:** Segmentation is not absolutely necessary. It is okay to omit segmentation and only focus on the classification tasks.
