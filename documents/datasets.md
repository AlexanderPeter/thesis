# Common datasets

| Name      | #Images    | #Classes | Comments | Links                                       |
| --------- | ---------- | -------- | -------- | ------------------------------------------- |
| CIFAR-10  | 60'000     | 10       |          | https://www.cs.toronto.edu/~kriz/cifar.html |
| CIFAR-100 | 60'000     | 100      |          | https://www.cs.toronto.edu/~kriz/cifar.html |
| ImageNet  | 14'197'122 | 1'000    |          | https://image-net.org/index.php             |
| MS COCO   | 328'000    | 80       |          | https://cocodataset.org/                    |
|           |            |          |          |                                             |

More examples on: https://paperswithcode.com/datasets

# Plant disease datasets

| Name                                               | #Images | #Classes | Comments                                   | Link                                                                             |
| -------------------------------------------------- | ------- | -------- | ------------------------------------------ | -------------------------------------------------------------------------------- |
| PlantVillage (PVD)                                 | 54'303  | 38       | Laboratory setup, many sources             | https://github.com/spMohanty/PlantVillage-Dataset                                |
| PlantDoc                                           | 2'598   | 28       | Real-life, scrapped from internet          | https://github.com/pratikkayal/PlantDoc-Dataset                                  |
| Leaf Images                                        | 4'502   | 2        |                                            | https://data.mendeley.com/datasets/hb74ynkjcn/1                                  |
| Rice leaf disease dataset                          | 120     | 3        |                                            | https://archive.ics.uci.edu/ml/datasets/Rice+Leaf+Diseases                       |
| Plant-Pathology-2020                               | 3'651   | 38       |                                            | https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data                         |
| Plant Pathology 2021                               | 18'600  |          |                                            | https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data              |
| PlantClef                                          | 55'306  |          | No diseases                                | https://www.aicrowd.com/clef_tasks/83/task_dataset_files?challenge_id=1009       |
| Cotton Disease Dataset                             | 2'310   |          |                                            | https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset              |
| Cotton Leaf Disease                                | 1'786   |          |                                            | https://www.kaggle.com/datasets/raaavan/cottonleafinfection                      |
| Northern Leaf Blight (NLB) Lesions                 | 234     | 1        |                                            | https://www.scidb.cn/en/c/p00001                                                 |
| Dataset for Crop Pest and Disease Detection (CCMT) | 24'881  | 22       | more augmented available, many repetitions | https://data.mendeley.com/datasets/bwh3zbpkpv/1                                  |
| Field images of maize                              | 18'000  | 2        |                                            | https://osf.io/p67rz/                                                            |
| IP102: Dataset for Insect Pest Recognition         | 75'222  | 102      | No plants                                  | https://github.com/xpwu95/IP102                                                  |
| Eight common tomato pest images                    | 609     | 8        | Insects                                    | https://data.mendeley.com/datasets/s62zm6djd2/1                                  |
| DARMA                                              | 231'414 | 1'000    | ~50GB                                      | https://drive.google.com/drive/folders/13bOuB7U15CgYMm1vrd0jgtOXFwMlHqXf         |
| PlantDataset                                       | 5'106   | 20       | many repetitions                           | https://www.kaggle.com/datasets/duggudurgesh/plantdataset                        |
| Leaf disease segmentation dataset                  | 588     | 0        | segmentation                               | https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset |
| CVPPP, LSC                                         | 532     | 0        | segmentation                               | https://github.com/lxfhfut/Self-Supervised-Leaf-Segmentation                     |
| New Plant Diseases Dataset                         | 87'000  | 38       | augmented from PlantVillage-Dataset        | https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data        |
| Cassava Leaf Disease Classification                | 21'398  | 5        |                                            | https://www.kaggle.com/competitions/cassava-leaf-disease-classification          |
| Agriculture-Vision Database                        |         |          | Aerial images                              | https://www.agriculture-vision.com/agriculture-vision-2020/dataset               |
| Agriculture-Vision Database                        |         |          | Aerial images                              | https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021          |
| Crop Disease dataset                               |         | 27       | ?                                          |                                                                                  |
| Open Plant Disease Dataset                         |         |          | ?                                          |                                                                                  |
| Plant Disease Detection in Cotton Images           |         |          | ?                                          |                                                                                  |
| Agronomi-Net                                       |         |          | ?                                          |                                                                                  |
| Insects from rice, maize, soybean                  |         |          | ?                                          |                                                                                  |
| Pest and Disease Image Database (PDID)             |         |          | No plants                                  |                                                                                  |
| Plant Disease and Pest Recognition (PDPR)          |         |          | ?                                          |                                                                                  |
| Plant disease diagnosis dataset (PDDD)             | 421'133 | 120      | combination, 8:1:1, 224 × 224              | https://plantpad.samlab.cn/image_down.html                                       |
|                                                    |         |          |                                            |                                                                                  |

Other lists:

- https://www.frontiersin.org/files/Articles/1158933/fpls-14-1158933-HTML-r2/image_m/fpls-14-1158933-t004.jpg
- https://plantmethods.biomedcentral.com/articles/10.1186/s13007-021-00722-9/tables/4
- https://spj.science.org/doi/10.34133/plantphenomics.0054

# Weights

| Dataset      | Model           | Training | Link                                                                                     |
| ------------ | --------------- | -------- | ---------------------------------------------------------------------------------------- |
| PlantVillage | ResNet9         | SL       | https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2/output |
| PDDD         | ResNet34        | SL       | https://zenodo.org/records/7890438                                                       |
| PDDD         | ResNet50        | SL       | https://zenodo.org/records/7890438                                                       |
| PDDD         | ResNet101       | SL       | https://zenodo.org/records/7890438                                                       |
| PDDD         | ViT Base        | SL       | https://zenodo.org/records/7890438                                                       |
| Cannabis     | ResNet18 + UNet | SSL      | https://drive.google.com/drive/folders/1zJBUnGh_A0xd4VZgkzx9ShC1XZ0ALaB6                 |
| CVPPP, LSC   | ResNet18 + UNet | SSL      | https://drive.google.com/drive/folders/1zJBUnGh_A0xd4VZgkzx9ShC1XZ0ALaB6                 |
|              |                 |          |                                                                                          |

# Dermatology datasets

| Name                             | #Images        | #Classes | Comments | Link                                                                                 |
| -------------------------------- | -------------- | -------- | -------- | ------------------------------------------------------------------------------------ |
| Diverse Dermatology Images (DDI) | 656            | 2, 78    |          | https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965 |
| PAD-UFES-20                      | 2'298          | 6        |          | https://data.mendeley.com/datasets/zr7vgbcyr2/1                                      |
| HAM10000                         | 10'015         | 7        |          | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T      |
| Fitzpatrick 17k                  | 16'536, 16'577 | 9        |          | https://github.com/mattgroh/fitzpatrick17k                                           |
|                                  |                |          |          |                                                                                      |
