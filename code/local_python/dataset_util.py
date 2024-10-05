import pandas as pd
import torch.utils.data as data
from torchvision import datasets, transforms


class DataframeImageDataset(data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        filepath_column="image",
        label_columns=["label"],
        label_map=None,
        class_counts=None,
        loader=datasets.folder.pil_loader,
    ):
        for column in [filepath_column, *label_columns]:
            assert column in df.columns, f"Dataframe does not include column: {column}"
        self.df = df
        self.transform = transform
        self.filepath_column = filepath_column
        self.label_columns = label_columns
        self.loader = loader
        self.label_map = label_map
        self.class_counts = class_counts

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = self.df[self.filepath_column].iloc[index]
        labels = self.df[self.label_columns].iloc[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)
        if self.label_map:
            return image, *[self.label_map[label] for label in labels]
        else:
            return image, *labels

    def get_label_map(self):
        return self.label_map

    def get_class_counts(self):
        return self.class_counts


def create_dataloader(
    data_dir=None,
    df=None,
    batch_size=64,
    img_size=224,
    normalise_mean=(0.485, 0.456, 0.406),  # ImageNet
    normalise_std=(0.229, 0.224, 0.225),  # ImageNet
    filepath_column="filepath",
    label_columns=["target_code", "set"],
    shuffle=False,
    label_map=None,
    class_counts=None,
):
    assert data_dir is not None or df is not None, f"data_dir or df is required"
    if df is None:
        df = pd.read_csv(data_dir)
    transform = transforms.Compose(
        [
            # NOTE: ResNet50_Weights.IMAGENET1K_V1 also uses these values to resize and crop
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(normalise_mean, normalise_std),
        ]
    )

    # NOTE: DataframeImageDataset uses pil_loader as default, which executes Image.convert("RGB") implicitly
    ds_full = DataframeImageDataset(
        df,
        filepath_column=filepath_column,
        label_columns=label_columns,
        transform=transform,
        label_map=label_map,
        class_counts=class_counts,
    )

    dl_full = data.DataLoader(
        ds_full,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return dl_full


def create_dataloaders(
    dataset_path,
    set_names=["train", "valid"],
    shuffle=True,
    batch_size=64,
    label_column="target_code",
    set_column="set",
):
    df_full = pd.read_csv(dataset_path)
    assert (
        label_column in df_full.columns.values
    ), f"Column {label_column} not available in: {df_full.columns.values}"
    df_full[label_column] = df_full[label_column].astype(str)
    label_map = {
        label: idx for idx, label in enumerate(df_full[label_column].unique())
    }  # NOTE: Keep original order
    num_classes = len(label_map)
    assert (
        set_column in df_full.columns.values
    ), f"Column {set_column} not available in: {df_full.columns.values}"

    dataloaders = {}
    for set_name in set_names:
        df_set = df_full[df_full[set_column] == set_name]
        print(f"Set {set_name} size: {len(df_set)}")
        assert num_classes == len(df_set[label_column].unique())
        series_class_counts = df_set.groupby([label_column]).size()
        class_counts = {
            k: series_class_counts[k] for k in label_map.keys()
        }  # NOTE: Guarantee same order
        dataloaders[set_name] = create_dataloader(
            df=df_set,
            label_columns=[label_column],
            shuffle=shuffle,
            batch_size=batch_size,
            label_map=label_map,
            class_counts=class_counts,
        )
    return dataloaders
