import pandas as pd
import torch.utils.data as data
from torchvision import datasets


class DataframeImageDataset(data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        filepath_column="image",
        label_columns=["label"],
        loader=datasets.folder.pil_loader,
    ):
        for column in [filepath_column, *label_columns]:
            assert column in df.columns, f"Dataframe does not include column: {column}"
        self.df = df
        self.transform = transform
        self.filepath_column = filepath_column
        self.label_columns = label_columns
        self.loader = loader

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = self.df[self.filepath_column].iloc[index]
        labels = self.df[self.label_columns].iloc[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, *labels
