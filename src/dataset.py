import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DrivingDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.data = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        angle = float(row["angle"])
        return image, angle


def time_split(labels_file, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(labels_file)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df