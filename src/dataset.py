import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import csv

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

def generate_labels_csv(path):
    with open(path, "r") as infile, open("data/labels.csv", "w", newline="") as outfile:
        data = infile.read()
        writer = csv.writer(outfile)
        writer.writerow(["filename", "angle"])
        lines = data.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue

            parts = line.split(" ")
            filename = parts[0]
            angle = parts[1].split(",")[0]
            writer.writerow([filename, angle])

if __name__ == "__main__":
    generate_labels_csv("data/raw/images/data.txt")


