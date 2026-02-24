import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Build full image paths and keep only existing ones
        self.data['img_path'] = self.data['Image Index'].apply(lambda x: os.path.join(self.img_dir, x))
        self.data = self.data[self.data['img_path'].apply(os.path.exists)]

        # Handle text label column
        if 'Finding Labels' in self.data.columns:
            self.data['Finding Labels'] = self.data['Finding Labels'].apply(lambda x: x.split('|'))
            all_labels = sorted(set(l for sub in self.data['Finding Labels'] for l in sub))
            for label in all_labels:
                self.data[label] = self.data['Finding Labels'].apply(lambda x: 1.0 if label in x else 0.0)
            print(f"✅ Found label columns: {all_labels}")

        # Replace invalid numeric values with NaN → then fill with 0
        self.data = self.data.apply(pd.to_numeric, errors='ignore')
        label_cols = [c for c in self.data.columns if c not in ['Image Index', 'Finding Labels', 'img_path']]
        self.data[label_cols] = self.data[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        print(f"✅ Loaded {len(self.data)} valid image entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_path']

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠ Skipping image {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        label_cols = [c for c in self.data.columns if c not in ['Image Index', 'Finding Labels', 'img_path']]
        labels = torch.tensor(row[label_cols].astype(float).values, dtype=torch.float32)

        return image, labels


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


if __name__ == "__main__":
    csv_path = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample_labels.csv"
    img_dir = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample\images"

    dataset = ChestXrayDataset(csv_path, img_dir, transform=get_transforms(train=True))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print("🔍 Testing DataLoader...")
    for batch in dataloader:
        if batch is None:
            continue
        images, labels = batch
        print("✅ Batch loaded successfully!")
        print("Image batch shape:", images.shape)
        print("Labels batch shape:", labels.shape)
        break

