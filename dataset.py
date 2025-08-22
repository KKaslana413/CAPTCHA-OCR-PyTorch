import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from config import Image_Width, Image_Height, CHAR_LIST
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore", message="Error fetching version info*")


char2idx = {char: idx + 1 for idx, char in enumerate(CHAR_LIST)}
idx2char = {idx + 1: char for idx, char in enumerate(CHAR_LIST)}


class CaptchaDataset(Dataset):
    def __init__(self, folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError(f"Dataset folder '{folder_path}' does not exist.")
        self.image_paths = []
        self.labels = []

        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".png"):
                label = fname.split("_", 1)[-1].replace(".png", "")
                self.image_paths.append(os.path.join(folder_path, fname))
                self.labels.append(label)

        if len(self.image_paths) == 0:
            raise ValueError(f"No .png images in {folder_path}")

        self.image_cache = {}
        self.aug = A.Compose([
            A.Resize(height=Image_Height, width=Image_Width),  # 唯一 resize 点
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(scale=(0.97, 1.03), translate_percent=(0.03, 0.03), rotate=(-5, 5), p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 显式三通道
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_str = self.labels[idx]

        # 校验 label 中字符合法性
        for c in label_str:
            if c not in char2idx:
                raise ValueError(f"Invalid char '{c}' in label for {image_path}")

        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            if w < h:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # 不做额外 resize（只在 augment 中做）
            self.image_cache[image_path] = image

        augmented = self.aug(image=image)
        image = augmented['image']
        label = torch.tensor([char2idx[c] for c in label_str], dtype=torch.long)

        return image, label
