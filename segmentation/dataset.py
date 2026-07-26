from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (CLIPImageProcessor,SiglipImageProcessor,AutoImageProcessor)

class FaceDataset(Dataset):
    def __init__(
            self,
            image_dir=None,
            csv_file=None,
            dataset=None,
            encoder_name="clip",
            processor_path=None
    ):

        self.dataset = dataset

        if dataset is None:
            self.image_dir = Path(image_dir)
            self.data = pd.read_csv(csv_file)
            self.use_csv = True

        # Original FairFace Dataset
        else:
            self.use_csv = False
            self.data = dataset

        encoder_name = encoder_name.lower()

        if encoder_name == "clip":
            if processor_path is None:
                processor_path = "../models/clip-vit-base-patch32"

            self.processor = CLIPImageProcessor.from_pretrained(
                processor_path
            )

        elif encoder_name == "siglip":
            if processor_path is None:
                processor_path = "../models/siglip-base-patch16-224"

            self.processor = SiglipImageProcessor.from_pretrained(
                processor_path
            )

        elif encoder_name == "dinov2":
            if processor_path is None:
                processor_path = "../models/dinov2-base"

            self.processor = AutoImageProcessor.from_pretrained(
                processor_path
            )

        else:
            raise ValueError(f"Unknown encoder : {encoder_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use_csv:
            row = self.data.iloc[idx]

            image = Image.open(self.image_dir / row["filename"]).convert("RGB")

            gender = row["gender"]
            age = row["age"]
            race = row["race"]

        else:
            row = self.data[idx]
            image = row["image"].convert("RGB")
            gender = row["gender"]
            age = row["age"]
            race = row["race"]

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "gender": torch.tensor(
                gender,
                dtype=torch.float32
            ),
            "age": torch.tensor(
                age,
                dtype=torch.long
            ),
            "race": torch.tensor(
                race,
                dtype=torch.long
            )
        }