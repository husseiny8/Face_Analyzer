from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    CLIPImageProcessor,
    SiglipImageProcessor,
    AutoImageProcessor
)


class FaceDataset(Dataset):

    def __init__(
        self,
        image_dir,
        csv_file,
        encoder_name="clip",
        processor_path=None,
    ):

        self.image_dir = Path(image_dir)

        self.data = pd.read_csv(csv_file)

        self.encoder_name = encoder_name.lower()

        ####################################################
        # Image Processor
        ####################################################

        if self.encoder_name == "clip":

            if processor_path is None:
                processor_path = "../models/clip-vit-base-patch32"

            self.processor = CLIPImageProcessor.from_pretrained(
                processor_path
            )

        elif self.encoder_name == "siglip":

            if processor_path is None:
                processor_path = "../models/siglip-base-patch16-224"

            self.processor = SiglipImageProcessor.from_pretrained(
                processor_path
            )

        elif self.encoder_name == "dinov2":

            if processor_path is None:
                processor_path = "../models/dinov2-base"

            self.processor = AutoImageProcessor.from_pretrained(
                processor_path
            )

        else:

            raise ValueError(
                f"Unknown encoder: {encoder_name}"
            )

    ########################################################

    def __len__(self):

        return len(self.data)

    ########################################################

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.image_dir / row["filename"]

        if not image_path.exists():

            raise FileNotFoundError(image_path)

        ####################################################
        # Read Image
        ####################################################

        image = Image.open(image_path).convert("RGB")

        ####################################################
        # Image -> Tensor
        ####################################################

        pixel_values = self.processor(

            images=image,

            return_tensors="pt"

        )["pixel_values"].squeeze(0)

        ####################################################
        # Labels
        ####################################################

        gender = torch.tensor(
            row["gender"],
            dtype=torch.float32
        )

        age = torch.tensor(
            row["age"],
            dtype=torch.long
        )

        race = torch.tensor(
            row["race"],
            dtype=torch.long
        )

        ####################################################

        return {
            "pixel_values": pixel_values,
            "gender": gender,
            "age": age,
            "race": race
        }