from transformers import CLIPImageProcessor
from torch.utils.data import Dataset
import torch


class FairFaceDataset(Dataset):

    def __init__(self, dataset, encoder_name="clip"):

        self.dataset = dataset

        self.processor = CLIPImageProcessor.from_pretrained(
            "../models/clip-vit-base-patch32"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"].convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {

            "pixel_values": pixel_values,

            "gender": torch.tensor(
                sample["gender"],
                dtype=torch.float32
            ),

            "age": torch.tensor(
                sample["age"],
                dtype=torch.long
            ),

            "race": torch.tensor(
                sample["race"],
                dtype=torch.long
            )

        }