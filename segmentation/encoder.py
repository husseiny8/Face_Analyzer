import torch
import torch.nn as nn
from transformers import (CLIPVisionModel,SiglipVisionModel,Dinov2Model)

class VisionEncoder(nn.Module):

    def __init__(
            self,
            encoder_name="clip",
            model_path=None,
            freeze=True
    ):

        super().__init__()
        encoder_name = encoder_name.lower()

        # CLIP
        if encoder_name == "clip":
            if model_path is None:
                model_path = "../models/clip-vit-base-patch32"

            self.encoder = CLIPVisionModel.from_pretrained(model_path)
            self.feature_dim = self.encoder.config.hidden_size

        # SigLIP
        elif encoder_name == "siglip":
            if model_path is None:
                model_path = "../models/siglip-base-patch16-224"

            self.encoder = SiglipVisionModel.from_pretrained(model_path)
            self.feature_dim = self.encoder.config.hidden_size

        # DINOv2
        elif encoder_name == "dinov2":
            if model_path is None:
                model_path = "../models/dinov2-base"

            self.encoder = Dinov2Model.from_pretrained(model_path)

            self.feature_dim = self.encoder.config.hidden_size

        else:
            raise ValueError(f"Unknown encoder : {encoder_name}")

        # Freeze Encoder
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False


    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)

        # CLS Token
        features = outputs.last_hidden_state[:, 0]

        return features