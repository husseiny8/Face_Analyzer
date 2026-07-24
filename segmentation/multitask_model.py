import torch
import torch.nn as nn

from encoder import VisionEncoder
from heads import (
    GenderHead,
    AgeHead,
    RaceHead
)


class MultiTaskModel(nn.Module):

    def __init__(
            self,
            encoder_name="clip",
            model_path=None,
            freeze_encoder=True,
            projection_dim=512,
            dropout=0.3
    ):

        super().__init__()

        ########################################################
        # Encoder
        ########################################################

        self.encoder = VisionEncoder(
            encoder_name=encoder_name,
            model_path=model_path,
            freeze=freeze_encoder
        )

        feature_dim = self.encoder.feature_dim

        ########################################################
        # Shared Projection
        ########################################################

        self.projection = nn.Sequential(

            nn.Linear(
                feature_dim,
                projection_dim
            ),

            nn.BatchNorm1d(
                projection_dim
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout)

        )

        ########################################################
        # Multi Task Heads
        ########################################################

        self.gender_head = GenderHead(
            projection_dim
        )

        self.age_head = AgeHead(
            projection_dim
        )

        self.race_head = RaceHead(
            projection_dim
        )

    ############################################################

    def forward(self, pixel_values):

        ########################################################
        # Encoder
        ########################################################

        features = self.encoder(pixel_values)

        ########################################################
        # Projection
        ########################################################

        embedding = self.projection(features)

        ########################################################
        # Heads
        ########################################################

        gender_logits = self.gender_head(
            embedding
        )

        age_logits = self.age_head(
            embedding
        )

        race_logits = self.race_head(
            embedding
        )

        ########################################################

        return {

            "gender": gender_logits,

            "age": age_logits,

            "race": race_logits

        }

    ############################################################

    @torch.no_grad()
    def predict(self, pixel_values):

        self.eval()

        outputs = self.forward(pixel_values)

        ########################################################
        # Gender
        ########################################################

        gender_prob = torch.sigmoid(
            outputs["gender"]
        )

        gender_pred = (
                gender_prob >= 0.5
        ).long().squeeze(1)

        ########################################################
        # Age
        ########################################################

        age_prob = torch.softmax(
            outputs["age"],
            dim=1
        )

        age_pred = age_prob.argmax(
            dim=1
        )

        ########################################################
        # Race
        ########################################################

        race_prob = torch.softmax(
            outputs["race"],
            dim=1
        )

        race_pred = race_prob.argmax(
            dim=1
        )

        ########################################################

        return {

            "gender_logits": outputs["gender"],
            "gender_probability": gender_prob,
            "gender_prediction": gender_pred,

            "age_logits": outputs["age"],
            "age_probability": age_prob,
            "age_prediction": age_pred,

            "race_logits": outputs["race"],
            "race_probability": race_prob,
            "race_prediction": race_pred

        }

    ############################################################

    def freeze_encoder(self):

        for p in self.encoder.parameters():

            p.requires_grad = False

    ############################################################

    def unfreeze_encoder(self):

        for p in self.encoder.parameters():

            p.requires_grad = True