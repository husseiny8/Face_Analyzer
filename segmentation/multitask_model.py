import torch
import torch.nn as nn
from encoder import VisionEncoder
from heads import (
    GenderHead,
    AgeHead,
    RaceHead
)

# Multi Task Model
# این کلاس کل مدل پروژه را تشکیل می‌دهد.
# ساختار کلی مدل به صورت زیر است:
#
#          Image
#             │
#             ▼
#     Vision Transformer
#             │
#             ▼
#      Feature Vector
#             │
#             ▼
#     Shared Projection
#      (Feature Refinement)
#             │
#      ┌──────┼────────┐
#      ▼      ▼        ▼
#   Gender   Age      Race
#     Head   Head      Head
#
# در این پروژه فقط Encoder مشترک است
# ولی هر وظیفه Head اختصاصی خودش را دارد.
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

        # Encoder
        # Encoder مسئول استخراج ویژگی از تصویر است.
        #
        # در این پروژه از CLIP استفاده شده است.
        #
        # در صورت نیاز می‌توان SigLIP یا DINOv2
        # را نیز جایگزین کرد.
        #
        # اگر freeze=True باشد
        # وزن‌های Encoder ثابت باقی می‌مانند
        # و فقط Headها آموزش می‌بینند.
        #
        # این همان Transfer Learning است.
        self.encoder = VisionEncoder(
            encoder_name=encoder_name,
            model_path=model_path,
            freeze=freeze_encoder
        )

        # تعداد ویژگی‌های خروجی Encoder
        feature_dim = self.encoder.feature_dim


        # Shared Projection
        # خروجی Encoder مستقیماً وارد Headها نمی‌شود.
        # ابتدا وارد Projection می‌شود.
        self.projection = nn.Sequential(
            nn.Linear(feature_dim,projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gender_head = GenderHead(projection_dim)
        self.age_head = AgeHead(projection_dim)
        self.race_head = RaceHead(projection_dim)

    def forward(self, pixel_values):

        # مرحله اول
        # استخراج ویژگی توسط CLIP
        features = self.encoder(pixel_values)


        # مرحله دوم
        # Feature Projection
        embedding = self.projection(features)


        # مرحله سوم
        # ارسال Feature مشترک
        # به سه Head مختلف

        gender_logits = self.gender_head(embedding)
        age_logits = self.age_head(embedding)
        race_logits = self.race_head(embedding)

        return {
            "gender": gender_logits,
            "age": age_logits,
            "race": race_logits
        }


    @torch.no_grad()
    def predict(self, pixel_values):
        self.eval()
        outputs = self.forward(pixel_values)



        # Gender
        # آن را به احتمال بین صفر و یک تبدیل می‌کند.
        # سپس اگر احتمال >=0.5 باشد
        # کلاس Male
        # در غیر اینصورت
        # Female
        gender_prob = torch.sigmoid(outputs["gender"])
        gender_pred = (gender_prob >= 0.5).long().squeeze(1)


        # Age
        # سپس کلاس با بیشترین احتمال انتخاب می‌شود.
        age_prob = torch.softmax(outputs["age"],dim=1)
        age_pred = age_prob.argmax(dim=1)



        race_prob = torch.softmax(outputs["race"],dim=1)
        race_pred = race_prob.argmax(dim=1)


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


    # Freeze Encoder
    # تمام وزن‌های Encoder قفل می‌شوند.
    # دیگر Gradient دریافت نمی‌کنند.
    # فقط Headها آموزش می‌بینند.
    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


    # Unfreeze Encoder
    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True