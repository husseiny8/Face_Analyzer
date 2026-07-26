import torch
import torch.nn as nn

# Gender Head
# چون فقط دو کلاس داریم (Male / Female)،
# مسئله از نوع Binary Classification است.
# خروجی این شبکه فقط یک عدد (Logit) است.
# در مرحله آموزش از BCEWithLogitsLoss استفاده می‌شود
class GenderHead(nn.Module):
    """
    Binary Classification

    Output:
        1 Logit

    Loss:
        BCEWithLogitsLoss
    """

    def __init__(self, input_dim):

        super().__init__()

        # ورودی:
        #     بردار ویژگی استخراج شده توسط CLIP
        # خروجی:
        #     فقط یک Logit
        self.fc = nn.Linear(input_dim, 1)

        # مقداردهی اولیه وزن‌ها با Xavier
        # باعث همگرایی بهتر شبکه می‌شود.
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):

        # x :
        # Feature Vector استخراج شده توسط Encoder
        # خروجی:
        # یک Logit
        return self.fc(x)


# Age Head
# تشخیص سن به صورت Classification انجام شده است.
# به دلیل دشوار بودن مسئله سن،
# از یک شبکه دو لایه استفاده شده است
# تا مدل بتواند روابط پیچیده‌تری را یاد بگیرد.
# خروجی:
#     9 Logit
# Loss:
#     CrossEntropyLoss
# در CrossEntropyLoss نیز Softmax داخل خود Loss انجام می‌شود،
class AgeHead(nn.Module):

    """
    9-Class Classification

    Output:
        9 Logits

    Loss:
        CrossEntropyLoss
    """

    def __init__(
            self,
            input_dim,
            hidden_dim=512,
            dropout=0.3
    ):

        super().__init__()
        self.classifier = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            # روابط غیرخطی را یاد می‌گیرد.
            nn.ReLU(inplace=True),

            # Dropout
            # جلوگیری از Overfitting
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 9)
        )

        # مقداردهی اولیه Xavier
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)

        nn.init.xavier_uniform_(self.classifier[3].weight)
        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, x):

        # خروجی:
        # 9 Logit
        return self.classifier(x)


# Race Head
# تشخیص نژاد نیز یک مسئله Multi-Class Classification است.
# خروجی:
#     7 Logit
# Loss:
#     CrossEntropyLoss
class RaceHead(nn.Module):

    """
    7-Class Classification

    Output:
        7 Logits

    Loss:
        CrossEntropyLoss
    """

    def __init__(self, input_dim):

        super().__init__()

        # تبدیل Feature Vector
        # به احتمال 7 کلاس نژادی
        self.fc = nn.Linear(input_dim, 7)

        # مقداردهی اولیه وزن‌ها
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # خروجی:
        # 7 Logit
        return self.fc(x)