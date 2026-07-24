import torch
import torch.nn as nn


##########################################################
# Gender Head
##########################################################

class GenderHead(nn.Module):

    """
    Binary Classification

    Output:
        Logit

    Loss:
        BCEWithLogitsLoss
    """

    def __init__(self, input_dim):

        super().__init__()

        self.fc = nn.Linear(input_dim, 1)

        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.zeros_(self.fc.bias)

    def forward(self, x):

        return self.fc(x)


##########################################################
# Age Head
##########################################################

class AgeHead(nn.Module):

    """
    9-Class Classification

    Output:
        Logits

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

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 9)

        )

        nn.init.xavier_uniform_(self.classifier[0].weight)

        nn.init.zeros_(self.classifier[0].bias)

        nn.init.xavier_uniform_(self.classifier[3].weight)

        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, x):

        return self.classifier(x)


##########################################################
# Race Head
##########################################################

class RaceHead(nn.Module):

    """
    7-Class Classification

    Output:
        Logits

    Loss:
        CrossEntropyLoss
    """

    def __init__(self, input_dim):

        super().__init__()

        self.fc = nn.Linear(input_dim, 7)

        nn.init.xavier_uniform_(self.fc.weight)

        nn.init.zeros_(self.fc.bias)

    def forward(self, x):

        return self.fc(x)