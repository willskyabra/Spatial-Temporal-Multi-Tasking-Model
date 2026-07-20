import torch.nn as nn
import torch.nn.functional as F

class DropoffZoneClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DropoffZoneClassifier, self).__init__()
        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Linear(256, 256)
        self.hidden2 = nn.Linear(256, 128)
        # self.hidden3 = nn.Linear(128, 128)
        # self.hidden4 = nn.Linear(128, 128)
        self.output = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x
