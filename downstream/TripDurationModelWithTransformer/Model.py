import torch.nn as nn
import torch.nn.functional as F
import torch

class TaxiTripTimeRegressor(nn.Module):
    def __init__(self, input_dim, transformer):
        super(TaxiTripTimeRegressor, self).__init__()
        self.transformer = transformer
        self.input_layer = nn.Linear(input_dim, 256)
        self.hidden1 = nn.Linear(256, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.output = nn.Linear(19, 1)

    def forward(self, X_numerical, X_categorical):
        masked = X_numerical.clone()
        X_numerical = torch.nan_to_num(X_numerical)

        x = self.transformer(X_numerical, X_categorical, masked)
        x = self.output(x)
        return x
