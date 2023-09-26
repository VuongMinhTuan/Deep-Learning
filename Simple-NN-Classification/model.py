import torch


class CircleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Sequential (
            torch.nn.Linear(in_features= 2, out_features= 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features= 10, out_features= 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features= 10, out_features= 1)
        )

    # Override forward function
    def forward(self, X):
        out = self.linear(X)
        return out