import torch


class CircleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Sequential (
            torch.nn.Linear(in_features= 2, out_features= 10),
            torch.nn.Linear(in_features= 10, out_features= 10),
            torch.nn.ReLU(True),
            torch.nn.Linear(in_features= 10, out_features= 1)
        )

        # self.activation_layers = torch.nn.Sequential (
        #     torch.nn.ReLU(True)
        # )

        self.output_layer = torch.nn.Sequential (
            torch.nn.Sigmoid()
        )

    # Override forward function
    def forward(self, X):
        out = self.linear(X)
        # out = self.activation_layers(out)
        out = self.output_layer(out)
        return out