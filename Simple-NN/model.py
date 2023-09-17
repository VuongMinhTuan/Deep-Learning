import torch

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features= 1, out_features= 1)
        
    # Override forward fuction
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)