from torch import nn



class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units= 10):
        super().__init__()

        self.training_layers = nn.Sequential(
            nn.Linear(in_features= input_features, out_features= hidden_units),
            nn.ReLU(True),
            nn.Linear(in_features= hidden_units, out_features= hidden_units),
            nn.ReLU(True),
            nn.Linear(in_features= hidden_units, out_features= output_features),
        )

    
    def forward(self, X):
        return self.training_layers(X)