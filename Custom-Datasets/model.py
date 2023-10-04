from torch import nn



class TinyVGG(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_features,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            nn.ReLU(),

            nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_layers * 16 * 16,
                      out_features= output_features)
        )

    
    def forward(self, X):
        out = self.block1(X)
        out = self.block2(out)
        out = self.classifier(out)

        return out