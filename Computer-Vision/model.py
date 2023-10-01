import torch


# Base Line model
class BaseLine(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_layers):
        super().__init__()

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features= input_features, out_features= hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= hidden_layers, out_features= hidden_layers),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features= hidden_layers, out_features= output_features)
        )

    def forward(self, X):
        return self.linear_layers(X)



# Simple CNN model
class CNN(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_layers):
        super().__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels= input_features,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )


        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels= hidden_layers,
                      out_channels= hidden_layers,
                      kernel_size= 3,
                      stride= 1,
                      padding= 1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size= 2,
                         stride= 2)
        )


        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(in_features= hidden_layers * 7 * 7,
                      out_features= output_features)
        )

    
    def forward(self, X: torch.Tensor):
        X = self.block1(X)
        X = self.block2(X)
        X = self.classifier(X)

        return X