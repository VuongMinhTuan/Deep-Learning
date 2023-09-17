import torch
from model import LinearRegression
from pathlib import Path
from visualize import plot_predictions

# Setup device agnostic code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Set the manual seed when creating the model
torch.manual_seed(42)
model = LinearRegression()
model.to(DEVICE)

# Create loss
loss_func = torch.nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params= model.parameters(), lr= 0.1)

# Set the number of epochs 
epochs = 1000 

# Put data on the available device
X_train = X_train.to(DEVICE)
X_test = X_test.to(DEVICE)
y_train = y_train.to(DEVICE)
y_test = y_test.to(DEVICE)


for epoch in range(epochs):
    ### Training
    model.train()

    # Forward pass
    y_pred = model(X_train)

    # Calculate loss
    loss = loss_func(y_pred, y_train)

    # Zero grad optmizer
    optimizer.zero_grad()

    # Loss backward
    loss.backward()

    # Step the optimizer
    optimizer.step()


    ### Testing
    model.eval()

    # Forward pass
    with torch.inference_mode():
        test_pred = model(X_test)

        test_loss = loss_func(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


# Create model directory
MODEL_PATH = Path("Best")
MODEL_PATH.mkdir(parents= True, exist_ok= True)

# 2. Create model save path 
MODEL_NAME = "01_Simple_Neural_Network.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict 
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj= model.state_dict(), f= MODEL_SAVE_PATH)


# Turn model into evaluation mode
model.eval()

# Making prediction on the test data
with torch.inference_mode():
    y_pred = model(X_test)


# Visualize the data
plot = plot_predictions(X_train, y_train, X_test, y_test, predictions= y_pred.cpu())
print(plot)