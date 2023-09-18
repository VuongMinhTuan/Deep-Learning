from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from model import CircleModel
import torch


# Make device agnostic code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Put data to target device
X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

# Call model and put model to target device
model = CircleModel().to(DEVICE)

# Create loss function
loss_func = torch.nn.BCEWithLogitsLoss()

# Create an optimizer
optimizer = torch.optim.Adam(params= model.parameters(), lr= 0.001)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

# Set random seed
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000000

for epoch in range(epochs):
    ### Training
    model.train()

    # Forward pass
    y_prob = model(X_train).squeeze()
    y_pred = torch.round(y_prob)


    # Calculate loss and accuracy
    loss = loss_func(y_prob, y_train)
    acc = accuracy_fn(y_train, y_pred)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model.eval()

    with torch.inference_mode():
        # Forward pass
        test_prob = model(X_test).squeeze()
        test_pred = torch.round(test_prob)

        # Calculate loss and accuracy
        test_loss = loss_func(test_prob, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")