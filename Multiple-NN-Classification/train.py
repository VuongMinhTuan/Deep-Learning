import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from model import BlobModel
from matplotlib import pyplot as plt


# Set the parameters for training and data creation
NUM_SAMPLES = 1000
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
EPOCHS = 100000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Set the seed
torch.manual_seed(RANDOM_SEED)


# Create multi-class data
X, y = make_blobs(
    n_samples= NUM_SAMPLES,
    n_features= NUM_FEATURES,
    centers= NUM_CLASSES,
    cluster_std= 1.5,
    random_state= RANDOM_SEED
)


# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= RANDOM_SEED)

# Create a instance of model
model = BlobModel(
    input_features= NUM_FEATURES,
    output_features= NUM_CLASSES,
    hidden_units= 10
).to(DEVICE)


# Create a loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)


# Put data to target device
X_train, X_test = X_train.to(DEVICE), X_test.to(DEVICE)
y_train, y_test = y_train.to(DEVICE), y_test.to(DEVICE)


# Calculate the accuracy
def accuracy(y_pred, y):
    correct = torch.eq(y_pred, y).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc



for epoch in range(EPOCHS):
    ### Training
    model.train()

    # Forward pass
    y_logit = model(X_train).squeeze()
    y_pred = torch.softmax(y_logit, dim= 1).argmax(dim= 1)

    # Calculate loss and accuracy of train dataset
    loss = loss_func(y_logit, y_train)
    acc = accuracy(y_pred, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    ### Testing
    model.eval()

    with torch.inference_mode():
        # Forward pass
        y_logit = model(X_test).squeeze()
        y_pred = torch.softmax(y_logit, dim= 1).argmax(dim= 1)

        # Calculate the loss and accuracy of test dataset
        test_loss = loss_func(y_logit, y_test)
        test_acc = accuracy(y_pred, y_test)


    # Print out what's happening
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

