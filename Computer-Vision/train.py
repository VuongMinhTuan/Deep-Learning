from model import BaseLine, CNN
from utils import train_step, test_step, accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch


# Set the parameters for training and data creation
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_FEATURES = 784
HIDDEN_LAYERS = 10
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(RANDOM_SEED)


# Set up training dataset
train_data = datasets.FashionMNIST(
    root= "Data",
    train= True,
    download= True,
    transform= ToTensor(),
    target_transform= None
)


# Set up testing dataset
test_data = datasets.FashionMNIST(
    root= "Data",
    train= False,
    download= True,
    transform= ToTensor(),
    target_transform= None
)


# Turn train dataset into DataLoader
train_dataloader = DataLoader(
    dataset= train_data,
    batch_size= BATCH_SIZE,
    shuffle= True
)


# Turn test dataset into DataLoader
test_dataloader = DataLoader(
    dataset= test_data,
    batch_size= BATCH_SIZE,
    shuffle= True
)


# Set up base line model
cnn = CNN(
    input_features= 1,
    output_features= NUM_CLASSES,
    hidden_layers= HIDDEN_LAYERS
).to(DEVICE)


# Set up loss, optimizer and evaluation metrics
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= cnn.parameters(), lr= 0.01)


for epoch in range(EPOCHS):
    train_loss, train_acc = train_step (
                                model= cnn,
                                data_loader= train_dataloader,
                                loss_func= loss_func,
                                optimizer= optimizer,
                                accuracy_func= accuracy,
                                device= DEVICE
                            )

    test_loss, test_acc = test_step (
                            model= cnn,
                            data_loader= test_dataloader,
                            loss_func= loss_func,
                            accuracy_func= accuracy,
                            device= DEVICE
                        )

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1} | Loss: {train_loss:.5f}, Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")