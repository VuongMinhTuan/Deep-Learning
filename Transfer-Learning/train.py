from pathlib import Path
from data import create_data, create_dataLoaders, os
from utils import train, accuracy, plot_curves, torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


PATH = Path("C:\Tuan\GitHub\Deep-Learning\Transfer-Learning\Data")
TRAIN_DIR = PATH / "pizza_steak_sushi" / "train"
TEST_DIR = PATH / "pizza_steak_sushi" / "test"
NUM_WORKERS = os.cpu_count()
RANDOM_SEED = 42
BATCH_SZIE = 32
EPOCHS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0001


# Set up the manual seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

if __name__ == "__main__":

    # Create dataset
    create_data(PATH)


    # Set up weights of pre-trained model
    weights = EfficientNet_B0_Weights.DEFAULT


    # Create data loader and data classes
    train_dataloader, test_dataloader, classes_name = create_dataLoaders(train_dir= TRAIN_DIR,
                                                                        test_dir= TEST_DIR,
                                                                        transform = weights.transforms(),
                                                                        batch_size= BATCH_SZIE,
                                                                        num_workers= NUM_WORKERS)


    # Set up the model
    model = efficientnet_b0(weights= weights).to(DEVICE)


    # Modify the classifier layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p= 0.2, inplace= True),

        torch.nn.Linear(in_features=1280,
                        out_features= len(classes_name),
                        bias= True).to(DEVICE)
    )


    # Set up loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE)


    # Training
    results = train(model= model,
                    train_dataloader= train_dataloader,
                    test_dataloader= test_dataloader,
                    optimizer= optimizer,
                    loss_func= loss_func,
                    device= DEVICE,
                    accuracy_func = accuracy,
                    epochs= EPOCHS)


    plot_curves(results)


# Epoch: 1000 | train_loss: 0.0574 | train_acc: 100.0000 | test_loss: 0.2767 | test_acc: 88.0000