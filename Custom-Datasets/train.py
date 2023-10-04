import os
import torch
from data import create_data, train_test_datasets
from torch.utils.data import DataLoader
from pathlib import Path
from model import TinyVGG
from utils import train, accuracy, plot_curves


PATH = Path("C:\\Tuan\\GitHub\\Deep-Learning\\Custom-Datasets\\Data\\")
TRAIN_DIR = PATH / "pizza_steak_sushi" / "train"
TEST_DIR = PATH / "pizza_steak_sushi" / "test"
NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACTH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 100


if __name__ == "__main__":

    # Create data folder
    create_data(PATH)


    # Create dataset
    dataset = train_test_datasets(train_dir= TRAIN_DIR, test_dir= TEST_DIR)
    train_data, test_data = dataset["Train"], dataset["Test"]


    # Turn train data into DataLoader
    train_dataloader = DataLoader(
        dataset= train_data,
        batch_size= BACTH_SIZE,
        num_workers= NUM_WORKERS,
        shuffle= True,
    )


    # Turn test data into DataLoader
    test_dataloader = DataLoader(
        dataset= test_data,
        batch_size= BACTH_SIZE,
        num_workers= NUM_WORKERS,
        shuffle= True
    )


    # Set up random seed
    torch.manual_seed(RANDOM_SEED)


    # Set up TinyVGG model
    model = TinyVGG(input_features= 3,
                    hidden_layers= 10,
                    output_features= len(train_data.classes)).to(DEVICE)


    # Set up loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params= model.parameters(), lr= 0.001)



    # Training model
    model_results = train(model= model,
                        train_dataloader= train_dataloader,
                        test_dataloader= test_dataloader,
                        optimizer= optimizer,
                        loss_func= loss_func,
                        device= DEVICE,
                        accuracy_func= accuracy,
                        epochs= EPOCHS)


    # Plot the results
    plot_curves(model_results)