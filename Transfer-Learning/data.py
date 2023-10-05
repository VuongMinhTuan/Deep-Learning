import requests
import zipfile
import os
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# Create dataset
def create_data(path):
    # Set up path to data folder
    data_path = Path(path)
    image_path = data_path / "pizza_steak_sushi"

    try:
        # If the image folder doesn't exist, download it and prepare it
        if image_path.is_dir():
            print(f"{image_path} directory exists!")
        else:
            print(f"Did not find {image_path} directory, creating one...")
            image_path.mkdir(parents= True, exist_ok= True)

            # Download data
            with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
                request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
                print("Downloading pizza, steak and sushi data...")
                f.write(request.content)

            # Unzip data
            with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
                print("Unzipping pizza, steak and sushi data...")
                zip_ref.extractall(image_path)

            print("Data are created successfully!!!")


            # Remove .zip file
            os.remove(data_path / "pizza_steak_sushi.zip")
    except:
        print("Can not create the data folder!!!")


# Create data loader
def create_dataLoaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int):
    
    
    # Create train and test datasets
    train_data = datasets.ImageFolder(root= train_dir, transform= transform)
    test_data = datasets.ImageFolder(root= test_dir, transform= transform)

    # Create data classes
    data_classes = train_data.classes

    # Create train and test dataloaders
    train_dataloader = DataLoader(dataset= train_data,
                                  batch_size= batch_size,
                                  shuffle= True,
                                  num_workers= num_workers,
                                  pin_memory= True)
    
    test_dataloader = DataLoader(dataset= test_data,
                                 batch_size= batch_size,
                                 num_workers= num_workers,
                                 pin_memory= True)
    

    return (train_dataloader, test_dataloader, data_classes)