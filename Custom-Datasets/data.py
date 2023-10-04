import requests
import zipfile
from pathlib import Path
from torchvision import transforms, datasets


# Create data
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
    except:
        print("Can not create the data folder!!!")



# Transform data
def transform_data():
    data_transform = transforms.Compose([
        transforms.Resize(size= (64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins= 31),
        transforms.ToTensor()
    ])
    
    return data_transform


# Create train and test datasets
def train_test_datasets(train_dir, test_dir):
    # Create train dataset
    train_data = datasets.ImageFolder(
        root= train_dir,
        transform= transform_data(),
        target_transform= None
    )

    # Create test dataset
    test_data = datasets.ImageFolder(
        root= test_dir,
        transform= transform_data()
    )

    return {"Train": train_data, "Test": test_data}