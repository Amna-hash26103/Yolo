from PIL import Image
import yaml

# Load dataset configuration
with open(r"D:/Bano/Projs/yol_back_change/Dataset/kitti.yaml", 'r') as stream:
    dataset_config = yaml.safe_load(stream)

# Assuming the dataset configuration contains paths to the training data
train_data_path = dataset_config['train']

# Load the dataset (this part may vary depending on how the dataset is structured)
# Here, we assume it's a list of image paths
with open(train_data_path, 'r') as f:
    train_data = f.readlines()

# Load the first image from the dataset
first_image_path = train_data[0].strip()
image = Image.open(first_image_path)

# Print the shape of the image
print(f"Image shape: {image.size}")  