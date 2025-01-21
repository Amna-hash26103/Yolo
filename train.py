from ultralytics import YOLO
import yaml
import cv2
import os

# Load the dataset configuration
with open(r"D:/Bano/Projs/yol_back_change/Dataset/kitti.yaml", 'r') as stream:
    dataset_config = yaml.safe_load(stream)

# Assuming the dataset configuration contains paths to the training data
train_data_path = dataset_config['train']
# Change the file permissions to read and write for the owner
os.chmod(train_data_path, 0o600)

# Load the dataset (this part may vary depending on how the dataset is structured)
# Here, we assume it's a list of image paths
with open(train_data_path, 'r') as f:
    train_data = f.readlines()

# Print the number of training samples
print(f"Number of training samples: {len(train_data)}")

# Load the first image to get its shape
first_image_path = train_data[0].strip()
first_image = cv2.imread(first_image_path)
print(f"Shape of the first training image: {first_image.shape}")

# Initialize and train the model
model = YOLO(r"D:/Bano/Projs/yol_back_change/ultralytics/ultralytics/cfg/models/11/yolo11.yaml")
model.train(
    epochs=2,
    batch=2,
    data="D:/Bano/Projs/yol_back_change/Dataset/kitti.yaml",
    name="backbonechange"
)