import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import numpy as np
from torchvision.models import vit_h_14, ViT_H_14_Weights
from transformers import ViTFeatureExtractor
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

# Load the pretrained ViT model
weights = ViT_H_14_Weights.DEFAULT
vit_model = vit_h_14(weights='DEFAULT') # Downloading: "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth" to /root/.cache/torch/hub/checkpoints/vit_h_14_swag-80465313.pth
model = vit_model.to(device)
vit_model.eval()



# image_path = '/content/drive/MyDrive/Cityscapes/test/berlin_000000_000019_leftImg8bit.png'  # Replace with your image path
# image_path = './UMD_322.png'
# image_path = '***/Cityscapes/test/berlin_000000_000019_leftImg8bit.png'  # Replace with your image path
image_path = 'deeplab1.png'  # deeplab dummy
image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
print(image.size)

# # resize as needed
# base_width = 640
# wpercent = (base_width / float(image.size[0]))
# hsize = int((float(image.size[1]) * float(wpercent)))
# image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)


# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(image).unsqueeze(0).to(device)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%") # traffic light: 63.0%

print("num categories:", len(weights.meta["categories"]))


class ViTSegmentation(nn.Module):
    def __init__(self, vit_model, num_classes):
        super(ViTSegmentation, self).__init__()
        self.vit = vit_model
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, kernel_size=2, stride=2), # Added transposed convolution layers
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
        )

    def forward(self, x):
        features = self.vit(x)
        features = features.unsqueeze(2).unsqueeze(3)
        features = features.expand(-1, 1000, 14, 14)
        output = self.decoder(features)
        return output

# Assuming you have already defined ViTSegmentation as before
# Load the pretrained ViT model and your custom segmentation model
n_classes = 45 # originally 21
segmentation_model = ViTSegmentation(vit_model=vit_model, num_classes=n_classes).to(device)

# Set the model to evaluation mode
segmentation_model.eval()

# Define a function to preprocess an existing image tensor
def preprocess_image_tensor(image):
    # Define the transformations (resize, normalization) similar to ViTFeatureExtractor
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # Resize to match ViT input size
        transforms.ConvertImageDtype(torch.float),  # Convert to float32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

    return image_tensor

# Define a function to postprocess the output
def postprocess_output(output):
    # Apply argmax to get the most likely class for each pixel
    output_predictions = torch.argmax(output, dim=1).squeeze().detach().cpu().numpy()

    return output_predictions

# Read the image
# image_tensor = read_image(image_path)  # This reads image as [C, H, W]
transform = transforms.ToTensor()
image_tensor = transform(image)

# Ensure the image has 3 channels
if image_tensor.shape[0] != 3:
    raise ValueError(f"Expected 3 channels, but got {image_tensor.shape[0]} channels")

# Preprocess the image tensor
input_tensor = preprocess_image_tensor(image_tensor).to(device)

# Perform inference
with torch.no_grad():
    output = segmentation_model(input_tensor)

# Postprocess the output
segmentation_map = postprocess_output(output)

# Convert to an image for visualization
segmentation_image = Image.fromarray(segmentation_map.astype(np.uint8))

# Save or display the segmentation image
# segmentation_image.save('***/Cityscapes/segmentation_map.png')
segmentation_image.save('segmentation_map.png')
segmentation_image.show()