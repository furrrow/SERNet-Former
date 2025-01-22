import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# !wget http://images.cocodataset.org/val2017/000000005477.jpg -q -O input.jpg
im0= read_image("input.jpg")
im1= read_image("UMD_585.png")
im1 = torchvision.transforms.Resize((349, 640))(im1)
print(im0.shape)
print(im1.shape)

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(im1).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

print(weights.meta["categories"])
interest_list = ["__background__", "car", "person"]
for mask_type in interest_list:
    mask = normalized_masks[0, class_to_idx[mask_type]]
    pil_img = to_pil_image(mask)
    pil_img.show()

# torch.save(mask, './mask.pt')