import torch
from torch import nn
import numpy as np
from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
"""

traverse dict for "facebook/mask2former-swin-small-cityscapes-semantic"
{0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}
note "facebook/mask2former-swin-large-cityscapes-instance" does worse?

"""
image = Image.open("UMD_322.png")

# Semantic segmentation from the same dataset:
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
inputs = processor(images=image, return_tensors="pt")
for k,v in inputs.items():
    print(k,v.shape)
with torch.no_grad():
    outputs = model(**inputs)

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits # torch.Size([1, 100, 20])
masks_queries_logits = outputs.masks_queries_logits # torch.Size([1, 100, 96, 96])

# you can pass them to processor for postprocessing
results = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
print(results.shape)
# print(model.config.id2label)
print(model.config.id2label)
traverse_def_dict = {
    1: "1. side walk or other traversable areas for pedestrians",
    2: "2. the parking lot, drive way for vehicles",
    3: "3. off-road including vegetation, grass, mud etc",
    4: "4. stairs, curbs etc for legged robot only",
    5: "5. obstacles or buildings",
    6: "6. do not segment, discard"
}
traversability_dict = {
    0: 2, # 'road',
    1: 1, # 'sidewalk',
    2: 5, # 'building',
    3: 5, # 'wall',
    4: 5, # 'fence',
    5: 5, # 'pole',
    6: 5, # 'traffic light',
    7: 5, # 'traffic sign',
    8: 3, # 'vegetation',
    9: 3, # 'terrain',
    10: 6, # 'sky',
    11: 6, # 'person', # go with 6 from here  and below... dynamic obstacle is cat 6.
    12: 5, # 'rider',
    13: 5, # 'car',
    14: 5, # 'truck',
    15: 5, # 'bus',
    16: 5, # 'train',
    17: 5, # 'motorcycle',
    18: 5, # 'bicycle'
}

assert len(traversability_dict) == len(model.config.id2label)

# convert segmentation results according to traversability_dict
segmentation = results.detach().cpu().numpy()

for key, value in traversability_dict.items():
    segmentation[segmentation == key] = value
print(segmentation.shape)

# image shape: (H, W, C) where H=480, W=640
# segmentation shape: (H, W), encoded 1 through 6
# depth_matrix shape? (H, W, D)? where D could range from 0 to 10

