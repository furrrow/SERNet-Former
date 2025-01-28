import torch
from PIL import Image
import os
import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
"""

traverse dict for "facebook/mask2former-swin-small-cityscapes-semantic"
{0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}
note "facebook/mask2former-swin-large-cityscapes-instance" does worse?

"""
img_pkl_path = f"/home/jim/Documents/Projects/GND_alignment/new_AU/AU/AU"
point_cloud_path = f"/home/jim/Documents/Projects/GND_alignment/AU/Scans"
times_txt_path = f"/home/jim/Documents/Projects/GND_alignment/AU/times.txt"
pretrain_model = "segformer" # select mask2former or segformer

# GMU intrinstic
intrinsic = [[248.621049856283, 0, 333.674701475662],
             [0, 249.443487017099, 174.156912509125],
             [0, 0, 1]]

# GMU extrinsic
extrinsic = [[0, -1, 0, -1.73960485887045],
             [0, 0, -1, -0.547552836129094],
             [1, 0, 0, 0.995256541782342],
             [0, 0, 0, 1]]

traverse_def_dict = {
    1: "1. side walk or other traversable areas for pedestrians",
    2: "2. the parking lot, drive way for vehicles",
    3: "3. off-road including vegetation, grass, mud etc",
    4: "4. stairs, curbs etc for legged robot only",
    5: "5. obstacles or buildings",
    6: "6. do not segment, discard"
}

traversability_dict = {  # applies for both mask2former and segformer
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
    12: 6, # 'rider',
    13: 6, # 'car',
    14: 6, # 'truck',
    15: 6, # 'bus',
    16: 6, # 'train',
    17: 6, # 'motorcycle',
    18: 6, # 'bicycle'
}


def lidar_to_egoview(image, pcd_points, intrinsic, extrinsic):
    image = np.array(image)
    image_size = image.shape
    H, W, D = image_size
    depth_image = np.full((H, W), np.inf)
    # Convert point cloud to homogeneous coordinates
    points_hom = np.hstack((pcd_points, np.ones((pcd_points.shape[0], 1))))
    # Transform to camera coordinates
    points_cam = (extrinsic @ points_hom.T).T
    points_cam = points_cam[points_cam[:, 2] > 0]  # Filter points with z > 0
    # print(points_hom)
    raw_pcd = (np.linalg.inv(extrinsic) @ points_cam.T).T
    # Project to image plane
    pixels_hom = (intrinsic @ points_cam[:, :3].T).T
    pixels = pixels_hom[:, :2] / pixels_hom[:, 2:]
    refined_pcd = []
    refined_color = []
    # Rasterize depth image
    for i, (u, v) in enumerate(pixels):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            z = points_cam[i, 2]
            depth_image[v, u] = min(depth_image[v, u], z)

            refined_pcd.append(raw_pcd[i, :])
            refined_color.append([u, v])
    # Replace inf with 0 (or a desired background value)
    depth_image[depth_image == np.inf] = 0
    refined_pcd = np.array(refined_pcd)[:, :3]
    refined_color = np.array(refined_color)
    return depth_image, refined_pcd, refined_color

# Semantic segmentation from the same dataset:
print(f"using {pretrain_model} model")
if pretrain_model == "segformer":
    processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
elif pretrain_model == "mask2former":
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(traverse_def_dict))]
color_palette = np.array(color_palette)

for i, (img_dir, lidar_dir) in enumerate(zip(os.listdir(img_pkl_path), os.listdir(point_cloud_path))):
    print(f"extracting {i} out of {len(os.listdir(img_pkl_path))}")
    img_path = os.path.join(img_pkl_path, img_dir)
    lidar_path = os.path.join(point_cloud_path, lidar_dir)
    if os.path.isfile(lidar_path):
        pcd = o3d.io.read_point_cloud(lidar_path)
        pcd_points = np.array(pcd.points)
    else:
        print(f"warning: file {lidar_path} does not exist")
        continue
    if os.path.isfile(img_path):
        with open(img_path, "rb") as f:
            img_data = pickle.load(f)
            img = img_data["camera"]
            pose = img_data["pose"]
            time = img_data["time"]
    else:
        print(f"warning: file {img_path} does not exist")
        continue
    image = Image.fromarray(img[0])
    # get depth image from lidar transformation
    depth_image, refined_pcd, refined_color = lidar_to_egoview(image, pcd_points, intrinsic, extrinsic)


    # semantic segmentation
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # you can pass them to processor for postprocessing
    results = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    print(f"segmentation shape {results.shape}")
    # print(model.config.id2label)
    assert len(traversability_dict) == len(model.config.id2label)
    # convert segmentation results according to traversability_dict
    segmentation = results.detach().cpu().numpy()

    color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for key, value in traversability_dict.items():
        segmentation[segmentation == key] = value

    print(segmentation.shape)
    i = 0
    for label, definition in traverse_def_dict.items():
        # text_label = model.config.id2label[label]
        text_label = definition
        # overlay
        color_seg[segmentation == label, :] = color_palette[i]

        # Show image + mask
        mask = (segmentation == label)
        visual_mask = (mask * 255).astype(np.uint8)
        visual_mask = Image.fromarray(visual_mask)
        # cv2.imshow(f"{text_label} mask", np.array(visual_mask))
        i+=1
    seg_overlay = np.array(image) * 0.5 + color_seg * 0.5
    seg_overlay = seg_overlay.astype(np.uint8)
    cv2.imshow(f"overlay segmentation", seg_overlay)
    cv2.imshow("input image", np.array(image))
    cv2.imshow("depth image", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break










# image shape: (H, W, C) where H=480, W=640
# segmentation shape: (H, W), encoded 1 through 6
# depth_matrix shape? (H, W, D)? where D could range from 0 to 10

