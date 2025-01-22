import torch
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def convert_xyz_to_latlon(x, y, z, pose, origin = (30.288130, -97.73762, 0)):
    """
    Convert points from LiDAR coordinate system to global latitude/longitude coordinates

    Parameters:
    x, y, z: coordinates in LiDAR system (x forward, y left, z up)
    pose: 8-dimensional array [timestamp, x, y, z, qx, qy, qz, qw]

    Returns:
    lat, lon: global latitude/longitude coordinates
    """
    # Define the origin coordinates (latitude, longitude) and altitude
    # origin = (30.28805556, -97.7375, 0)  # UT Austin Start Coordinates
    # origin = (30.288114, -97.737699, 0)  # AHG original
    # origin = (30.288130, -97.73762, 0)     # Very Good

    # Get quaternion from pose
    qw, qx, qy, qz = pose[4:]

    # Create rotation object
    r = R.from_quat([qx, qy, qz, qw])

    # Transform point from LiDAR coordinate system to global coordinate system
    point_local = r.apply(np.array([x, y, z]))

    # Add vehicle position offset
    point_global = point_local + np.array([pose[1], pose[2], pose[3]])

    # Return the latitude, longitude, and altitude
    return point_global


H = 480
W = 640
dummy_image = np.ones((H, W, 3))
dummy_traverse = np.ones((H, W)) # 1- 6
dummy_lidar = np.ones((200, 3)) # lidar frame
lidar_global = np.ones((200, 4))
# throw into convert function to get global frame

dummy_lidar_indices_rgb_frame = np.ones((200, 2)) # camera frame
ldiar_semantic = np.ones((200, 1))

bands = 4
incr = int(H / bands)
for i in np.arange(bands):
    dummy_traverse[i*incr: (i+1)*incr] = i+1
    dummy_depth[i*incr: (i+1)*incr] = (i+1)*10
# plt.imshow(dummy_traverse)
# plt.show()

# pose: 8-dimensional array [timestamp, x, y, z, qx, qy, qz, qw]
dummy_pose = [12.3456, 1, 1, 1, 1, 1, 1, 0.5]
origin = (30, -30, 0)
for i in range(H):
    for j in range(W):
        depth = dummy_depth[i, j]
        traverse = dummy_traverse[i, j]
        pose = dummy_pose.copy()
        x = depth
        y = -j
        z = i
        lat_long = convert_xyz_to_latlon(x, y, z, pose, origin)
