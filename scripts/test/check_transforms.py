import time
import copy

import open3d as o3d
import numpy as np
import airsim
from airsim.types import ImageResponse, ImageRequest, Quaternionr, Vector3r
import matplotlib.pyplot as plt
import quaternion

from airsimcollect.helper.helper_transforms import parse_lidarData, classify_points
from airsimcollect.helper.o3d_util import get_extrinsics, set_view
from airsimcollect.helper.helper import update_airsim_settings, AIR_SIM_SETTINGS

def set_up_aisim():
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)
    print("Taking off!")
    client.takeoffAsync(timeout_sec=3).join()
    print("Increasing altitude 10 meters!")
    client.moveToZAsync(-10, 2).join()
    print("Reached Altitude, launching Lidar Visualizer")

    return client


def get_lidar_data(client: airsim.MultirotorClient):
    data = client.getLidarData()
    points = parse_lidarData(data)
    pose = data.pose
    return points, pose


def get_image_data(client: airsim.MultirotorClient, channels=3):
    responses = client.simGetImages([airsim.ImageRequest(
        "0", airsim.ImageType.Segmentation, False, False)])
    response: ImageResponse = responses[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgba = img1d.reshape((response.height, response.width, channels))
    # airsim is actually bgr!!
    img_rgba[:, :, [0, 2]] = img_rgba[:, :, [2, 0]]

    img_meta = dict()
    img_meta['rotation'] = response.camera_orientation
    img_meta['position'] = response.camera_position
    img_meta['width'] = response.width
    img_meta['height'] = response.height

    return img_rgba, img_meta


def update_view(vis):
    extrinsics = get_extrinsics(vis)
    vis.reset_view_point(True)
    set_view(vis, extrinsics)


def main():
    airsim_settings = update_airsim_settings()
    client = set_up_aisim()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd = o3d.geometry.PointCloud()
    pcd_camera = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_camera)
    vis.add_geometry(axis)

    prev_time = time.time()
    wait_time = 0.05
    while True:
        if time.time() - prev_time > wait_time:
            points, pose_lidar = get_lidar_data(client)
            if points.size < 10:
                continue
            img, img_meta = get_image_data(client)
            # points = points[~np.isnan(points).any(axis=1)]
            colors, mask, pixels = classify_points(
                img, points, img_meta, airsim_settings)
            img[pixels[:, 1], pixels[:, 0]] = [0, 255, 0]
            # plt.imshow(img)
            # plt.show()
            pcd_colors = np.zeros_like(points)
            pcd_colors = AIR_SIM_SETTINGS['cmap_list'][colors, :3]

            # o3d valid mask...
            nan_mask = ~np.isnan(points).any(axis=1)
            points = points[nan_mask,:]
            pcd_colors = pcd_colors[nan_mask,:]
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            prev_time = time.time()
            # wait_time = 1000
        vis.update_geometry(pcd)
        vis.poll_events()
        update_view(vis)
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    main()
