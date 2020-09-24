import time
import copy

import open3d as o3d
import numpy as np
import airsim
from airsim.types import ImageResponse, ImageRequest, Quaternionr, Vector3r
import matplotlib.pyplot as plt
import quaternion

from airsimcollect.helper.helper_transforms import parse_lidarData, create_projection_matrix
from airsimcollect.helper.o3d_util import get_extrinsics, set_view
from airsimcollect.helper.helper import get_airsim_settings_file, AIR_SIM_SETTINGS

lidar_to_camera_quat = AIR_SIM_SETTINGS['lidar_to_camera_quat']
lidar_to_camera_pos = AIR_SIM_SETTINGS['lidar_to_camera_pos']


def transform_to_cam(points, cam_pos=lidar_to_camera_pos, cam_quat=lidar_to_camera_quat, invert=False, points_in_unreal=False):
    temp = points.copy()
    points = np.ones(shape=(4, points.shape[0]))
    points[:3, :] = temp.transpose()

    if points_in_unreal:
        # Need to scale down to meters
        points[:3, :] = points[:3, :] / 100.0
        # Need to convert to NED coordinate for homogoneous transformation matrix
        temp = points.copy()
        points[0, :], points[1, :], points[2,
                                           :] = temp[0, :], temp[1, :], -temp[2, :]

    # Points are in NED, wide form
    # Now transform them
    hom_transform = create_homogenous_transform(
        cam_pos, cam_quat, invert=invert)

    point_cam_ned = hom_transform.dot(points)
    point_cam_hom = point_cam_ned.copy()
    # Ha, so that was where I was fixing the camera coordinates, not needed anymore
    # point_cam_hom[0, :], point_cam_hom[1, :], point_cam_hom[2,
    #                                                         :] = point_cam_ned[1, :], point_cam_ned[2, :], point_cam_ned[0, :]
    return point_cam_hom


def create_homogenous_transform(cam_pos=lidar_to_camera_pos, cam_quat=lidar_to_camera_quat, invert=False):
    cam_pos = np.array([cam_pos.x_val, cam_pos.y_val, cam_pos.z_val])
    rot_mat = quaternion.as_rotation_matrix(cam_quat)

    hom_tran = np.zeros(shape=(4, 4))
    hom_tran[:3, :3] = rot_mat
    hom_tran[:3, 3] = -1 * rot_mat.dot(cam_pos) if invert else cam_pos
    hom_tran[3, 3] = 1

    return hom_tran


def project_points_img(points, proj_mat, width, height, points_orig):
    pixels = proj_mat.dot(points)
    pixels = np.divide(pixels[:2, :], pixels[2, :]).transpose().astype(np.int)

    # Remove pixels that are outside the image
    mask_x = (pixels[:, 0] < width) & (pixels[:, 0] > 0)
    mask_y = (pixels[:, 1] < height) & (pixels[:, 1] > 0)
    mask = mask_x & mask_y
    # Return the pixels and points that are inside the image
    pixels = pixels[mask]
    points_orig = points_orig[mask, :]
    return pixels, points_orig, mask


def classify_points(img_meta, points, airsim_settings):

    cam_ori = img_meta['rotation']
    cam_quat = np.quaternion(cam_ori.w_val, cam_ori.x_val,
                             cam_ori.y_val, cam_ori.z_val)
    cam_pos = img_meta['position']
    height = img_meta['height']
    width = img_meta['width']
    if airsim_settings['lidar_local_frame']:
        transform_pos = airsim_settings['lidar_to_camera_pos']
        transform_rot = airsim_settings['lidar_to_camera_quat']
        invert = False
    else:
        transform_pos = cam_pos
        transform_rot = airsim_settings['lidar_to_camera_quat'] * \
            cam_quat.conjugate()
        invert = True

    proj_mat = create_projection_matrix(height, width)
    # Transform NED points to camera coordinate system (not NED)
    points_transformed = transform_to_cam(
        points, cam_pos=transform_pos, cam_quat=transform_rot, invert=invert, points_in_unreal=False)
    # Project Points into image, filter points outside of image
    pixels, points, mask = project_points_img(
        points_transformed, proj_mat, width, height, points)

    return pixels, points_transformed, mask


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
    airsim_settings = get_airsim_settings_file()
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
            img, img_meta = get_image_data(client)
            points = points[~np.isnan(points).any(axis=1)]
            pixels, points_transformed, mask = classify_points(
                img_meta, points, airsim_settings)
            img[pixels[:, 1], pixels[:, 0]] = [0, 255, 0]
            plt.imshow(img)
            plt.show()
            # points_transformed = points_transformed.transpose()[:, :3]
            # print(points_transformed)
            # pcd_camera.points = o3d.utility.Vector3dVector(points_transformed)
            # pcd_camera.paint_uniform_color([1,0,0])
            # wait_time = 10000
            pcd.points = o3d.utility.Vector3dVector(points)
            prev_time = time.time()
        vis.update_geometry(pcd)
        vis.poll_events()
        update_view(vis)
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    main()
