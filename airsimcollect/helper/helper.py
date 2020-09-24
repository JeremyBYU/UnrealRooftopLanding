from os import path
import logging
import collections
import json
import sys

from shapely.geometry import shape
import geojson
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from functools import reduce
import quaternion

from airsim.types import ImageType, Quaternionr, Vector3r
from airsim.utils import to_quaternion

from airsimcollect.helper.helper_transforms import get_seg2rgb_map

logger = logging.getLogger("AirSimCapture")

DIR_PATH = path.dirname(path.realpath(__file__))

WINDOWS_AIRSIM_SETTINGS_PATH = '~/Documents/AirSim/settings.json'
WINDOWS_AIRSIM__SETTINGS_PATH_FULL = path.expanduser(
    WINDOWS_AIRSIM_SETTINGS_PATH)

DEFAULT_SEGMENTATION = {
    "sensor": "Image",
    "type": "Segmentation",
    "camera_name": "0",
    "image_type":  ImageType.Segmentation,
    "pixels_as_float": False,
    "compress": True,
    "retain_data": False
}

DEFAULT_SCENE = {
    "sensor": "Image",
    "type": "Scene",
    "camera_name": "0",
    "image_type": ImageType.Scene,
    "pixels_as_float": False,
    "compress": True,
    "retain_data": False
}


DEFAULT_LIDAR = {
    "sensor": "Lidar",
    "type": "Lidar",
    "segmented": False,
    "lidar_name": "",
    "vehicle_name": "",
    "camera_name": "0",
    "camera_img_type": ImageType.Segmentation,
    "retain_data": False,
    "save_as": "numpy"
}

DEFAULT_CONFIG = {
    "name": "AirSimCollect",
    "sim_mode": "ComputerVision",
    "save_dir": "AirSimCollectData",
    "collector_file_prefix": "",
    "ignore_collision": True,
    "segmentation_codes": [],
    "collectors": [
        DEFAULT_SCENE,
        DEFAULT_SEGMENTATION
    ],
    "collection_points": "",
    "global_id_start": 0,
    "collector_file_prefix": ""
}

AIR_SIM_SETTINGS = dict()

# Lidar frame is NED (x-forward, y-right, z-down), need to transfrom to camera frame.
AIR_SIM_SETTINGS['lidar_to_camera_quat'] = np.quaternion(0.5, -0.5, -0.5, -0.5)

# Default no offset
AIR_SIM_SETTINGS['lidar_to_camera_pos'] = Vector3r(
    x_val=0.0, y_val=0.0, z_val=0.0)


def update_airsim_settings():
    with open(WINDOWS_AIRSIM__SETTINGS_PATH_FULL) as fh:
        data = json.load(fh)

    # Determine if point cloud generated from lidar frame is in NED frame or local sensor frame
    # if in sensor local frame then the 'X' axis (0) hold the 'range' measurement when pointed straight down
    AIR_SIM_SETTINGS['lidar_local_frame'] = False
    lidar_frame = deep_get(data, 'Vehicles.Drone1.Sensors.0.DataFrame')
    AIR_SIM_SETTINGS['lidar_z_col'] = 2
    if lidar_frame == 'SensorLocalFrame':
        AIR_SIM_SETTINGS['lidar_z_col'] = 0
        AIR_SIM_SETTINGS['lidar_local_frame'] = True

    # Determine relative pose offset between camera and lidar frame
    lidar_x = deep_get(data, 'Vehicles.Drone1.Sensors.0.X')
    lidar_y = deep_get(data, 'Vehicles.Drone1.Sensors.0.Y')
    lidar_z = deep_get(data, 'Vehicles.Drone1.Sensors.0.Z')

    camera_x = deep_get(data, 'Vehicles.Drone1.Cameras.0.X')
    camera_y = deep_get(data, 'Vehicles.Drone1.Cameras.0.Y')
    camera_z = deep_get(data, 'Vehicles.Drone1.Cameras.0.Z')

    lidar_roll = deep_get(data, 'Vehicles.Drone1.Sensors.0.Roll')
    lidar_pitch = deep_get(data, 'Vehicles.Drone1.Sensors.0.Pitch')
    lidar_yaw = deep_get(data, 'Vehicles.Drone1.Sensors.0.Yaw')

    camera_roll = deep_get(data, 'Vehicles.Drone1.Cameras.0.Roll')
    camera_pitch = deep_get(data, 'Vehicles.Drone1.Cameras.0.Pitch')
    camera_yaw = deep_get(data, 'Vehicles.Drone1.Cameras.0.Yaw')

    # get delta postion offset
    if lidar_x is not None and camera_x is not None:
        delta_pose: Vector3r = AIR_SIM_SETTINGS['lidar_to_camera_pos']
        dx = lidar_x - camera_x
        dy = lidar_y - camera_y
        dz = lidar_z - camera_z
        # these delta poses must be in the CAMERA frame
        delta_pose.x_val = dy
        delta_pose.y_val = -dx
        delta_pose.z_val = dz
    # get delta rotation, only need this if: 1. Lidar and Camera are not pointed in the same direction. 2: If point clouds are in lidar local frame.
    if lidar_roll is not None and camera_roll is not None and AIR_SIM_SETTINGS['lidar_local_frame']:
        lidar_to_camera_quat = AIR_SIM_SETTINGS['lidar_to_camera_quat']
        d_roll = np.radians(lidar_roll - camera_roll)
        d_pitch = np.radians(lidar_pitch - camera_pitch)
        d_yaw = np.radians(lidar_yaw - camera_yaw)

        d_quat: Quaternionr = to_quaternion(d_pitch, d_roll, d_yaw)
        d_quat = np.quaternion(d_quat.w_val, d_quat.x_val,
                               d_quat.y_val, d_quat.z_val)
        AIR_SIM_SETTINGS['lidar_to_camera_quat'] = lidar_to_camera_quat * d_quat

    cmap_list, seg2rgb_map = get_seg2rgb_map()
    AIR_SIM_SETTINGS['cmap_list'] = np.array(cmap_list)
    AIR_SIM_SETTINGS['seg2rgb_map'] = seg2rgb_map

    return AIR_SIM_SETTINGS


def deep_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_collectors(collectors):
    collectors_new = []
    for collector in collectors:
        if collector['type'] == 'Segmentation':
            new_collector = DEFAULT_SEGMENTATION.copy()
            update(new_collector, collector)
            collectors_new.append(new_collector)
        elif collector['type'] == 'Scene':
            new_collector = DEFAULT_SCENE.copy()
            update(new_collector, collector)
            collectors_new.append(new_collector)
        elif collector['type'] == 'Lidar':
            new_collector = DEFAULT_LIDAR.copy()
            update(new_collector, collector)
            collectors_new.append(new_collector)
    return collectors_new


def import_world(json_fname):
    feature_collection = None
    with open(json_fname) as f:
        feature_collection = geojson.load(f)

    collisions = []
    for feature in feature_collection['features']:
        try:
            feature['geometry'] = shape(feature['geometry'])
            height = feature['properties']['height']
        except KeyError as e:
            logger.error(
                "Feature does not have height property.  GeoJSON feature must have property key with a 'height' key.")
            raise
        if feature['geometry'] .geom_type != 'Point':
            collisions.append(
                (feature['geometry'].bounds, feature['properties']['height']))

    return feature_collection['features'], collisions


def plot_collection_points(points, center, radius):
    fig, ax = plt.subplots(
        1, 1, subplot_kw={'projection': '3d', 'aspect': 'auto'})
    # Plot points
    uvw = center - points[:, :3]
    ax.quiver(points[:, 0], points[:, 1],
              points[:, 2], *uvw.T, length=0.25)

    # generate wire mesh for sphere
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi)) * radius + center[0]
    y = np.outer(np.sin(theta), np.sin(phi)) * radius + center[1]
    z = np.outer(np.cos(theta), np.ones_like(phi)) * radius + center[2]
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, linewidth=0.25)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def image_meta_data_json(images_meta):
    img_meta_data = []
    for img_meta in images_meta:
        data = {
            "camera_name": img_meta['camera_name'],
            "position": dict(img_meta['position'].__dict__),
            "rotation": dict(img_meta['rotation'].__dict__),
            "height": img_meta["height"],
            "width": img_meta["width"],
            "type": img_meta['type']
        }
        img_meta_data.append(data)
    return img_meta_data
