from os import path
import logging
import collections

from shapely.geometry import shape
import geojson
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from airsim.types import ImageType

logger = logging.getLogger("AirSimCapture")

DIR_PATH = path.dirname(path.realpath(__file__))


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
        1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
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
