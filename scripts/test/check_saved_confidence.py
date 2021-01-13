"""Will check all the lidar returned
"""
import logging
import json
import sys
import time
from pathlib import Path
from os import listdir
from os.path import isfile, join
from functools import partial
import argparse

import yaml
import quaternion
from rich import print as rprint
import numpy as np
from shapely.geometry import shape, Polygon
import shapely
from airsim.types import Vector3r, Quaternionr
import cv2
import pandas as pd
import matplotlib.cm as cm
from skimage.transform import resize, rescale

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.helper_transforms import get_seg2rgb_map, colors2class, get_pixels_from_points

from airsimcollect.helper.helper_mesh import (create_meshes_cuda)

ROOT_DIR = Path(__file__).parent.parent.parent
SAVED_DATA_DIR = ROOT_DIR / 'AirSimCollectData/LidarRoofManualTest'
GEOSON_MAP = ROOT_DIR / Path("assets/maps/roof-lidar-manual.geojson")
O3D_VIEW = ROOT_DIR / Path("assets/o3d/o3d_view_default.json")
RESULTS_DIR = ROOT_DIR / Path("assets/results")

FOV = 90


def convert_dict(directory, suffix='.'):
    return {f.split('.')[0]: directory / f for f in listdir(directory)}


def load_records(directory):
    lidar_directory = directory / Path("Lidar")
    scene_directory = directory / Path("Scene")
    segmentation_directory = directory / Path("Segmentation")
    records_path = directory / 'records.json'

    with open(records_path) as f:
        records = json.load(f)

    lidar_paths_dict = convert_dict(lidar_directory)
    scene_paths_dict = convert_dict(scene_directory)
    segmentation_paths_dict = convert_dict(segmentation_directory)

    return records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict


def extract_mesh(points_all, config, lidar_beams=64):
    points = points_all[:, :3]
    num_cols = int(points.shape[0] / lidar_beams)
    opc = points.reshape((lidar_beams, num_cols, 3))
    # 1. Create mesh
    alg_timings = dict()
    tri_mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    alg_timings.update(timings)
    return opc, tri_mesh


def create_fake_confidence_map_seg(img_seg, seg2rgb_map, ds=4, roof_class=4):
    img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
    img_seg_rgb_ = rescale(img_seg_rgb, 1 / ds, multichannel=True,
                           mode='edge',
                           anti_aliasing=False,
                           anti_aliasing_sigma=None,
                           preserve_range=True,
                           order=0)
    classes = colors2class(img_seg_rgb_, seg2rgb_map).astype(np.float32)
    classes[classes != roof_class] = 0.0
    classes[classes == roof_class] = 0.90

    return classes


def create_confidence_map_planarity(tri_mesh, img_meta, airsim_settings, ds=4):
    shape = np.array(
        (img_meta['height'], img_meta['width']), dtype=np.int) // ds
    img_meta['height'] = shape[0]
    img_meta['width'] = shape[1]

    conf_map_plan = np.zeros(
        (img_meta['height'], img_meta['width']), dtype=np.float32)
    t1 = time.perf_counter()
    triangles_np = np.asarray(tri_mesh.triangles)
    vertices_np = np.asarray(tri_mesh.vertices)
    triangle_normals_np = np.asarray(tri_mesh.triangle_normals)
    t2 = time.perf_counter()
    triangle_centroids = vertices_np[triangles_np].mean(axis=1)
    t3 = time.perf_counter()
    triangles_planarity = triangle_normals_np @ np.array([[0], [0], [-1]])
    t4 = time.perf_counter()
    #
    pixels, mask = get_pixels_from_points(
        triangle_centroids, img_meta, airsim_settings)
    t5 = time.perf_counter()
    triangles_planarity_filt = np.squeeze(triangles_planarity[mask, :])
    t6 = time.perf_counter()
    conf_map_plan[pixels[:, 1], pixels[:, 0]] = triangles_planarity_filt
    t7 = time.perf_counter()

    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000
    ms3 = (t4-t3) * 1000
    ms4 = (t5-t4) * 1000
    ms5 = (t6-t5) * 1000
    ms6 = (t7-t6) * 1000
    # print(ms1, ms2, ms3, ms4, ms5, ms6)

    return conf_map_plan


def update_state(record, position='position', rotation='rotation'):
    # print("before")
    # print(record)
    position_data = record[position] if isinstance(
        record[position], list) else list(record[position].values())
    rotation_data = record[rotation] if isinstance(
        record[rotation], list) else list(record[rotation].values())
    record[position] = Vector3r(*position_data)
    record[rotation] = np.quaternion(*rotation_data)
    # print("after")
    # print(record)


def main(gui=True, segmented=False):
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict = load_records(
        SAVED_DATA_DIR)
    airsim_settings = records['airsim_settings']
    update_state(airsim_settings, position='lidar_to_camera_pos',
                 rotation='lidar_to_camera_quat')

    _, seg2rgb_map = get_seg2rgb_map(normalized=False)
    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        config = yaml.safe_load(file)
    result_records = []
    for record in records['records']:
        path_key = f"{record['uid']}-{record['sub_uid']}-0"
        bulding_label = record['label']  # building name
        if record['uid'] in [25, 26, 27, 28, 29]:
            logger.warn("Skipping record; UID: %s; SUB-UID: %s; Building Name: %s. Rooftop assets don't match map. Rooftop assets randomness wasn't fixed on this asset!",
                        record['uid'], record['sub_uid'], bulding_label)
            continue
        # uid #45 is best segmentation example
        if record['uid'] < 0:
            continue

        img_meta = record['sensors'][0]
        update_state(img_meta)

        logger.info("Inspecting record; UID: %s; SUB-UID: %s; Building Name: %s",
                    record['uid'], record['sub_uid'], bulding_label)

        # Load Images
        img_scene = cv2.imread(str(scene_paths_dict[path_key]))
        img_seg = cv2.imread(str(segmentation_paths_dict[path_key]))
        # Load LiDAR Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))

        conf_map_seg = create_fake_confidence_map_seg(img_seg, seg2rgb_map)
        conf_map_seg_color = cm.viridis(
            conf_map_seg)[:, :, :3][..., ::-1].copy()

        _, tri_mesh = extract_mesh(pc_np, config)
        t1 = time.perf_counter()
        conf_map_plan = create_confidence_map_planarity(
            tri_mesh, img_meta, airsim_settings)
        t2 = time.perf_counter()
        print(t2-t1)
        conf_map_plan_color = cm.viridis(conf_map_plan)[
            :, :, :3][..., ::-1].copy()

        img_scene = np.concatenate((img_scene, img_seg), axis=1)
        img_conf = np.concatenate(
            (conf_map_seg_color, conf_map_plan_color), axis=1)
        cv2.imshow('Scene View', img_scene)
        cv2.imshow('Confidence View', img_conf)
        res = cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Check LiDAR")
    parser.add_argument('--gui', dest='gui', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.gui)
