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
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
from astropy.convolution import convolve, Box2DKernel
from shapely.geometry import shape, Polygon
import shapely
from airsim.types import Vector3r, Quaternionr
import cv2
import pandas as pd
import matplotlib.cm as cm
from skimage.transform import resize, rescale

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.helper_transforms import get_seg2rgb_map

from airsimcollect.helper.helper_mesh import (create_meshes_cuda)
from airsimcollect.helper.helper_metrics import load_records
from airsimcollect.helper.helper_confidence_maps import create_fake_confidence_map_seg, create_confidence_map_planarity, create_confidence_map_combined

ROOT_DIR = Path(__file__).parent.parent.parent
SAVED_DATA_DIR = ROOT_DIR / 'AirSimCollectData/LidarRoofManualTest'
GEOSON_MAP = ROOT_DIR / Path("assets/maps/roof-lidar-manual.geojson")
O3D_VIEW = ROOT_DIR / Path("assets/o3d/o3d_view_default.json")
RESULTS_DIR = ROOT_DIR / Path("assets/results")

FOV = 90


def extract_mesh(points_all, config, lidar_beams=64):
    points = points_all[:, :3]
    num_cols = int(points.shape[0] / lidar_beams)
    opc = points.reshape((lidar_beams, num_cols, 3))
    # 1. Create mesh
    alg_timings = dict()
    tri_mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    alg_timings.update(timings)
    return opc, tri_mesh


def update_state(record, position='position', rotation='rotation'):
    position_data = record[position] if isinstance(
        record[position], list) else list(record[position].values())
    rotation_data = record[rotation] if isinstance(
        record[rotation], list) else list(record[rotation].values())
    record[position] = Vector3r(*position_data)
    record[rotation] = np.quaternion(*rotation_data)


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
    for record in records['records']:
        path_key = f"{record['uid']}-{record['sub_uid']}-0"
        bulding_label = record['label']  # building name
        if record['uid'] in [25, 26, 27, 28, 29]:
            logger.warn("Skipping record; UID: %s; SUB-UID: %s; Building Name: %s. Rooftop assets don't match map. Rooftop assets randomness wasn't fixed on this asset!",
                        record['uid'], record['sub_uid'], bulding_label)
            continue
        # uid #45 is best segmentation example
        if record['uid'] < 100:
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

        _, tri_mesh = extract_mesh(pc_np, config)
        t1 = time.perf_counter()
        conf_map_plan = create_confidence_map_planarity(
            tri_mesh, img_meta, airsim_settings)
        t2 = time.perf_counter()
        conf_map_comb = create_confidence_map_combined(
            conf_map_seg, conf_map_plan)

        conf_map_seg_color = cm.viridis(
            conf_map_seg)[:, :, :3][..., ::-1].copy()
        conf_map_plan_color = cm.viridis(conf_map_plan)[
            :, :, :3][..., ::-1].copy()
        conf_map_comb_color = cm.viridis(conf_map_comb)[
            :, :, :3][..., ::-1].copy()
        img_scene = np.concatenate((img_scene, img_seg), axis=1)
        img_conf = np.concatenate(
            (conf_map_seg_color, conf_map_plan_color, conf_map_comb_color), axis=1)

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
