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
import joblib

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
from descartes import PolygonPatch
import matplotlib.pyplot as plt

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.helper_transforms import get_seg2rgb_map

from airsimcollect.helper.helper_mesh import (create_meshes_cuda)
from airsimcollect.helper.helper_metrics import load_records, select_building, load_map, compute_metric
from airsimcollect.helper.o3d_util import create_frustum
from airsimcollect.helper.helper_confidence_maps import (create_fake_confidence_map_seg,
                                                         create_confidence_map_planarity,
                                                         create_confidence_map_combined,
                                                         get_homogenous_projection_matrices,
                                                         create_bbox_raster_from_polygon,
                                                         create_confidence_map_planarity2)
from airsimcollect.helper.helper_polylidar import extract_polygons

from fastga import GaussianAccumulatorS2Beta, IcoCharts
from polylidar import Polylidar3D

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


def get_polygon_inside_frustum(pc_np, pl, ga, ico, config, distance_to_camera, camera_position, building_feature):

    frustum_points = create_frustum(
        distance_to_camera, camera_position, hfov=FOV, vfov=FOV)
    pl_planes, alg_timings, tri_mesh, avg_peaks = extract_polygons(
        pc_np, None, pl, ga, ico, config, segmented=True)
    if pl_planes:
        seg_gt_iou, pl_poly_estimate_seg, gt_poly = compute_metric(
            building_feature, pl_planes, frustum_points)
        return tri_mesh, avg_peaks, pl_poly_estimate_seg, gt_poly
    else:
        return None, None, None, None


def main(gui=True, segmented=False):
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict = load_records(
        SAVED_DATA_DIR)
    airsim_settings = records['airsim_settings']

    # have to turn some json keys into proper objects, quaternions...
    update_state(airsim_settings, position='lidar_to_camera_pos',
                 rotation='lidar_to_camera_quat')

    _, seg2rgb_map = get_seg2rgb_map(normalized=False)
    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        config = yaml.safe_load(file)

    start_offset_unreal = np.array(records['start_offset_unreal'])
    map_features_dict = load_map(GEOSON_MAP, start_offset_unreal)

    # Create Polylidar Objects
    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

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

        # map feature of the building
        building_features = map_features_dict[bulding_label]
        camera_position = Vector3r(
            **record['sensors'][0]['position']).to_numpy_array()
        building_feature = select_building(building_features, camera_position)
        distance_to_camera = building_feature['ned_height'] - \
            camera_position[2]

        # have to turn some json keys into proper objects, quaternions...
        img_meta = record['sensors'][0]
        update_state(img_meta)

        logger.info("Inspecting record; UID: %s; SUB-UID: %s; Building Name: %s",
                    record['uid'], record['sub_uid'], bulding_label)

        # Load Images
        img_scene = cv2.imread(str(scene_paths_dict[path_key]))
        img_seg = cv2.imread(str(segmentation_paths_dict[path_key]))
        # Load LiDAR Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))


        tri_mesh, avg_peaks, pl_poly_estimate_seg, gt_poly = get_polygon_inside_frustum(
            pc_np, pl, ga, ico, config, distance_to_camera, camera_position, building_feature)
        
        if tri_mesh is None:
            continue

        
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[1].add_patch(PolygonPatch(pl_poly_estimate_seg, ec='k', alpha=0.5, zorder=2),)
        ax[1].autoscale_view()
        ax[1].axis('equal')

        raster = create_confidence_map_planarity2(tri_mesh, pl_poly_estimate_seg)
        ax[0].imshow(raster, origin='lower')

        plt.show()


        # conf_map_seg = create_fake_confidence_map_seg(img_seg, seg2rgb_map)
        # t1 = time.perf_counter()
        # conf_map_plan = create_confidence_map_planarity(
        #     tri_mesh, img_meta, airsim_settings)
        # t2 = time.perf_counter()
        # conf_map_comb = create_confidence_map_combined(
        #     conf_map_seg, conf_map_plan)

        # conf_map_seg_color = cm.viridis(
        #     conf_map_seg)[:, :, :3][..., ::-1].copy()
        # conf_map_plan_color = cm.viridis(conf_map_plan)[
        #     :, :, :3][..., ::-1].copy()
        # conf_map_comb_color = cm.viridis(conf_map_comb)[
        #     :, :, :3][..., ::-1].copy()
        # img_scene = np.concatenate((img_scene, img_seg), axis=1)
        # img_conf = np.concatenate(
        #     (conf_map_seg_color, conf_map_plan_color, conf_map_comb_color), axis=1)

        # hom, proj = get_homogenous_projection_matrices(
        #     img_meta, airsim_settings)

        # data = dict(hom=hom, proj=proj, conf_map_comb=conf_map_comb, conf_map_plan=conf_map_plan, conf_map_seg_gt=conf_map_seg,
        #             poly=pl_poly_estimate_seg, poly_normal=avg_peaks, lidar_local_frame=airsim_settings['lidar_local_frame'])
        # joblib.dump(data, SAVED_DATA_DIR / 'Processed' / (path_key[:-2] + '.pkl'), compress=True)
        # cv2.imshow('Scene View', img_scene)
        # cv2.imshow('Confidence View', img_conf)
        # res = cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Check LiDAR")
    parser.add_argument('--gui', dest='gui', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.gui)
