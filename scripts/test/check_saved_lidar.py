"""Will check all the lidar returned
"""
import logging
import json
import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join
from functools import partial
import argparse

import yaml
from rich import print as rprint
import numpy as np
from shapely.geometry import shape, Polygon
import shapely
from airsim.types import Vector3r, Quaternionr
import cv2
import pandas as pd

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.o3d_util import (update_linemesh, handle_linemeshes, init_vis, create_frustum,
                                           update_o3d_colored_point_cloud, create_linemesh_from_shapely,
                                           update_frustum, load_view_point, save_view_point, toggle_visibility)
from airsimcollect.helper.helper_mesh import (
    create_meshes_cuda, update_open_3d_mesh_from_tri_mesh)
from airsimcollect.helper.helper_metrics import create_frustum_intersection, select_polygon, select_building, choose_dominant_plane_normal
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_classified_mesh

from fastga import GaussianAccumulatorS2Beta, IcoCharts
from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D, MatrixUInt8


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


def load_map(fpath, start_offset_unreal):
    """Attempts to load a polygon geojson file"""
    with open(fpath) as f:
        poly_geojson = json.load(f)
    # print(poly_geojson)
    features = dict()
    for feature in poly_geojson['features']:
        # in the unreal coordiante systems
        # rprint(feature)
        height = feature['properties']['height']
        class_label = feature['properties']['class_label']
        polygon = shape(feature['geometry'])
        # translate the polygon to make make UCF coincide with NED
        polygon = shapely.affinity.translate(
            polygon, *(-1 * start_offset_unreal).tolist())
        # Scale polgyon from cm to meters
        polygon = shapely.affinity.scale(
            polygon, xfact=0.01, yfact=0.01, zfact=0.01, origin=(0, 0, 0))
        ned_height = -height/100.0 + start_offset_unreal[2] * .01
        # rprint(polygon)
        line_meshes = create_linemesh_from_shapely(polygon, ned_height)
        centroid = np.array(
            [polygon.centroid.x, polygon.centroid.y, ned_height])
        feature_data = dict(ned_height=ned_height, polygon=polygon,
                            line_meshes=line_meshes, class_label=class_label, centroid=centroid)
        if class_label in features:
            features[class_label].append(feature_data)
        else:
            features[class_label] = [feature_data]

    return features


def extract_polygons(points_all, all_polys, pl, ga, ico, config,
                     lidar_beams=64, segmented=True, roof_class=4):
    points = points_all[:, :3]
    num_cols = int(points.shape[0] / lidar_beams)
    opc = points.reshape((lidar_beams, num_cols, 3))
    # 1. Create mesh
    alg_timings = dict()
    tri_mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    alg_timings.update(timings)

    # Get classes for each vertex and set them
    classes = np.expand_dims(points_all[:, 3].astype(np.uint8), axis=1)
    classes[classes == 255] = roof_class
    classes[classes != roof_class] = 0
    classes[classes == roof_class] = 1
    classes_mat = MatrixUInt8(classes)
    tri_mesh.set_vertex_classes(classes_mat, True)

    # 2. Get dominant plane normals
    avg_peaks, _, _, _, timings = extract_all_dominant_plane_normals(
        tri_mesh, ga_=ga, ico_chart_=ico, **config['fastga'])
    # only looking for most dominant plane of the rooftop
    avg_peaks = choose_dominant_plane_normal(avg_peaks)
    alg_timings.update(timings)
    # 3. Extract Planes and Polygons
    planes, obstacles, timings = extract_planes_and_polygons_from_classified_mesh(tri_mesh, avg_peaks, pl_=pl,
                                                                                  filter_polygons=True, segmented=segmented,
                                                                                  postprocess=config['polygon']['postprocess'])
    alg_timings.update(timings)
    # 100 ms to plot.... wish we had opengl line-width control
    if all_polys is not None:
        update_linemesh(planes, all_polys)
    return planes, alg_timings


def compute_metric(building_feature, pl_planes, frustum_points):
    # Create 3D shapely polygons of polylidar estimate and "ground truth" surface LIMITED to the sensor field of view of the camera frustum
    pl_poly_estimate = create_frustum_intersection(select_polygon(
        building_feature, pl_planes), frustum_points)
    gt_poly = create_frustum_intersection(
        building_feature['polygon'], frustum_points)

    base_iou = pl_poly_estimate.intersection(
        gt_poly).area / pl_poly_estimate.union(gt_poly).area

    return base_iou, pl_poly_estimate, gt_poly


def main(gui=True, segmented=False):
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict = load_records(
        SAVED_DATA_DIR)

    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        config = yaml.safe_load(file)

    start_offset_unreal = np.array(records['start_offset_unreal'])
    map_features_dict = load_map(GEOSON_MAP, start_offset_unreal)

    # Create Polylidar Objects
    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

    # Initialize 3D Viewer, Map and Misc geometries
    if gui:
        vis, geometry_set = init_vis()
        line_meshes = [feature['line_meshes']
                       for features in map_features_dict.values() for feature in features]
        line_meshes = [
            line_mesh for line_mesh_set in line_meshes for line_mesh in line_mesh_set]
        geometry_set['map_polys'].line_meshes = handle_linemeshes(
            vis, geometry_set['map_polys'].line_meshes, line_meshes)
        load_view_point(vis, str(O3D_VIEW))

        vis.register_key_callback(ord("X"), partial(
            toggle_visibility, geometry_set, 'pcd'))
        vis.register_key_callback(ord("C"), partial(
            toggle_visibility, geometry_set, 'map_polys'))
        vis.register_key_callback(ord("V"), partial(
            toggle_visibility, geometry_set, 'pl_polys'))
        vis.register_key_callback(ord("B"), partial(
            toggle_visibility, geometry_set, 'frustum'))
        vis.register_key_callback(ord("N"), partial(
            toggle_visibility, geometry_set, 'pl_isec'))
        vis.register_key_callback(ord("M"), partial(
            toggle_visibility, geometry_set, 'gt_isec'))
    else:
        geometry_set = dict(pl_polys=None)

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

        # ground truth to fix, 162, 135, 40, 22

        logger.info("Inspecting record; UID: %s; SUB-UID: %s; Building Name: %s",
                    record['uid'], record['sub_uid'], bulding_label)
        # map feature of the building
        building_features = map_features_dict[bulding_label]
        camera_position = Vector3r(
            **record['sensors'][0]['position']).to_numpy_array()
        building_feature = select_building(building_features, camera_position)
        distance_to_camera = building_feature['ned_height'] - \
            camera_position[2]

        # Load LiDAR Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))
        # handle gui
        if gui:
            # Create Frustum
            update_frustum(vis, distance_to_camera, camera_position,
                           hfov=FOV, vfov=FOV,
                           frustum=geometry_set['frustum'])

            # Load Images
            img_scene = cv2.imread(str(scene_paths_dict[path_key]))
            img_seg = cv2.imread(str(segmentation_paths_dict[path_key]))
            img = np.concatenate((img_scene, img_seg), axis=1)
            cv2.imshow('Scene View'.format(record['uid']), img)

            # Load Lidar Data
            update_o3d_colored_point_cloud(pc_np, geometry_set['pcd'].geometry)
            frustum_points = geometry_set['frustum'].line_meshes[0].points
        else:
            frustum_points = create_frustum(
                distance_to_camera, camera_position, hfov=FOV, vfov=FOV)

        # Polygon Extraction of surface
        pl_planes, alg_timings = extract_polygons(pc_np, geometry_set['pl_polys'] if not segmented else None, pl, ga,
                                                  ico, config, segmented=False)

        pl_planes_seg, alg_timings_seg = extract_polygons(pc_np, geometry_set['pl_polys'] if segmented else None, pl, ga,
                                                  ico, config, segmented=True)

        if pl_planes and True:
            base_iou, pl_poly_estimate, gt_poly = compute_metric(
                building_feature, pl_planes, frustum_points)
            seg_gt_iou, pl_poly_estimate_seg, _ = compute_metric(
                building_feature, pl_planes_seg, frustum_points)
            logger.info("Polylidar3D Base IOU - %.1f; Seg GT IOU - %.1f", base_iou * 100, seg_gt_iou * 100)

            result_records.append(dict(uid=record['uid'], sub_uid=record['sub_uid'],
                                       building=bulding_label, pl_base_iou=base_iou,
                                       pl_seg_gt_iou=seg_gt_iou,
                                       **alg_timings))
            # Visualize these intersections
            if gui:
                # Visualize the polylidar with segmentation results 
                if segmented:
                    pl_poly_estimate = pl_poly_estimate_seg
                update_linemesh([pl_poly_estimate], geometry_set['pl_isec'])
                update_linemesh([gt_poly], geometry_set['gt_isec'])
        elif gui:
            update_linemesh([], geometry_set['pl_isec'])
            update_linemesh([], geometry_set['gt_isec'])

        if gui:
            # Update geometry and view
            vis.update_geometry(geometry_set['pcd'].geometry)
            vis.update_renderer()
            while(True):
                vis.poll_events()
                vis.update_renderer()
                res = cv2.waitKey(10)
                if res != -1:
                    break
    df = pd.DataFrame.from_records(result_records)
    print(df)
    df['iou_diff'] = df['pl_base_iou'] - df['pl_seg_gt_iou']
    # df.to_csv(RESULTS_DIR / "test.csv")
    print(df.mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Check LiDAR")
    parser.add_argument('--gui', dest='gui', action='store_true')
    parser.add_argument('--seg', dest='seg', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.gui, args.seg)
