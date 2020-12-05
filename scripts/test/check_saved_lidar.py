"""Will check all the lidar returned
"""
import logging
import json
import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join
from functools import partial

import yaml
from rich import print as rprint
import open3d as o3d
import numpy as np
from shapely.geometry import shape, Polygon
import shapely
from airsimcollect.helper.LineMesh import LineMesh
from airsim.types import Vector3r, Quaternionr
import cv2

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.o3d_util import (get_extrinsics, set_view, handle_shapes, update_point_cloud,
                                           translate_meshes, handle_linemeshes, init_vis, clear_polys,
                                           update_o3d_colored_point_cloud, create_linemesh_from_shapely,
                                           update_frustum, load_view_point, save_view_point, toggle_visibility)
from airsimcollect.helper.helper_mesh import (create_meshes_cuda, update_open_3d_mesh_from_tri_mesh,
                                              decimate_column_opc, get_planar_point_density, map_pd_to_decimate_kernel)
from airsimcollect.helper.helper_metrics import create_frustum_intersection, select_polygon, select_building
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh

from fastga import GaussianAccumulatorS2Beta, GaussianAccumulatorS2, IcoCharts
from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D


directory = Path(
    r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\AirSimCollectData\LidarRoofManualTest")
geoson_map = Path(
    r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\assets\maps\roof-lidar-manual.geojson")

o3d_view = Path(
    r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\assets\o3d\o3d_view_default.json")

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


def extract_polygons(points_all, vis, mesh, all_polys, pl, ga, ico, config,
                     lidar_beams=64):
    points = points_all[:, :3]
    num_cols = int(points.shape[0] / lidar_beams)
    opc = points.reshape((lidar_beams, num_cols, 3))
    # 1. Create mesh
    alg_timings = dict()
    tri_mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
    alg_timings.update(timings)
    # 2. Get dominant plane normals
    avg_peaks, _, _, _, timings = extract_all_dominant_plane_normals(
        tri_mesh, ga_=ga, ico_chart_=ico, **config['fastga'])
    alg_timings.update(timings)
    # 3. Extract Planes and Polygons
    planes, obstacles, timings = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks, pl_=pl,
                                                                       filter_polygons=True, optimized=True,
                                                                       postprocess=config['polygon']['postprocess'])
    alg_timings.update(timings)
    # 100 ms to plot.... wish we had opengl line-width control
    all_polys.line_meshes = handle_shapes(
        vis, planes, all_polys.line_meshes, visible=all_polys.visible)
    # isec_polys = intersect_polys()

    # if update_mesh:
    #     update_open_3d_mesh_from_tri_mesh(mesh, tri_mesh)
    return planes


def main():
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict = load_records(
        directory)

    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        config = yaml.safe_load(file)

    start_offset_unreal = np.array(records['start_offset_unreal'])
    map_features_dict = load_map(geoson_map, start_offset_unreal)

    # Create Polylidar Objects
    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

    # Initialize 3D Viewer and Map
    vis, geometry_set = init_vis()
    line_meshes = [feature['line_meshes']
                   for features in map_features_dict.values() for feature in features]
    line_meshes = [
        line_mesh for line_mesh_set in line_meshes for line_mesh in line_mesh_set]
    geometry_set['map_polys'].line_meshes = handle_linemeshes(
        vis, geometry_set['map_polys'].line_meshes, line_meshes)

    load_view_point(vis, str(o3d_view))

    vis.register_key_callback(ord("X"), partial(
        toggle_visibility, geometry_set, 'pcd'))
    vis.register_key_callback(ord("C"), partial(
        toggle_visibility, geometry_set, 'map_polys'))
    vis.register_key_callback(ord("V"), partial(
        toggle_visibility, geometry_set, 'pl_polys'))
    vis.register_key_callback(ord("B"), partial(
        toggle_visibility, geometry_set, 'frustum'))
    # vis.register_key_callback(ord("C"), toggle_pcd_visibility)
    # vis.register_key_callback(ord("V"), toggle_pcd_visibility)

    for record in records['records']:
        if record['uid'] < 10:
            continue
        logger.info("Inspecting record; UID: %s; SUB-UID: %s",
                    record['uid'], record['sub_uid'])
        path_key = f"{record['uid']}-{record['sub_uid']}-0"
        bulding_label = record['label']  # building name
        # map feature of the building
        building_features = map_features_dict[bulding_label]
        camera_position = Vector3r(
            **record['sensors'][0]['position']).to_numpy_array()
        building_feature = select_building(building_features, camera_position)
        distance_to_camera = building_feature['ned_height'] - \
            camera_position[2]

        # Create Frustum
        update_frustum(vis, distance_to_camera, camera_position,
                       hfov=FOV, vfov=FOV,
                       frustum=geometry_set['frustum'])

        # Load Images
        img = cv2.imread(str(scene_paths_dict[path_key]))
        cv2.imshow('Scene View'.format(record['uid']), img)

        # Load Lidar Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))
        update_o3d_colored_point_cloud(pc_np, geometry_set['pcd'].geometry)

        # Polygon Extraction of surface
        pl_planes = extract_polygons(
            pc_np, vis, geometry_set['mesh'], geometry_set['pl_polys'], pl, ga, ico, config)

        # Create 3D shapely polygons of polylidar estimate and "ground truth" surface LIMITED to the sensor field of view of the camera frustum
        pl_poly_estimate = create_frustum_intersection(select_polygon(
            building_feature, pl_planes), geometry_set['frustum'].line_meshes[0])
        gt_poly = create_frustum_intersection(
            building_feature['polygon'], geometry_set['frustum'].line_meshes[0])

        iou = pl_poly_estimate.intersection(gt_poly).area / pl_poly_estimate.union(gt_poly).area
        # print(iou)

        # Update geometry and view
        vis.update_geometry(geometry_set['pcd'].geometry)
        vis.update_renderer()
        while(True):
            vis.poll_events()
            vis.update_renderer()
            res = cv2.waitKey(10)
            if res != -1:
                break


if __name__ == "__main__":
    main()
