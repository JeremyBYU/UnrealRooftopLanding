"""Will check all the lidar returned
"""
import logging
import json
import sys
from pathlib import Path
from os import listdir
from os.path import isfile, join

import yaml
from rich.logging import RichHandler
from rich import print as rprint
import open3d as o3d
import numpy as np
from shapely.geometry import shape
import shapely
from airsimcollect.helper.LineMesh import LineMesh
import cv2

from airsimcollect.helper.o3d_util import get_extrinsics, set_view, handle_shapes, update_point_cloud, translate_meshes, handle_linemeshes, init_vis, clear_polys, create_o3d_colored_point_cloud, create_linemesh_from_shapely
from airsimcollect.helper.helper_mesh import (
    create_meshes_cuda, update_open_3d_mesh_from_tri_mesh, decimate_column_opc, get_planar_point_density, map_pd_to_decimate_kernel)
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh

from fastga import GaussianAccumulatorS2Beta, GaussianAccumulatorS2, IcoCharts
from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("UnrealLanding")

directory = Path(
    r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\AirSimCollectData\LidarRoofManualTest")
geoson_map = Path(r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\assets\maps\poi-roof-lidar-modified.geojson")



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
        polygon = shapely.affinity.translate(polygon, *(-1 * start_offset_unreal).tolist())
        # Scale polgyon from cm to meters
        polygon = shapely.affinity.scale(polygon, xfact=0.01, yfact=0.01, zfact=0.01, origin=(0, 0, 0))
        ned_height = -height/100.0 + start_offset_unreal[2] * .01
        # rprint(polygon)
        line_meshes = create_linemesh_from_shapely(polygon, ned_height)
        centroid = np.array([polygon.centroid.x, polygon.centroid.y])
        feature_data = dict(ned_height=ned_height, polygon=polygon, line_meshes=line_meshes, class_label=class_label, centroid=centroid)
        if class_label in features:
            features[class_label].append(feature_data)
        else:
            features[class_label] = [feature_data]

    return features


def extract_polygons(points_all, vis, mesh, all_polys, pl, ga, ico, config, lidar_beams=64):
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
    all_polys = handle_shapes(vis, planes, obstacles, all_polys)
    update_open_3d_mesh_from_tri_mesh(mesh, tri_mesh)
    return all_polys

def main():
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict = load_records(directory)

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
    line_meshes = [feature['line_meshes'] for features in map_features_dict.values() for feature in features]
    line_meshes = [line_mesh for line_mesh_set in line_meshes for line_mesh in line_mesh_set]
    geometry_set['line_meshes'] = handle_linemeshes(vis, geometry_set['line_meshes'], line_meshes)

    for record in records['records']:
        logger.info("Inspecting record; UID: %s; SUB-UID: %s",
                    record['uid'], record['sub_uid'])
        path_key = f"{record['uid']}-{record['sub_uid']}-0"
        bulding_label = record['label']

        # Load Images
        img = cv2.imread(str(scene_paths_dict[path_key]))
        cv2.imshow('Image View'.format(record['uid']), img)

        # Load Lidar Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))
        pcd = create_o3d_colored_point_cloud(pc_np, geometry_set['pcd'])

        # Polygon Extraction

        geometry_set['all_polys'] = extract_polygons(pc_np, vis, geometry_set['mesh'], geometry_set['all_polys'], pl, ga, ico, config)

        # Load GT Bulding Data
        # feature = map_features_dict[bulding_label]
        # geometry_set['line_meshes'] = handle_shapes(vis, geometry_set['line_meshes'], feature[0]['line_meshes'])
        
        # Update geometry and view
        vis.update_geometry(pcd)
        vis.reset_view_point(True)
        vis.update_renderer()
      
        while(True):
            vis.poll_events()
            vis.update_renderer()
            res = cv2.waitKey(10)
            # current_extrinsics = get_extrinsics(vis)
            if res != -1:
                break



if __name__ == "__main__":
    main()