"""Will check all the lidar returned
"""
from pathlib import Path
from os import listdir
from os.path import isfile, join
from functools import partial
import argparse
import time

import yaml
# import matplotlib.pyplot as plt
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
                                           update_frustum, load_view_point, save_view_point, toggle_visibility, BLUE, PURPLE)
from airsimcollect.helper.helper_transforms import classify_points, polygon_to_pixel_coords
from airsimcollect.helper.helper_metrics import (update_projected_image, BLUE_NORM, GOLD_NORM, PURPLE_NORM,
                                                 load_map, select_building, load_map, load_records, compute_metric, update_state, get_inscribed_circle_polygon)
from airsimcollect.helper.helper_polylidar import extract_polygons

from fastga import GaussianAccumulatorS2Beta, IcoCharts
from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D, MatrixUInt8


ROOT_DIR = Path(__file__).parent.parent
SAVED_DATA_DIR = ROOT_DIR / 'AirSimCollectData/LidarRoofManualTest'
GEOSON_MAP = ROOT_DIR / Path("assets/maps/roof-lidar-manual.geojson")
RESULTS_DIR = ROOT_DIR / Path("assets/results")
O3D_VIEW = ROOT_DIR / Path("assets/o3d/o3d_view_default.json")
FOV = 90

# Nice Pictures UIDs - 31, 34, 38, 39, 42, 45
# Bad Segmentation Drone but still success - 40, 41,


def main(save_data_dir: Path, geoson_map, results_fname, gui=True, segmented=False, computer='desktop', save_images=True):
    records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict, seg_infer_path_dict, seg_infer_dict = load_records(
        save_data_dir)

    output_dir = (save_data_dir / "Output")
    output_dir.mkdir(parents=False, exist_ok=True)

    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        config = yaml.safe_load(file)

    start_offset_unreal = np.array(records['start_offset_unreal'])
    map_features_dict = load_map(geoson_map, start_offset_unreal)
    airsim_settings = records.get('airsim_settings', dict())
    lidar_beams = airsim_settings.get('lidar_beams', 64)
    range_noise = airsim_settings.get('range_noise', 0.05)
    # have to turn some json keys into proper objects, quaternions...
    update_state(airsim_settings, position='lidar_to_camera_pos',
                 rotation='lidar_to_camera_quat')

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
        vis.register_key_callback(ord(","), partial(
            toggle_visibility, geometry_set, 'circle_polys'))
        vis.register_key_callback(ord("U"), partial(
            save_view_point, filename='temp.json'))
        vis.register_key_callback(ord("Y"), partial(
            load_view_point, filename='temp.json'))
    else:
        geometry_set = dict(pl_polys=None)

    result_records = []
    for record in records['records']:
        path_key = f"{record['uid']}-{record['sub_uid']}-0"
        bulding_label = record['label']  # building name
        if record['uid'] in [5, 6, 25, 26, 27, 28, 29]:
            logger.warn("Skipping record; UID: %s; SUB-UID: %s; Building Name: %s. Rooftop assets don't match map. Rooftop assets randomness wasn't fixed on this asset!",
                        record['uid'], record['sub_uid'], bulding_label)
            continue
        # uid #45 is best segmentation example
        # if record['uid'] < 8:
        #     continue

        logger.info("Inspecting record; UID: %s; SUB-UID: %s; Building Name: %s",
                    record['uid'], record['sub_uid'], bulding_label)
        # Get camera data
        img_meta = record['sensors'][0]
        update_state(img_meta)
        camera_position = img_meta['position'].to_numpy_array()
        # map feature of the building
        building_features = map_features_dict[bulding_label]
        building_feature = select_building(building_features, camera_position)
        distance_to_camera = building_feature['ned_height'] - camera_position[2]

        # Load LiDAR Data
        pc_np = np.load(str(lidar_paths_dict[path_key]))
        # Load Images
        img_scene = cv2.imread(str(scene_paths_dict[path_key]))
        img_seg = cv2.imread(str(segmentation_paths_dict[path_key]))
        img_seg_infer = cv2.imread(str(seg_infer_path_dict[path_key]))
        img_meta['data'] = seg_infer_dict[path_key]
        # Combine all images
        img = np.concatenate((img_scene, img_seg, img_seg_infer), axis=1)

        # Update LIDAR Data to use inference from neural network
        pc_np_infer = np.copy(pc_np)
        t1 = time.perf_counter()
        point_classes, mask, pixels = classify_points(
            img_meta['data'], pc_np[:, :3], img_meta, airsim_settings)
        t2 = time.perf_counter()
        t_classify_pointcloud = (t2-t1) * 1000
        pc_np_infer[:, 3] = point_classes

        # Update projected image to have lidar data
        img_projected = np.copy(img_scene)

        # Points that define the camera FOV frustum
        frustum_points = create_frustum(
            distance_to_camera, camera_position, hfov=FOV, vfov=FOV)

        # Polygon Extraction of surface
        # Only Polylidar3D
        pl_planes, alg_timings, _, _, _ = extract_polygons(pc_np, geometry_set['pl_polys'] if not segmented else None, pl, ga,
                                                           ico, config, segmented=False, lidar_beams=lidar_beams, drone_pose=camera_position)

        # Polylidar3D with Perfect (GT) Segmentation
        pl_planes_seg_gt, alg_timings_seg, _, _, _ = extract_polygons(pc_np, geometry_set['pl_polys'] if segmented else None, pl, ga,
                                                                      ico, config, segmented=True, lidar_beams=lidar_beams, drone_pose=camera_position,
                                                                      prefilter=True)

        # Polylidar3D with Inferred (NN) Segmentation
        pl_planes_seg_infer, alg_timings_seg, _, _, _ = extract_polygons(pc_np_infer, geometry_set['pl_polys'] if segmented else None, pl, ga,
                                                                         ico, config, segmented=True, lidar_beams=lidar_beams, drone_pose=camera_position, 
                                                                         prefilter=True)
        alg_timings_seg.update(t_classify_pointcloud=t_classify_pointcloud)
        if pl_planes and True:
            base_iou, pl_poly_baseline, gt_poly = compute_metric(
                building_feature, pl_planes, frustum_points)
            seg_gt_iou, pl_poly_estimate_seg, _ = compute_metric(
                building_feature, pl_planes_seg_gt, frustum_points)
            seg_infer_iou, pl_poly_estimate_seg, _ = compute_metric(
                building_feature, pl_planes_seg_infer, frustum_points)
            logger.info("Polylidar3D Base IOU - %.1f; Seg GT IOU - %.1f; Seg Infer IOU - %.1f",
                        base_iou * 100, seg_gt_iou * 100, seg_infer_iou * 100)

            # Get Largest Inscribed Circle
            circle_poly, circle = get_inscribed_circle_polygon(pl_poly_estimate_seg, config['polylabel']['precision'])
            circle_poly_baseline, circle_baseline = get_inscribed_circle_polygon(
                pl_poly_baseline, config['polylabel']['precision'])
            alg_timings_seg.update(t_polylabel=circle['t_polylabel'])

            result_records.append(dict(uid=record['uid'], sub_uid=record['sub_uid'],
                                       building=bulding_label, pl_base_iou=base_iou,
                                       pl_seg_gt_iou=seg_gt_iou, pl_seg_infer_iou=seg_infer_iou,
                                       computer=computer,
                                       **alg_timings_seg))

            img_projected_baseline = np.copy(img_projected)
            # img_projected[pixels[:,1], pixels[:, 0]] = [0, 255,0]
            update_projected_image(img_projected, circle_poly, pl_poly_estimate_seg,
                                   gt_poly, pixels, img_meta, airsim_settings)
            update_projected_image(img_projected_baseline, circle_poly_baseline,
                                   pl_poly_baseline, gt_poly, pixels, img_meta, airsim_settings)
            # Visualize these intersections
            if gui:
                # Visualize the polylidar with segmentation results
                if segmented:
                    pl_poly_baseline = pl_poly_estimate_seg
                update_linemesh([pl_poly_baseline], geometry_set['pl_isec'])
                update_linemesh([gt_poly], geometry_set['gt_isec'], color=PURPLE)
                update_linemesh([circle_poly], geometry_set['circle_polys'], color=BLUE)
        elif gui:
            update_linemesh([], geometry_set['pl_isec'])
            update_linemesh([], geometry_set['gt_isec'])
            update_linemesh([], geometry_set['circle_polys'])

        if save_images:
            baseline_fname = output_dir / "{}-projected-baseline.png".format(path_key)
            semantic_fname = output_dir / "{}-projected-semantic.png".format(path_key)
            cv2.imwrite(str(baseline_fname), img_projected_baseline)
            cv2.imwrite(str(semantic_fname), img_projected)

        if gui:
            # Create Frustum
            update_frustum(vis, distance_to_camera, camera_position,
                           hfov=FOV, vfov=FOV,
                           frustum=geometry_set['frustum'])
            # Load Lidar Data
            update_o3d_colored_point_cloud(pc_np_infer, geometry_set['pcd'].geometry)
            # Update Drone Position
            geometry_set['drone'].geometry.translate(img_meta['position'].to_numpy_array(), relative=False)
            # Update Plot Images
            img = np.concatenate((img, img_projected, img_projected_baseline), axis=1)
            cv2.imshow('Scene View'.format(record['uid']), img)
            # Update geometry and view
            vis.update_geometry(geometry_set['pcd'].geometry)
            vis.update_geometry(geometry_set['drone'].geometry)
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
    df.to_csv(RESULTS_DIR / results_fname)
    print(df.mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Check LiDAR")
    parser.add_argument('--data', type=str, default=SAVED_DATA_DIR)
    parser.add_argument('--map', type=str, default=GEOSON_MAP)
    parser.add_argument('--results', type=str, default='results.csv')
    parser.add_argument('--gui', dest='gui', action='store_true')
    parser.add_argument('--seg', dest='seg', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.data), Path(args.map), args.results, args.gui, args.seg)
