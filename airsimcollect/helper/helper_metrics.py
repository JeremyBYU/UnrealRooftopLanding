import time
import numpy as np
from shapely.geometry import Polygon
from pathlib import Path
from shapely.geometry import shape, Polygon, Point, asPolygon
from os import listdir
import json
import shapely
import joblib
from airsim.types import Vector3r
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from polylabelfast import polylabelfast

from airsimcollect.helper.o3d_util import create_linemesh_from_shapely


def update_state(record, position='position', rotation='rotation'):
    position_data = record[position] if isinstance(
        record[position], list) else list(record[position].values())
    rotation_data = record[rotation] if isinstance(
        record[rotation], list) else list(record[rotation].values())
    record[position] = Vector3r(*position_data)
    record[rotation] = np.quaternion(*rotation_data)

def convert_dict(directory, suffix='.', required_extension=''):
    return {f.split('.')[0]: directory / f for f in listdir(directory) if f.endswith(required_extension)}

def tuple_to_list(tuplelist):
    return [np.array(row)[:2].tolist() for row in list(tuplelist)]

def add_column(array, z_value):
    ones = np.ones((array.shape[0], 1)) * z_value
    stacked = np.column_stack((array, ones))
    return stacked

def poly_to_rings(poly):
    ext_coord = tuple_to_list(list(poly.exterior.coords))
    holes = [tuple_to_list(ring.coords) for ring in poly.interiors]
    holes.insert(0, ext_coord)
    return holes

def get_inscribed_circle(polygon, precision=0.1):
    height = np.array(polygon.exterior.coords)[:, 2].mean()
    rings = poly_to_rings(polygon)
    t1 = time.perf_counter()
    point_, dist_ = polylabelfast(rings, precision)
    t2 = time.perf_counter()
    dt = (t2-t1) * 1000

    new_point = np.array([point_[0], point_[1], height])
    return dict(point=new_point, dist=dist_, t_polylabel=dt)

def get_inscribed_circle_polygon(polygon, precision=0.1):
    circle_dict = get_inscribed_circle(polygon, precision)
    p1 = Point(circle_dict['point'].tolist())
    p2 = p1.buffer(circle_dict['dist'], resolution=16)
    coords = np.array(p2.exterior.coords)
    coords = add_column(coords, circle_dict['point'][2])
    circle = asPolygon(coords)
    return circle, circle_dict

def found_hole(poly_gt, poly_pl:Polygon, remaining_pct=0.05):
    hole_gt = Polygon(poly_gt.interiors[0])
    num_holes = len(poly_pl.interiors)

    if num_holes < 1:
        return False, 1.0
    
    min_dist = 100000
    min_poly = None
    for index, hole in enumerate(poly_pl.interiors):
        hole_pl = Polygon(hole)
        dist = hole_pl.distance(hole_gt)
        if dist < min_dist:
            min_poly = hole_pl
            min_dist = dist
    
    remaining_poly = hole_gt.difference(min_poly)
    remaining = remaining_poly.area / hole_gt.area

    # print(remaining)
    # fig, ax = plt.subplots(1,1)
    # patch1 = PolygonPatch(hole_gt, fill=False)
    # patch2 = PolygonPatch(min_poly, fill=False, ec='k')
    # ax.add_patch(patch1)
    # ax.add_patch(patch2)
    # points = np.array(min_poly.exterior.coords)
    # ax.scatter(points[:, 0], points[:, 1])
    # plt.show()


    return remaining <= remaining_pct, remaining

def compute_metric(building_feature, pl_planes, frustum_points):
    # Create 3D shapely polygons of polylidar estimate and "ground truth" surface LIMITED to the sensor field of view of the camera frustum
    pl_poly_estimate = create_frustum_intersection(select_polygon(
        building_feature, pl_planes), frustum_points)
    gt_poly = create_frustum_intersection(
        building_feature['polygon'], frustum_points)

    base_iou = pl_poly_estimate.intersection(
        gt_poly).area / pl_poly_estimate.union(gt_poly).area


    return base_iou, pl_poly_estimate, gt_poly

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


def load_records(directory):
    lidar_directory = directory / Path("Lidar")
    scene_directory = directory / Path("Scene")
    segmentation_directory = directory / Path("Segmentation")
    segmentation_infer_directory:Path = directory / Path("SegmentationInfer")
    records_path = directory / 'records.json'

    with open(records_path) as f:
        records = json.load(f)

    lidar_paths_dict = convert_dict(lidar_directory)
    scene_paths_dict = convert_dict(scene_directory)
    segmentation_paths_dict = convert_dict(segmentation_directory)
    if segmentation_infer_directory.exists():
        seg_infer_paths_dict = convert_dict(segmentation_infer_directory, required_extension='.png')
        predictions = joblib.load(segmentation_infer_directory / "predictions.joblib")
        seg_infer_dict = { prediction['fname']: prediction['data'] for prediction in predictions}
    else:
        seg_infer_paths_dict = {}
        seg_infer_dict = {}

    return records, lidar_paths_dict, scene_paths_dict, segmentation_paths_dict, seg_infer_paths_dict, seg_infer_dict

def choose_dominant_plane_normal(avg_peaks, rooftop_normal=[0.0, 0.0, -1.0]):
    dots = np.array([avg_peaks[i, :].dot(rooftop_normal) for i in range(avg_peaks.shape[0])])
    index = np.argmax(dots)
    new_avg_peaks = np.array([avg_peaks[index,:]])
    return new_avg_peaks

def select_polygon(building_feature, pl_planes):
    building_poly = building_feature['polygon']
    areas = np.array([building_poly.intersection(pl_poly).area for pl_poly, _ in pl_planes])
    index = np.argmax(areas)
    return pl_planes[index][0]

def make_poly_3d(poly, height=0.0):
    points_exterior = np.array(poly.exterior)
    points_exterior = np.concatenate((points_exterior, np.ones((points_exterior.shape[0], 1)) * height), axis=1)
    points_interior_list = []
    for lr in poly.interiors:
        points_interior = np.array(lr)
        points_interior = np.concatenate((points_interior, np.ones((points_interior.shape[0], 1)) * height), axis=1)
        points_interior_list.append(points_interior)
    poly_3d = Polygon(shell=points_exterior, holes=points_interior_list)
    return poly_3d

def validate_polygon(poly):
    if poly.geom_type == 'MultiPolygon':
        areas = np.array([poly_.area for poly_ in poly.geoms])
        index = np.argmax(areas)
        return poly.geoms[index]
    else:
        return poly

def create_frustum_intersection(poly, frustum_points):
    points = frustum_points[1:, :2]
    ned_height = frustum_points[1, 2]
    frustum_plane = Polygon(points)
    gt_building_poly = frustum_plane.intersection(poly)
    gt_building_poly_ = validate_polygon(gt_building_poly)
    if np.array(gt_building_poly_.exterior).shape[1] == 2:
        return make_poly_3d(gt_building_poly_, ned_height)
    else:
        return gt_building_poly_


def select_building(features, point, pont_offset=10):
    plane_point = point + [0, 0, 10]
    dists = np.array([np.linalg.norm(feature['centroid'] - plane_point) for feature in features])
    index = np.argmin(dists)
    return features[index]