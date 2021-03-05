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
import cv2

from airsimcollect.helper.o3d_util import create_linemesh_from_shapely
from airsimcollect.helper.helper_transforms import polygon_to_pixel_coords


# BGR Notation
ORANGE_NORM = [0,165,255]
BLUE_NORM = [255, 0, 0]
BRIGHT_GREEN_NORM = [0, 255, 0]
DARK_GREEN_NORM = [0, 112, 0]
GOLD_NORM = [0, 204, 255]
PURPLE_NORM = [177, 36, 199]

# Nice Pictures UIDs - 38, 39
# Bad Segmentation Drone but still success - 40

def create_star(x, y, size=10):
    coords = []
    for i in range(5):
        x_outer = size * np.cos(2 * np.pi * i/5.0  + np.pi/2.0)
        y_outer = size * np.sin(2 * np.pi * i/5.0  + np.pi/2.0)

        x_inner = size/2.0 * np.cos(2 * np.pi * i/5.0  + np.pi/2.0 + 2*np.pi/10.0)
        y_inner = size/2.0 * np.sin(2 * np.pi * i/5.0  + np.pi/2.0 + 2*np.pi/10.0)

        coords.append([x_outer, y_outer])
        coords.append([x_inner, y_inner])
    coords = np.array(coords)
    coords = coords + [x,y]
    return coords

def create_star_poly(x, y, size=10):
    star = create_star(x, y, size)
    return Polygon(star)

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if dist < gap:
        s = tuple(pt1)
        e = tuple(pt2)
        # print(s)
        cv2.line(img,s,e,color,thickness)
        return
    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawspecialpoly(img,coords,color,thickness=1,style='dotted',):
    pts = np.array(coords).tolist()
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def plot_opencv_polys(image, polygon, color_exterior=DARK_GREEN_NORM, color_interior=ORANGE_NORM, thickness=5, fill=True, style='solid'):
    pix_coords = np.array(polygon.exterior.coords).astype(np.int32)
    pix_coords_cv = pix_coords.reshape((-1, 1, 2))
    if fill:
        cv2.fillPoly(image, [pix_coords_cv], color_exterior)
    elif style == 'solid':
        cv2.polylines(image, [pix_coords_cv], False, color_exterior, thickness=thickness)
    else:
        drawspecialpoly(image, pix_coords, color_exterior, thickness, style=style)
    for hole in polygon.interiors:
        pix_coords = np.array(hole).astype(np.int32)
        pix_coords_cv = pix_coords.reshape((-1, 1, 2))
        if style == 'solid':
            cv2.polylines(image, [pix_coords], False, color_interior, thickness=thickness)
        else:
            drawspecialpoly(image, pix_coords, color_exterior, thickness, style=style)

def update_projected_image(img, circle_poly, pl_poly, gt_poly, lidar_pixels, img_meta, airsim_settings, ):
    circle_poly_pix = polygon_to_pixel_coords(circle_poly, img_meta, airsim_settings)
    pl_poly_pix = polygon_to_pixel_coords(pl_poly, img_meta, airsim_settings)
    gt_poly_pix = polygon_to_pixel_coords(gt_poly, img_meta, airsim_settings)
    star_poly_pix = create_star_poly(circle_poly_pix.centroid.x, circle_poly_pix.centroid.y)

    img[lidar_pixels[:,1], lidar_pixels[:, 0]] = [0, 255,0]
    # for i in range(lidar_pixels.shape[0]):
    #     cv2.circle(img,tuple(lidar_pixels[i,:]), 1, (0,255,0), -1)

    plot_opencv_polys(img, pl_poly_pix, fill=False)
    plot_opencv_polys(img, circle_poly_pix, color_exterior=BLUE_NORM, fill=False)
    plot_opencv_polys(img, gt_poly_pix, fill=False, color_exterior=PURPLE_NORM, style='dashed')
    plot_opencv_polys(img, star_poly_pix, color_exterior=GOLD_NORM, fill=True)

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