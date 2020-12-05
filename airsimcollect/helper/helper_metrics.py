import numpy as np
from shapely.geometry import Polygon


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

def create_frustum_intersection(poly, frustum):
    points = frustum.points[1:, :2]
    ned_height = frustum.points[1, 2]
    frustum_plane = Polygon(points)
    gt_building_poly = frustum_plane.intersection(poly)
    if np.array(gt_building_poly.exterior).shape[1] == 2:
        return make_poly_3d(gt_building_poly, ned_height)
    else:
        return gt_building_poly


def select_building(features, point, pont_offset=10):
    plane_point = point + [0, 0, 10]
    dists = np.array([np.linalg.norm(feature['centroid'] - plane_point) for feature in features])
    index = np.argmin(dists)
    return features[index]