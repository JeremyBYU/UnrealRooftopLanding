"""Will check all the lidar returned
"""
import time
from pathlib import Path
import argparse
import joblib
import os

from descartes import PolygonPatch
import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import affine_transform

from scipy.spatial.transform import Rotation as R

from airsimcollect.helper.helper_logging import logger
from airsimcollect.helper.helper_transforms import get_seg2rgb_map, project_points_img

from airsimcollect.helper.helper_mesh import (create_meshes_cuda)
from airsimcollect.helper.helper_metrics import load_records, select_building, load_map, compute_metric
from airsimcollect.helper.o3d_util import create_frustum
from airsimcollect.helper.helper_confidence_maps import (create_fake_confidence_map_seg,
                                                         create_confidence_map_planarity,
                                                         create_confidence_map_combined,
                                                         get_homogenous_projection_matrices)
from airsimcollect.helper.helper_polylidar import extract_polygons


ROOT_DIR = Path(__file__).parent.parent
SAVED_DATA_DIR = ROOT_DIR / 'AirSimCollectData/LidarRoofManualTest'
PROCESSED_DATA_DIR = SAVED_DATA_DIR / 'Processed'
GEOSON_MAP = ROOT_DIR / Path("assets/maps/roof-lidar-manual.geojson")
O3D_VIEW = ROOT_DIR / Path("assets/o3d/o3d_view_default.json")
RESULTS_DIR = ROOT_DIR / Path("assets/results")


# def plot_polygons(polygons, points, ax, linewidth=2, shell_color='green', hole_color='orange'):
#     for poly in polygons:
#         shell_coords = poly.exteror
#         outline = Polygon(shell=shell_coords)
#         outlinePatch = PolygonPatch(outline, ec=shell_color, fill=False, linewidth=linewidth)
#         ax.add_patch(outlinePatch)

#         for hole_poly in poly.holes:
#             shell_coords = [get_point(pi, points) for pi in hole_poly]
#             outline = Polygon(shell=shell_coords)
#             outlinePatch = PolygonPatch(outline, ec=hole_color, fill=False, linewidth=linewidth)
#             ax.add_patch(outlinePatch)

def transform_points(points, hom_transform):
    temp = np.ones(shape=(4, points.shape[0]))
    temp[:3, :] = points.transpose()
    point_cam_ned = hom_transform.dot(temp)
    return point_cam_ned

def transform_ponts_raw(points, hom_transform):
    pass

def convert_points(points, hom, proj, width, height):

    cam_poly_points = transform_points(points, hom)
    pixels, _ = project_points_img(cam_poly_points, proj, width, height, None)
    return pixels


def rot_to_hom(rot, invert=False):
    rot_ = rot.T if invert else rot
    ans = np.identity(4)
    ans[:3,:3] = rot_
    return ans

def affine_mat(hom):
    return [hom[0,0], hom[0, 1], hom[0,2], hom[1, 0], hom[1, 1], hom[1,2], hom[2,0], hom[2,1], hom[2, 2], hom[0,3], hom[1, 3], hom[2,3]]
# def poly_rotate(rm, )

def main():
    records = sorted(os.listdir(PROCESSED_DATA_DIR), key=lambda x: 100 * int(x[:-4].split('-')[0]) + int(x[:-4].split('-')[1]) )
    for record_str in records:
        record = joblib.load(PROCESSED_DATA_DIR / record_str)
        conf_map_comb = record['conf_map_comb']
        hom = record['hom']
        proj = record['proj']
        poly = record['poly']
        poly_normal = record['poly_normal']
        rm, _ = R.align_vectors([[0, 0, 1]], poly_normal)
        rm = rot_to_hom(rm.as_matrix(), invert=True)

        # poly = affine_transform(poly, affine_mat(rm))
        poly_points = np.array(poly.exterior)

        # if in lidar local frame where xy is not flat
        new_hom = hom @ rm

        pixels = convert_points(poly_points, hom, proj, conf_map_comb.shape[0], conf_map_comb.shape[1])
        fig, ax = plt.subplots(nrows=1, ncols=2)
        conf_map_comb[pixels[:, 1], pixels[:, 0]] = 0.5

        ax[0].imshow(conf_map_comb)
        ax[1].add_patch(PolygonPatch(poly, ec='k', alpha=0.5, zorder=2),)
        ax[1].axis('equal')
        ax[1].scatter(poly_points[:, 0], poly_points[:, 1])

        plt.show()






def parse_args():
    parser = argparse.ArgumentParser(description="Check LiDAR")
    parser.add_argument('--gui', dest='gui', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main()



# Check the center, if less than 0.5, then take max of 4 corners, if all 4 corners bad then stop