import time
import copy
import sys
import json
import os

import open3d as o3d
import numpy as np
import airsim
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import yaml
from airsim import ImageRequest, ImageResponse, LidarData, Vector3r
# from ue4poly import UE4Poly, create_draw_polygon_cmd, draw_polygon
# from ue4poly.types import DPCommand

from airsimcollect.helper.helper import update_airsim_settings
from airsimcollect.helper.helper_transforms import parse_lidarData
from airsimcollect.helper.o3d_util import get_extrinsics, set_view, handle_shapes, update_point_cloud, translate_meshes
from airsimcollect.helper.helper_mesh import (
    create_meshes_cuda, update_open_3d_mesh_from_tri_mesh, decimate_column_opc, get_planar_point_density, map_pd_to_decimate_kernel)
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh
from airsimcollect.segmentation import DEFAULT_REGEX_CODES, set_segmentation_ids


from fastga import GaussianAccumulatorS2Beta, GaussianAccumulatorS2, IcoCharts

from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D

# Lidar Point Cloud Image
lidar_beams = 64


def set_up_airsim(client):
    # connect to the AirSim simulator
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)
    print("Taking off!")
    client.takeoffAsync(timeout_sec=3).join()
    print("Increasing altitude 10 meters!")
    client.moveToZAsync(-10, 2).join()
    print("Reached Altitude, launching Lidar Visualizer")

    return client


def convert_to_ue4(points, scale=100, ue4_origin=[0, 0, 0], flatten=True, skip_conversion=False, **kwargs):
    """
    Converts NX3 Numpy Array in AirSim NED frame to a flattened UE4 frame 
    """
    if not skip_conversion:
        points = points * scale  # meters to cm
        points[:, -1] = -points[:, -1]  # AirSim NED z axis is inverted
        points = points + ue4_origin  # Shift to true unreal origin
    if flatten:
        points = points.flatten().tolist()  # Flatten data, return list for msgpack
    else:
        points = points.tolist()  # Flatten data, return list for msgpack
    return points


def shapely_to_lr_list(poly, **kwargs):
    shapely_list = dict(shell=convert_to_ue4(np.array(poly.exterior), **kwargs))
    holes = []
    for hole in poly.interiors:
        holes.append(convert_to_ue4(np.array(hole), **kwargs))
    shapely_list["holes"] = holes
    return shapely_list

def convert_list_to_vec3(points):
    return [Vector3r(*point) for point in points]

def draw_polygon(
    client,
    poly,
    duration=10.0,
    shell_color = [0.0, 1.0, 0.0, 1.0],
    hole_color = [1.0, 0.4, 0.0, 1.0],
    thickness=10.0,
    is_persistent=False,
    scale=100,
    ue4_origin=[0, 0, 0],
):
    lr_list = shapely_to_lr_list(poly, scale=scale, ue4_origin=ue4_origin, flatten=False, skip_conversion=True)
    conv_list = convert_list_to_vec3(lr_list['shell'])
    client.simPlotLineStrip(conv_list, shell_color, thickness, duration, is_persistent )
    for holes in lr_list['holes']:
        client.simPlotLineStrip(convert_list_to_vec3(holes), hole_color, thickness, duration, is_persistent )

def get_lidar_data(client: airsim.MultirotorClient):
    data:LidarData = client.getLidarData()
    points = parse_lidarData(data)
    # seg = np.array(data.segmentation, dtype=np.dtype('i4'))

    lidar_meta = dict(position=data.pose.position, rotation=data.pose.orientation)
    return points, lidar_meta


def get_image_data(client: airsim.MultirotorClient):
    response:ImageResponse = client.simGetImage("0", airsim.ImageType.Segmentation)

    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    # TODO shape should be tuple
    img_rgba = img1d.reshape(
        response.height, response.width, 4)
    plt.imshow(img_rgba)


def update_view(vis):
    extrinsics = get_extrinsics(vis)
    vis.reset_view_point(True)
    set_view(vis, extrinsics)

def main():
    # Load yaml file
    with open('./assets/config/PolylidarParams.yaml') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing yaml")
    client = airsim.MultirotorClient()
    set_segmentation_ids(client, DEFAULT_REGEX_CODES)

    air_sim_settings = update_airsim_settings()
    z_col = air_sim_settings['lidar_z_col']

    set_up_airsim(client)
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    # pcd = o3d.geometry.PointCloud()
    # mesh_noisy = o3d.geometry.TriangleMesh()
    # mesh_smooth = o3d.geometry.TriangleMesh()
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=1280, height=720)
    # vis.add_geometry(pcd)
    # # vis.add_geometry(pcd_pd)
    # vis.add_geometry(mesh_smooth)
    # vis.add_geometry(axis)
    # all_polys = []

    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])


    ue4_origin = [0, 0, 90]
    ue4_origin = [2900, 4660, 2450]

    prev_time = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info):
        while True:
            if time.time() - prev_time > 1.0:
                points, lidar_meta = get_lidar_data(client)
                # print(f"Full Point Cloud Size (including NaNs): {points.shape}")
                if np.count_nonzero(~np.isnan(points)) < 300:
                    continue
                # get columns of organized point cloud
                num_cols = int(points.shape[0] / lidar_beams)
                opc = points.reshape((lidar_beams, num_cols, 3))
                # # 1. Create mesh
                tri_mesh, timings = create_meshes_cuda(opc, **config['mesh']['filter'])
                # 2. Get dominant plane normals
                avg_peaks, _, _, _, timings = extract_all_dominant_plane_normals(
                    tri_mesh, ga_=ga, ico_chart_=ico, **config['fastga'])
                # print(avg_peaks)
                # 3. Extract Planes and Polygons
                planes, obstacles, timings = extract_planes_and_polygons_from_mesh(tri_mesh, avg_peaks, pl_=pl,
                                                                                   filter_polygons=True, optimized=True,
                                                                                   postprocess=config['polygon']['postprocess'])


                draw_polygon(client, planes[0][0], ue4_origin=ue4_origin, duration=1.0, thickness=10)
                # cmd = create_draw_polygon_cmd(planes[0][0], ue4_origin=ue4_origin, lifetime=0.15, thickness=10.0)
                # poly_client.draw_polygon(cmd)
                print(planes)
                prev_time = time.time()




if __name__ == "__main__":
    main()
