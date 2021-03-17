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
from airsimcollect.helper.helper_transforms import parse_lidarData, classify_points, seg2rgb
from airsimcollect.helper.o3d_util import get_extrinsics, set_view, handle_shapes, update_point_cloud, translate_meshes
from airsimcollect.helper.helper_mesh import (
    create_meshes_cuda, update_open_3d_mesh_from_tri_mesh, decimate_column_opc, get_planar_point_density, map_pd_to_decimate_kernel)
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh
from airsimcollect.segmentation import DEFAULT_REGEX_CODES, set_segmentation_ids
from airsimcollect.helper.helper_metrics import get_inscribed_circle_polygon


from fastga import GaussianAccumulatorS2Beta, GaussianAccumulatorS2, IcoCharts

from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D

# Lidar Point Cloud Image
lidar_beams = 64

colors_mapping = seg2rgb()


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
    shell_color=[0.0, 1.0, 0.0, 1.0],
    hole_color=[1.0, 0.4, 0.0, 1.0],
    thickness=10.0,
    is_persistent=False,
    scale=100,
    ue4_origin=[0, 0, 0],
):
    lr_list = shapely_to_lr_list(poly, scale=scale, ue4_origin=ue4_origin, flatten=False, skip_conversion=True)
    conv_list = convert_list_to_vec3(lr_list['shell'])
    client.simPlotLineStrip(conv_list, shell_color, thickness, duration, is_persistent)
    for holes in lr_list['holes']:
        client.simPlotLineStrip(convert_list_to_vec3(holes), hole_color, thickness, duration, is_persistent)


def get_lidar_data(client: airsim.MultirotorClient):
    data: LidarData = client.getLidarData()
    points = parse_lidarData(data)
    # seg = np.array(data.segmentation, dtype=np.dtype('i4'))

    lidar_meta = dict(position=data.pose.position, rotation=data.pose.orientation)
    return points, lidar_meta

def get_numpy_array(response):
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb[:, :, [0, 2]] = img_rgb[:, :, [2, 0]]
    return img_rgb

def make_square(pixels, size=4, shape=(512, 512)):
    half_size = size//2
    new_pixels = []
    for i in range(pixels.shape[0]):
        col, row = pixels[i, 0], pixels[i, 1]
        start_row = row - half_size
        start_col = col - half_size
        for row in range(size + 1):
            new_row = start_row + row
            new_row = np.clip(new_row, 0, shape[0] - 1)
            for col in range(size+1):
                new_col = start_col + col
                new_col = np.clip(new_col, 0, shape[1] - 1)
                new_pixels.append([new_col, new_row])
    new_pixels = np.array(new_pixels)
    return new_pixels


def get_image_data(client: airsim.MultirotorClient, pc_np, airsim_settings,
                   img_type=airsim.ImageType.Segmentation, fname='scene.png',
                   ):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                                     airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
    seg_response = responses[1]
    img_meta = {"position": seg_response.camera_position, "rotation": seg_response.camera_orientation,
                "height": seg_response.height, "width": seg_response.width,
                "type": img_type}

    scene_img = get_numpy_array(responses[0])
    seg_img = get_numpy_array(seg_response)
    img_meta['data'] = seg_img

    point_classes, mask, lidar_pixels = classify_points(
        img_meta['data'], pc_np[:, :3], img_meta, airsim_settings)

    more_pixels = make_square(lidar_pixels)
    scene_img[more_pixels[:, 1], more_pixels[:, 0]] =  seg_img[more_pixels[:, 1], more_pixels[:, 0]] 

    # img1d = np.fromstring(response, dtype=np.uint8)
    # # TODO shape should be tuple
    # img_rgba = img1d.reshape(
    #     response.height, response.width, 4)
    from PIL import Image
    im = Image.fromarray(scene_img)
    im.save('scene.png')
    im = Image.fromarray(seg_img)
    im.save('segmentation.png')
    # plt.imshow(img_rgb)
    # plt.show()


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
    set_up_airsim(client)

    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

    ue4_origin = [0, 0, 90]
    ue4_origin = [2900, 4660, 2450]
    delay = 10
    prev_time = time.time() - delay
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info):
        while True:
            if time.time() - prev_time > delay:
                points, lidar_meta = get_lidar_data(client)
                get_image_data(client, points, air_sim_settings)
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
                break
                circle_poly, circle = get_inscribed_circle_polygon(planes[0][0], config['polylabel']['precision'])
                draw_polygon(client, planes[0][0], ue4_origin=ue4_origin, duration=delay, thickness=10)
                draw_polygon(client, circle_poly, ue4_origin=ue4_origin, duration=delay,
                             thickness=10, shell_color=[0.0, 0.0, 1.0, 1.0])
                print(planes)
                prev_time = time.time()
                


if __name__ == "__main__":
    main()
