import time
import copy
import sys

import open3d as o3d
import numpy as np
import airsim
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import yaml

from airsimcollect.helper.helper_transforms import parse_lidarData
from airsimcollect.helper.o3d_util import get_extrinsics, set_view, handle_shapes, update_point_cloud, translate_meshes
from airsimcollect.helper.helper_mesh import (
    create_meshes_cuda, update_open_3d_mesh_from_tri_mesh, decimate_column_opc, get_planar_point_density, map_pd_to_decimate_kernel)
from airsimcollect.helper.helper_polylidar import extract_all_dominant_plane_normals, extract_planes_and_polygons_from_mesh


from fastga import GaussianAccumulatorS2Beta, IcoCharts

from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D

# Lidar Point Cloud Image
lidar_beams = 64

def set_up_aisim():
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)
    print("Taking off!")
    client.takeoffAsync(timeout_sec=3).join()
    print("Increasing altitude 10 meters!")
    client.moveToZAsync(-10, 2).join()
    print("Reached Altitude, launching Lidar Visualizer")

    return client


def get_lidar_data(client: airsim.MultirotorClient):
    data = client.getLidarData()
    points = parse_lidarData(data)
    return points


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

    client = set_up_aisim()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd = o3d.geometry.PointCloud()
    mesh_noisy = o3d.geometry.TriangleMesh()
    mesh_smooth = o3d.geometry.TriangleMesh()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_smooth)
    vis.add_geometry(axis)
    all_polys = []

    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

    path = [airsim.Vector3r(0, 0, -5), airsim.Vector3r(0, 0, -10)] * 10
    client.moveOnPathAsync(path, 2, 60)

    prev_time = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Info):
        while True:
            if time.time() - prev_time > 0.1:
                points = get_lidar_data(client)
                print(f"Full Point Cloud Size (including NaNs): {points.shape}")
                if np.count_nonzero(~np.isnan(points)) < 300:
                    continue
                # get columns of organized point cloud
                num_cols = int(points.shape[0] / lidar_beams)
                opc = points.reshape((lidar_beams, num_cols, 3))
                point_density = get_planar_point_density(opc)
                decimate_kernel = map_pd_to_decimate_kernel(point_density)
                print(
                    f"Planar point density: {point_density:.1f}; Decimate Kernel: {decimate_kernel}")
                # 0. Decimate
                opc_decimate, alg_timings = decimate_column_opc(
                    opc, kernel_size=decimate_kernel, num_threads=1)
                # 1. Create mesh
                tri_mesh, timings = create_meshes_cuda(
                    opc_decimate, **config['mesh']['filter'])
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
                all_polys = handle_shapes(vis, planes, obstacles, all_polys) # 100 ms to plot.... wish we had opengl line-width control
                # print(planes)
                # update the open3d geometries
                # update_point_cloud(pcd, points)
                update_open_3d_mesh_from_tri_mesh(mesh_smooth, tri_mesh)
                translate_meshes([mesh_smooth, pcd])
                prev_time = time.time()

            vis.update_geometry(pcd)
            vis.update_geometry(mesh_smooth)
            vis.poll_events()
            update_view(vis)
            vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    main()
