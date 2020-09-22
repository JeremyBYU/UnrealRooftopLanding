import time
import copy
import sys

import open3d as o3d
import numpy as np
import airsim
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from airsimcollect.helper.helper_transforms import parse_lidarData
from airsimcollect.helper.o3d_util import get_extrinsics, set_view, update_point_cloud, translate_meshes
from airsimcollect.helper.helper_mesh import decimate_column_opc
from organizedpointfilters.utility.helper import (laplacian_opc, laplacian_then_bilateral_opc_cuda,
                                                  create_mesh_from_organized_point_cloud_with_o3d)

from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation


COLOR_PALETTE = list(
    map(colors.to_rgb, plt.rcParams['axes.prop_cycle'].by_key()['color']))

# Lidar Point Cloud Image
lidar_beams = 64


def pick_valid_normals(opc_normals):
    # I think that we need this with open3d 0.10.0
    mask = ~np.isnan(opc_normals).any(axis=1)
    tri_norms = np.ascontiguousarray(opc_normals[mask, :])
    return tri_norms


def set_up_aisim():
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.armDisarm(True)
    print("Taking off!")
    client.takeoffAsync().join()
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


def update_mesh(mesh, opc, opc_normals=None):
    opc_new = opc.astype('f8')
    opc_ = opc_new
    if opc_.ndim == 3:
        rows = opc_new.shape[0]
        cols = opc_new.shape[1]
        stride = 1
        opc_ = opc_new.reshape((rows * cols, 3))

    pcd_mat = MatrixDouble(opc_)
    tri_mesh, tri_map = extract_tri_mesh_from_organized_point_cloud(
        pcd_mat, rows, cols, stride)

    # Have to filter out nan vertices and normals
    mask = np.isnan(opc_).any(axis=1)
    opc_[mask, :] = [0, 0, 0]

    # Update the mesh
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array(tri_mesh.triangles, dtype=int))
    mesh.vertices = o3d.utility.Vector3dVector(opc_)
    mesh.paint_uniform_color(COLOR_PALETTE[0])
    mesh.compute_vertex_normals()

    if opc_normals is not None:
        opc_normals = pick_valid_normals(opc_normals)
        mesh.triangle_normals = o3d.utility.Vector3dVector(opc_normals)
    else:
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()


def main():
    client = set_up_aisim()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd = o3d.geometry.PointCloud()
    # pcd_smooth = o3d.geometry.PointCloud()
    mesh_noisy = o3d.geometry.TriangleMesh()
    mesh_smooth = o3d.geometry.TriangleMesh()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_noisy)
    vis.add_geometry(mesh_smooth)
    vis.add_geometry(axis)

    path = [airsim.Vector3r(0, 0, -5),airsim.Vector3r(0, 0, -10)] * 20
    client.moveOnPathAsync(path, 2, 120)

    prev_time = time.time()
    while True:
        if time.time() - prev_time > 0.05:
            points = get_lidar_data(client)
            print(f"Full Point Cloud Size (including NaNs): {points.shape}")

            # get columns of organized point cloud
            num_cols = int(points.shape[0] / lidar_beams)
            opc = points.reshape((lidar_beams, num_cols, 3))
            opc_decimate, timings = decimate_column_opc(opc, kernel_size=2, num_threads=1)
            # smooth organized point cloud
            opc_smooth, opc_normals = laplacian_then_bilateral_opc_cuda(
                opc_decimate, loops_laplacian=1, _lambda=0.5, loops_bilateral=4, sigma_angle=0.2, sigma_length=0.3)
            opc_smooth = opc_smooth.reshape((lidar_beams, opc_decimate.shape[1], 3))

            # update the open3d geometries
            update_point_cloud(pcd, points)
            update_mesh(mesh_noisy, opc)
            update_mesh(mesh_smooth, opc_smooth, opc_normals)
            translate_meshes([pcd, mesh_noisy, mesh_smooth])
            prev_time = time.time()

        vis.update_geometry(pcd)
        vis.update_geometry(mesh_noisy)
        vis.update_geometry(mesh_smooth)
        vis.poll_events()
        update_view(vis)
        vis.update_renderer()
    vis.destroy_window()


if __name__ == "__main__":
    main()
