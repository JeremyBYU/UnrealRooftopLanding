import time
import copy

import open3d as o3d
import numpy as np
import airsim

from airsimcollect.helper_transforms import parse_lidarData
from airsimcollect.o3d_util import get_extrinsics, set_view

from organizedpointfilters.utility.helper import (laplacian_opc, laplacian_then_bilateral_opc_cuda,
                                                    create_mesh_from_organized_point_cloud_with_o3d)


# Lidar Point Cloud Image
lidar_beams = 64

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

def update_point_cloud(pcd, points):
    if points.ndim > 2:
        points = points.reshape((points.shape[0] * points.shape[1], 3))
    points_filt = points[~np.isnan(points).any(axis=1)]
    pcd.points = o3d.utility.Vector3dVector(points_filt)


def translate_meshes(meshes, x_amt=0, y_amt=20):
    for i,mesh in enumerate(meshes):
        if i == 0:
            continue
        mesh.translate([x_amt, y_amt, 0])
    

def main():
    client = set_up_aisim()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    
    pcd = o3d.geometry.PointCloud()
    pcd_smooth = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_smooth)
    vis.add_geometry(axis)

    prev_time = time.time()
    while True:
        if time.time() - prev_time > 0.5:
            points = get_lidar_data(client)
            print(f"Full Point Cloud Size (including NaNs): {points.shape}")

            # get columns of organized point cloud
            num_cols = int(points.shape[0] / lidar_beams)
            opc = points.reshape((lidar_beams, num_cols, 3))
            # smooth organized pont cloud
            opc_smooth, opc_normals = laplacian_then_bilateral_opc_cuda(opc, loops_laplacian=1, loops_bilateral=2, sigma_angle=0.26, sigma_length=0.3)

            # update the open3d geometries
            update_point_cloud(pcd, points)
            update_point_cloud(pcd_smooth, opc_smooth)
            translate_meshes([pcd, pcd_smooth])
            
        vis.update_geometry(pcd)
        vis.update_geometry(pcd_smooth)
        vis.poll_events()
        update_view(vis)
        vis.update_renderer()
    vis.destroy_window()

if __name__ == "__main__":
    main()