import time
import copy

import open3d as o3d
import numpy as np
import airsim

from airsimcollect.helper.helper_transforms import parse_lidarData
from airsimcollect.o3d_util import get_extrinsics, set_view

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

def main():
    client = set_up_aisim()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    
    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(axis)

    prev_time = time.time()
    while True:
        if time.time() - prev_time > 0.05:
            points = get_lidar_data(client)
            points = points[~np.isnan(points).any(axis=1)]
            pcd.points = o3d.utility.Vector3dVector(points)
            print(points.shape)
            prev_time = time.time()
        vis.update_geometry(pcd)
        vis.poll_events()
        update_view(vis)
        vis.update_renderer()
    vis.destroy_window()

if __name__ == "__main__":
    main()