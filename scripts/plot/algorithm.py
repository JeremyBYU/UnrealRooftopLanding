"""Example Multipolygon Extraction
"""
import time
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from airsimcollect.helper.helper_polylidar import extract_polygons

from fastga import GaussianAccumulatorS2Beta, IcoCharts
from polylidar import MatrixDouble, extract_tri_mesh_from_organized_point_cloud, HalfEdgeTriangulation, Polylidar3D, MatrixUInt8
from polylidar.polylidarutil import (plot_polygons_3d, generate_3d_plane, set_axes_equal, plot_planes_3d,
                                     scale_points, rotation_matrix, apply_rotation)

from airsimcollect.helper.helper_metrics import (update_projected_image, BLUE_NORM, GOLD_NORM, PURPLE_NORM,
                                                 load_map, select_building, load_map, load_records, compute_metric, update_state, get_inscribed_circle_polygon)

from airsimcollect.helper.helper_transforms import seg2rgb

colors_mapping = seg2rgb()



def filter_points(pc, camera_position, max_height=11):
    # Make point cloud centered on drone
    pc[:, :3] = pc[:, :3] - camera_position
    pc[:, 2] = -pc[:, 2]

    # only focus on rooftop
    mask = pc[:, 2] < -max_height
    pc[mask, :3] = np.nan

    # Focus on only these objects
    mask = (pc[:, 3] == 4) | (pc[:, 3] == 5) | (pc[:, 3] == 6)
    pc[~mask, :3] = np.nan

    return pc

def load_data(config, record_number=140, lidar_beams=64, vert_ds=2, hor_ds=2 ):
    base_path =  Path("./AirSimCollectData/LidarRoofManualTestDuplicate")
    records, _, _, _, _, _ = load_records(base_path)

    start_offset_unreal = np.array(records['start_offset_unreal'])
    airsim_settings = records.get('airsim_settings', dict())
    lidar_beams = airsim_settings.get('lidar_beams', 64)



    records_ = records['records']
    record = list(filter(lambda d: d['uid'] == record_number, records_))[0]

    img_meta = record['sensors'][0]
    update_state(img_meta)
    camera_position = img_meta['position'].to_numpy_array()

    lidar_path = base_path / "Lidar"
    pc_np = np.load(str(lidar_path / f"{record_number}-0-0.npy"))
    pc_np = filter_points(pc_np, camera_position)

    orig_points = pc_np.shape[0]
    num_cols = int(pc_np.shape[0] / lidar_beams)
    opc = pc_np.reshape((lidar_beams, num_cols, 4))

    opc = opc[::vert_ds, ::hor_ds, :]
    new_points = int(opc.shape[0] * opc.shape[1])
    pc_np = opc.reshape((new_points, 4))

    return pc_np, camera_position

def create_color_map():
    cmap = colors.ListedColormap(['k','b','y','g','r'])
    return cmap

def main():
    lidar_beams = 64
    vert_ds = 2
    hor_ds = 2
    new_lidar_beams = int(lidar_beams / vert_ds)

    cmap = create_color_map()

    # Load yaml file
    config = None
    with open('./assets/config/PolylidarParams.yaml') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing yaml")

    # Create Polylidar Objects
    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])


    pc_np, camera_position = load_data(config, 140, lidar_beams, vert_ds, hor_ds)

    # Polygon Extraction of surface
    # Only Polylidar3D
    pl_planes, alg_timings, _, _, _ = extract_polygons(pc_np, None, pl, ga,
                                                        ico, config, segmented=True, lidar_beams=new_lidar_beams, drone_pose=camera_position)

    # Show Point Cloud
    print("Should see point raw point cloud")
    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
                           subplot_kw=dict(projection='3d'))
    # plot points
    elev = 45
    azim = 45
    colors = colors_mapping(pc_np[:, 3])[:, :3]
    ax.scatter(*scale_points(pc_np[:, :3]), s=5.0, c=colors)
    set_axes_equal(ax)
    ax.view_init(elev=elev, azim=azim)
    fig.savefig("assets/imgs/Algorithm_pointcloud.pdf", bbox_inches='tight')
    fig.savefig("assets/imgs/Algorithm_pointcloud.png", bbox_inches='tight', pad_inches=-0.8)
    plt.show()


    # triangles = np.asarray(mesh.triangles)
    # all_planes = [np.arange(triangles.shape[0])]

    # # Show Triangulation
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
    #                        subplot_kw=dict(projection='3d'))

    # plot_planes_3d(points, triangles, all_planes, ax, alpha=0.0, z_value=-4.0)
    # plot_planes_3d(points, triangles, all_planes, ax, alpha=0.5)
    # # plot points
    # ax.scatter(*scale_points(points), s=0.1, c='k')
    # set_axes_equal(ax, ignore_z=True)
    # ax.set_zlim3d([-4, 6])
    # ax.view_init(elev=elev, azim=azim)
    # print("Should see triangulation point cloud")
    # fig.savefig("assets/scratch/Basic25DAlgorithm_mesh.pdf", bbox_inches='tight')
    # fig.savefig("assets/scratch/Basic25DAlgorithm_mesh.png", bbox_inches='tight', pad_inches=-0.8)
    # plt.show()

    # print("")
    # print("Should see two planes extracted, please rotate.")
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
    #                        subplot_kw=dict(projection='3d'))
    # # plot all triangles
    # plot_polygons_3d(points, polygons, ax)
    # plot_planes_3d(points, triangles, planes, ax)
    # # plot points
    # ax.scatter(*scale_points(points), c='k', s=0.1)
    # set_axes_equal(ax)
    # ax.view_init(elev=elev, azim=azim)
    # fig.savefig("assets/scratch/Basic25DAlgorithm_polygons.pdf", bbox_inches='tight')
    # fig.savefig("assets/scratch/Basic25DAlgorithm_polygons.png", bbox_inches='tight', pad_inches=-0.8)
    # plt.show()
    # print("")


if __name__ == "__main__":
    main()