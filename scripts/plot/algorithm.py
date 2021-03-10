"""Example Multipolygon Extraction
"""
import time
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from stl import mesh
from shapely.geometry import Polygon
from descartes import PolygonPatch

from airsimcollect.helper.helper_polylidar import extract_polygons

from fastga import GaussianAccumulatorS2Beta, IcoCharts
from polylidar import Polylidar3D
from polylidar.polylidarutil import (set_axes_equal, plot_planes_3d, scale_points)

from airsimcollect.helper.helper_metrics import (load_records, update_state)

from airsimcollect.helper.helper_transforms import seg2rgb


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc, 3)
        arhsv[:, 1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:, 2] = np.linspace(chsv[2], 1, nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc, :] = rgb
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap


# colors_mapping = seg2rgb()
colors_mapping = categorical_cmap(8, 1, cmap='Dark2')


def filter_points(pc, camera_position, max_height=12):
    # Make point cloud centered on drone
    pc[:, :3] = pc[:, :3] - camera_position
    pc[:, 2] = -pc[:, 2]

    # only focus on rooftop
    mask = pc[:, 2] < -max_height
    pc[mask, :3] = np.nan

    # Focus on only these objects
    # mask = (pc[:, 3] == 4) | (pc[:, 3] == 5) | (pc[:, 3] == 6) | (pc[:, 3] == 7)
    # pc[~mask, :3] = np.nan

    return pc


def load_data(config, record_number=140, lidar_beams=64, vert_ds=2, hor_ds=2, start_x=0.05, end_x=0.8,
              start_y=0.10, end_y=0.90):
    base_path = Path("./AirSimCollectData/LidarRoofManualTestDuplicate")
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
    start_row_idx = int(start_y * opc.shape[0])
    end_row_idx = int(end_y * opc.shape[0])
    start_col_idx = int(start_x * opc.shape[1])
    end_col_idx = int(end_x * opc.shape[1])
    opc = opc[start_row_idx:end_row_idx, start_col_idx:end_col_idx, :]
    new_shape = opc.shape
    new_points = int(opc.shape[0] * opc.shape[1])
    pc_np = opc.reshape((new_points, 4))

    # temp_x = np.copy(pc_np[:, 0])
    # pc_np[:, 0] = pc_np[:, 1]
    # pc_np[:, 1] = temp_x

    return pc_np, camera_position, new_shape


def create_color_map():
    cmap = colors.ListedColormap(['k', 'b', 'y', 'g', 'r'])
    return cmap


def plot_polygons(polygons, ax, linewidth=2, shell_color='green', hole_color='orange', flip_xy=False):
    for poly, height in polygons:
        shell_coords = np.array(poly.exterior)
        if flip_xy:
            shell_coords[:,[1,0]] = shell_coords[:,[0,1]]
        outline = Polygon(shell=shell_coords)
        outlinePatch = PolygonPatch(outline, ec=shell_color, fill=False, linewidth=linewidth)
        ax.add_patch(outlinePatch)

        for hole_poly in poly.interiors:
            shell_coords = np.array(hole_poly)
            if flip_xy:
                shell_coords[:,[1,0]] = shell_coords[:,[0,1]]
            outline = Polygon(shell=shell_coords)
            outlinePatch = PolygonPatch(outline, ec=hole_color, fill=False, linewidth=linewidth)
            ax.add_patch(outlinePatch)


def plot_polygons_3d(polygons, ax, shell_color='green', hole_color='orange', linewidth=6):
    for poly, height in polygons:
        shell_coords = np.array(poly.exterior)
        ax.plot(shell_coords[:, 0], shell_coords[:, 1], shell_coords[:, 2],
                c=shell_color, linewidth=linewidth, label="Polygon Outline", alpha=1.0)
        for hole in poly.interiors:
            hole = np.array(hole)
            ax.plot(hole[:, 0], hole[:, 1], hole[:, 2], c=hole_color, linewidth=linewidth, alpha=1.0)


def main():
    lidar_beams = 64
    vert_ds = 4
    hor_ds = 4
    record_number = 140

    # Load yaml file
    config = None
    with open('./assets/config/PolylidarParams.yaml') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error parsing yaml")
    config['polylidar']['lmax'] = 2.5
    config['polylidar']['min_triangles'] = 100
    config['polygon']['postprocess']['positive_buffer'] = 0.0
    config['polygon']['postprocess']['negative_buffer'] = 0.0
    config['polygon']['postprocess']['simplify'] = 0.0
    # Create Polylidar Objects
    pl = Polylidar3D(**config['polylidar'])
    ga = GaussianAccumulatorS2Beta(level=config['fastga']['level'])
    ico = IcoCharts(level=config['fastga']['level'])

    drone_mesh = mesh.Mesh.from_file('assets/models/drone2.stl')
    drone_mesh.vectors = drone_mesh.vectors * 0.05
    # import ipdb; ipdb.set_trace()

    pc_np, camera_position, new_shape = load_data(config, record_number, lidar_beams, vert_ds, hor_ds)
    new_lidar_beams = new_shape[0]

    # Polygon Extraction of surface
    # Only Polylidar3D
    pl_planes, alg_timings, tri_mesh, avg_peaks, triangle_sets = extract_polygons(pc_np, None, pl, ga,
                                                                                  ico, config, segmented=True, lidar_beams=new_lidar_beams, drone_pose=camera_position)

    triangles = np.array(tri_mesh.triangles)
    all_planes = [np.arange(triangles.shape[0])]

    font = {'size': 14}
    matplotlib.rc('font', **font)
    elev = 33#40
    azim = 4#48
    colors = colors_mapping(pc_np[:, 3].astype(np.int))[:, :3]

    # # Show Point Cloud
    # print("Should see point raw point cloud")
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
    #                        subplot_kw=dict(projection='3d'))
    # # plot points
    # ax.scatter(*scale_points(pc_np[:, :3]), s=40.0, c=colors,  edgecolor="k")
    # set_axes_equal(ax)
    # ax.view_init(elev=elev, azim=azim)
    # ax.dist = 8
    # fig.savefig("assets/imgs/Algorithm_pointcloud.pdf", bbox_inches='tight')
    # fig.savefig("assets/imgs/Algorithm_pointcloud.png", bbox_inches='tight', pad_inches=-0.8)
    # plt.show()

    # Show Triangulation
    print("Should see triangulation point cloud")
    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1,
                           subplot_kw=dict(projection='3d'))

    plot_planes_3d(pc_np, triangles, all_planes, ax, alpha=0.1, edgecolor=(0, 0, 0, 0.7), linewidth=1.0)  # black mesh
    plot_planes_3d(pc_np, triangles, triangle_sets, ax, alpha=0.3,
                   edgecolor=(0, 0, 0, 0.0), linewidth=1.0)  # extracted segmetns
    ax.scatter(*scale_points(pc_np[:, :3]), s=25.0, c=colors, edgecolor="k", alpha=1.0)  # points
    set_axes_equal(ax, ignore_z=False)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim3d([-10, 5])
    ax.set_ylim3d([-5, 10])
    ax.set_zlim3d([-17, -3])
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    # 
    ax.view_init(elev=elev, azim=azim)
    # ax.dist = 8

    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(drone_mesh.vectors))  # edgecolors='k', linewidths=0.05
    fig.savefig("assets/imgs/Algorithm_mesh.pdf", bbox_inches='tight')
    plt.show()

    print("Should see polygon")
    fig, ax = plt.subplots(figsize=(3, 4), nrows=1, ncols=1)
    plot_polygons(pl_planes, ax, flip_xy=True)
    ax.scatter(pc_np[:, 1], pc_np[:, 2], alpha=0.0)
    ax.axis('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (m)', labelpad=1)
    ax.set_ylabel('Y (m)', labelpad=-5)
    fig.savefig("assets/imgs/Algorithm_polygon.pdf", bbox_inches='tight')
    plt.show()

    # # Show Triangulation
    # print("Should see Mesh and Polygons")
    # fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1,
    #                        subplot_kw=dict(projection='3d'))

    # plot_planes_3d(pc_np, triangles, all_planes, ax, alpha=0.1, edgecolor=(0, 0, 0, 0.7), linewidth=1.0) # black mesh
    # plot_planes_3d(pc_np, triangles, triangle_sets, ax, alpha=0.5, edgecolor=(0, 0, 0, 0.7), linewidth=1.0) # extracted segmetns
    # ax.scatter(*scale_points(pc_np[:, :3]), s=40.0, c=colors, edgecolor="k", alpha=1.0) # points
    # set_axes_equal(ax, ignore_z=False)
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')
    # ax.view_init(elev=elev, azim=azim)

    # plot_polygons_3d(pl_planes, ax)
    # # ax.dist = 8

    # ax.add_collection3d(mplot3d.art3d.Poly3DCollection(drone_mesh.vectors)) #  edgecolors='k', linewidths=0.05
    # fig.savefig("assets/imgs/Algorithm_mesh.pdf", bbox_inches='tight')
    # plt.show()

    # plot_planes_3d(pc_np, triangles, all_planes, ax, alpha=0.5)

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
