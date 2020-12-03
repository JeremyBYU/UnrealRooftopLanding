import open3d as o3d
import numpy as np
from .LineMesh import LineMesh
from airsimcollect.helper.helper_transforms import seg2rgb

ORANGE = (255/255, 188/255, 0)
GREEN = (0, 255/255, 0)

colors_mapping = seg2rgb()

def remove_nans(a):
    return a[~np.isnan(a).any(axis=1)]

def clear_polys(all_polys, vis):
    for line_mesh in all_polys:
        line_mesh.remove_line(vis, False)
    return []


def get_extrinsics(vis):
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    return camera_params.extrinsic


def set_view(vis, extrinsics):
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsics
    ctr.convert_from_pinhole_camera_parameters(camera_params)


def update_point_cloud(pcd, points):
    if points.ndim > 2:
        points = points.reshape((points.shape[0] * points.shape[1], 3))
    points_filt = points[~np.isnan(points).any(axis=1)]
    pcd.points = o3d.utility.Vector3dVector(points_filt)


def translate_meshes(meshes, shift_x=True):
    x_amt_ = 0
    y_amt_ = 0
    for i, mesh in enumerate(meshes):
        x_amt, y_amt, z_amt = mesh.get_axis_aligned_bounding_box().get_extent()
        if not shift_x:
            x_amt = 0.0
        else:
            y_amt = 0.0
        mesh.translate([x_amt_, y_amt_, 0])
        x_amt_ += x_amt
        y_amt_ += y_amt


def handle_shapes(vis, planes, obstacles, all_polys, line_radius=0.15):
    all_polys = clear_polys(all_polys, vis)
    for plane, _ in planes:
        points = np.array(plane.exterior)
        line_mesh = LineMesh(points, colors=GREEN, radius=line_radius)
        line_mesh.add_line(vis, False)
        all_polys.append(line_mesh)

    for plane, _ in obstacles:
        points = np.array(plane.exterior)
        line_mesh = LineMesh(points, colors=ORANGE, radius=line_radius)
        line_mesh.add_line(vis, False)
        all_polys.append(line_mesh)

    return all_polys


def init_vis(width=700, height=700):

    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Viewer", width, height)

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    # create empty polygons list
    line_meshes = []
    # Create axis frame
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # Create a mesh
    mesh = o3d.geometry.TriangleMesh()

    # add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(axis_frame)
    vis.add_geometry(mesh)

    geometry_set = dict(pcd=pcd, line_meshes=line_meshes, all_polys=[], axis_frame=axis_frame, mesh=mesh)

    return vis, geometry_set


def handle_linemeshes(vis, old_line_meshes, new_line_meshes):
    all_polys = clear_polys(old_line_meshes, vis)
    for line_mesh in new_line_meshes:
        line_mesh.add_line(vis)
    return new_line_meshes


def create_o3d_colored_point_cloud(pc_np, pcd=None):
    pc_vis = remove_nans(pc_np)
    label = pc_vis[:, 3].astype(np.int)

    colors = colors_mapping(label)[:, :3]
    if pcd is not None:
        pc = pcd
    else:
        pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_vis[:, :3])
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def create_linemesh_from_linear_ring(linear_ring, height=0, line_radius=0.15, rotate_func=None, color=GREEN):
    points = np.array(linear_ring)
    if points.ndim == 2:
        height_np = np.ones((points.shape[0], 1)) * height
        points = np.concatenate((points, height_np), axis=1)
    if rotate_func:
        points = rotate_func(points)
    return LineMesh(points, colors=color, radius=line_radius)

def create_linemesh_from_shapely(polygon, height=0, line_radius=0.15, rotate_func=None):
    all_line_meshes = [create_linemesh_from_linear_ring(polygon.exterior, height, line_radius, rotate_func, color=GREEN)]

    for hole in polygon.interiors:
        all_line_meshes.append(create_linemesh_from_linear_ring(hole, height, line_radius, rotate_func, color=ORANGE))

    return all_line_meshes
