import open3d as o3d
import numpy as np
from .LineMesh import LineMesh

ORANGE = (255/255, 188/255, 0)
GREEN = (0, 255/255, 0)


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


def translate_meshes(meshes, x_amt=0, y_amt=20):
    x_amt_ = 0
    y_amt_ = 0
    for i, mesh in enumerate(meshes):
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
