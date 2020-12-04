import open3d as o3d
import numpy as np
from .LineMesh import LineMesh
from airsimcollect.helper.helper_transforms import seg2rgb
from airsimcollect.helper.helper_logging import logger

ORANGE = (255/255, 188/255, 0)
GREEN = (0, 255/255, 0)

colors_mapping = seg2rgb()


def remove_nans(a):
    return a[~np.isnan(a).any(axis=1)]


def clear_polys(all_polys, vis):
    for line_mesh in all_polys:
        line_mesh.remove_line(vis, False)
    return []

def add_polys(all_polys, vis):
    for line_mesh in all_polys:
        line_mesh.add_line(vis, False)
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


# def handle_shapes(vis, planes, obstacles, all_polys, line_radius=0.15, visible=True):
#     all_polys = clear_polys(all_polys, vis)
#     for plane, _ in planes:
#         points = np.array(plane.exterior)
#         line_mesh = LineMesh(points, colors=GREEN, radius=line_radius)
#         if visible:
#             line_mesh.add_line(vis, False)
#         all_polys.append(line_mesh)

#     for plane, _ in obstacles:
#         points = np.array(plane.exterior)
#         line_mesh = LineMesh(points, colors=ORANGE, radius=line_radius)
#         if visible:
#             line_mesh.add_line(vis, False)
#         all_polys.append(line_mesh)

#     return all_polys


def handle_shapes(vis, planes, all_polys, line_radius=0.15, visible=True):
    # print(all_polys, visible)
    all_polys = clear_polys(all_polys, vis)
    for plane, _ in planes:
        lm = create_linemesh_from_shapely(plane)
        all_polys.extend(lm)
        if visible:
            add_polys(lm, vis)
    return all_polys

def init_vis(width=700, height=700):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Viewer", width, height)

    # create point cloud
    pcd = o3d.geometry.PointCloud()
    # create empty polygons list
    map_polys = []
    # Create axis frame
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    # Create a mesh
    mesh = o3d.geometry.TriangleMesh()
    # Create a LineMesh for the Frustum
    frustum = None
    # Create a Line Mesh for the interstion of the Frustum and predicted polygons
    isec_poly = None

    # add geometries
    vis.add_geometry(pcd)
    vis.add_geometry(axis_frame)
    # vis.add_geometry(mesh)

    geometry_set = dict(pcd=pcd, map_polys=map_polys, pl_polys=[
    ], axis_frame=axis_frame, mesh=mesh, frustum=frustum, isec_poly=isec_poly, vis_pcd=True,
        vis_map=True, vis_pl=True, vis_mesh=False, vis_frustum=True, vis_isec=False)

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
    if points.shape[1] == 2:
        height_np = np.ones((points.shape[0], 1)) * height
        points = np.concatenate((points, height_np), axis=1)
    if rotate_func:
        points = rotate_func(points)
    return LineMesh(points, colors=color, radius=line_radius)


def create_linemesh_from_shapely(polygon, height=0, line_radius=0.15, rotate_func=None):
    all_line_meshes = [create_linemesh_from_linear_ring(
        polygon.exterior, height, line_radius, rotate_func, color=GREEN)]

    for hole in polygon.interiors:
        all_line_meshes.append(create_linemesh_from_linear_ring(
            hole, height, line_radius, rotate_func, color=ORANGE))

    return all_line_meshes


def create_frustum(vis, dist_to_plane=5.0, start_pos=np.array([0.0, 0.0, 0.0]), hfov=90, vfov=90,
                   old_frustum=None, radius=0.15, color=[1, 0, 0], vis_frustum=True):
    if old_frustum is not None:
        old_frustum.remove_line(vis, reset_bounding_box=False)
        # clear_polys(old_frustum, vis)

    x = np.tan(np.radians(hfov/2.0)) * dist_to_plane
    y = np.tan(np.radians(vfov/2.0)) * dist_to_plane

    point0 = start_pos
    point1 = start_pos - [x, y, -dist_to_plane]
    point2 = start_pos - [-x, y, -dist_to_plane]
    point3 = start_pos - [-x, -y, -dist_to_plane]
    point4 = start_pos - [x, -y, -dist_to_plane]
    points = np.stack([point0, point1, point2, point3, point4])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]

    frustum = LineMesh(points, lines, colors=color, radius=radius)
    if vis_frustum:
        frustum.add_line(vis, reset_bounding_box=False)
    return frustum


def save_view_point(vis, filename=r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\assets\o3d\o3d_view_default.json"):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)


def load_view_point(vis, filename=r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\assets\o3d\o3d_view_default.json"):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)


def toggle_visibility(geometry_set, visibility_key, geometry_key, vis):
    """Toggles visibility of geometries by adding and removing to visualizer. Open3D has no opacity or visibility mechanism..."""
    geometry = geometry_set[geometry_key]
    if isinstance(geometry, list):
        # print("This is a list of LineMeshes")
        # make invisible by removing geometry
        if geometry_set[visibility_key]:
            clear_polys(geometry, vis)
        else:
            add_polys(geometry, vis)
        # toggle geometry
        geometry_set[visibility_key] = not geometry_set[visibility_key]
    elif issubclass(geometry.__class__, o3d.geometry.Geometry):
        # print("This is an Open3D geometry")
        # make invisible by removing geometry
        if geometry_set[visibility_key]:
            vis.remove_geometry(geometry, False)
        else:
            vis.add_geometry(geometry, False)
        # toggle geometry
        geometry_set[visibility_key] = not geometry_set[visibility_key]
    elif isinstance(geometry, LineMesh):
        # print("This is a LineMesh")
        if geometry_set[visibility_key]:
            geometry.remove_line(vis, False)
        else:
            geometry.add_line(vis, False)
        # toggle geometry
        geometry_set[visibility_key] = not geometry_set[visibility_key]
    elif geometry is None:
        pass
        # print("This is nothing")
    else:
        logger.info("Not able to handle this geometry type")

    logger.info("Toggled visibility of %s", geometry_key)
