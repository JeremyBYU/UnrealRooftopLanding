from airsim.types import Quaternionr, Vector3r
import matplotlib.pyplot as plt
import matplotlib as mpl
import quaternion
import numpy as np
import logging
from os import path
import json
import warnings
import time

# ignore quaternions warning about numba not being installed
# ignore vispy warning about matplotlib 2.2+ issues
warnings.simplefilter("ignore")


LOGGER = logging.getLogger('AirSimVis')

BASE_DIR = path.dirname(path.dirname(path.dirname(__file__)))
DATA_DIR = path.join(BASE_DIR, 'assets', 'data')
SEG_RGB_FILE = path.join(DATA_DIR, 'seg_rgbs.txt')
TOLERANCE = 0.01


REGEX_CATCH_ALL = "[\w*. ]*"


def parse_lidarData(data):
    # reshape array of floats to array of [X,Y,Z]
    points = np.array(data.point_cloud, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0]/3), 3))

    return points


def tol(x, y):
    return (abs(x - y) < TOLERANCE)


def same_pose(op, np):
    if op is None:
        return False
    return (tol(op.x_val, np.x_val) and tol(op.y_val, np.y_val) and tol(op.z_val, np.z_val))


def seg2rgb(number_of_classes, cmap_file=SEG_RGB_FILE):
    cmap_list, _ = get_seg2rgb_map(number_of_classes, cmap_file)
    cmap, norm = create_cmap(cmap_list)

    def colors(values):
        return cmap(norm(values))
    return colors


def get_seg2rgb_map(number_of_classes, cmap_file=SEG_RGB_FILE, normalized=True):
    with open(cmap_file, 'r') as f:
        all_rows = f.read().splitlines()
    seg2rgb_map = {}
    alpha = 1 if normalized else 255
    for row in all_rows:
        seg = row.split('\t')[0]
        if normalized:
            rgb = list(map(lambda x: int(x)/255,
                           row.split('\t')[1][1:-1].split(',')))
        else:
            rgb = list(
                map(lambda x: int(x), row.split('\t')[1][1:-1].split(',')))
        rgb.append(alpha)
        seg2rgb_map[int(seg)] = rgb

    cmap_list = list(seg2rgb_map.values())[:number_of_classes]
    return cmap_list, seg2rgb_map


def colors2class(colors, seg2rgb_map):
    n_points = colors.shape[0]
    classes = np.zeros((n_points,), dtype=np.float32)
    for code, color in seg2rgb_map.items():
        mask = (colors == color)[:, 0]
        classes[mask] = code
    return classes


def project_ned_points(points, img_meta):
    # Transform and project point cloud into segmentation image
    cam_ori = img_meta['rotation']
    cam_pos = img_meta['position']
    height = img_meta['height']
    width = img_meta['width']
    proj_mat = create_projection_matrix(height, width)
    # Transform NED points to camera coordinate system (not NED)
    points_transformed = transform_to_cam(
        points, cam_pos, cam_ori, points_in_unreal=False)
    # Project Points into image, filter points outside of image
    pixels, points = project_points_img(
        points_transformed, proj_mat, width, height, points)
    return pixels


def classify_points(points, img_meta, img, seg2rgb_map):
    # Transform and project point cloud into segmentation image
    cam_ori = img_meta['rotation']
    cam_pos = img_meta['position']
    height = img_meta['height']
    width = img_meta['width']
    proj_mat = create_projection_matrix(height, width)
    # Transform NED points to camera coordinate system (not NED)
    points_transformed = transform_to_cam(
        points, cam_pos, cam_ori, points_in_unreal=False)
    # Project Points into image, filter points outside of image
    pixels, points = project_points_img(
        points_transformed, proj_mat, width, height, points)

    # Ensure we have valid points
    if points.shape[0] < 1:
        print("No points for lidar in segmented image")
        return None
    # Check if we have an RGBA image or just a 2D numpy array of classes
    remove_time = 0
    color = get_colors_from_image(pixels, img, normalize=False)
    if len(img.shape) > 2:
        # converts colors to numbered class
        t1 = time.time()
        color = colors2class(color, seg2rgb_map)
        remove_time = (time.time() - t1) * 1000

    points = np.column_stack((points, color))
    return points, remove_time


def create_cmap(cmap_list):
    number_of_classes = len(cmap_list)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Unreal', cmap_list, number_of_classes)
    # define the bins and normalize
    bounds = np.linspace(0, number_of_classes, number_of_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, number_of_classes)
    return cmap, norm


def height2rgb(height_range=[-25, 25], cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(height_range[0], height_range[1])

    def colors(values):
        return cmap(norm(values))

    return colors


def map_colors(values, cmap, norm):
    return cmap(norm(values))


def transform_to_cam(points, cam_pos, cam_ori, points_in_unreal=False):
    temp = points.copy()
    points = np.ones(shape=(4, points.shape[0]))
    points[:3, :] = temp.transpose()

    if points_in_unreal:
        # Need to scale down to meters
        points[:3, :] = points[:3, :] / 100.0
        # Need to convert to NED coordinate for homogoneous transformation matrix
        temp = points.copy()
        points[0, :], points[1, :], points[2,
                                           :] = temp[0, :], temp[1, :], -temp[2, :]

    # Points are in NED, wide form
    # Now transform them
    hom_transform = create_homogenous_transform(cam_pos, cam_ori)

    point_cam_ned = hom_transform.dot(points)
    # print(point_cam_ned)
    point_cam_hom = point_cam_ned.copy()
    point_cam_hom[0, :], point_cam_hom[1, :], point_cam_hom[2,
                                                            :] = point_cam_ned[1, :], point_cam_ned[2, :], point_cam_ned[0, :]
    # print(point_cam_hom)
    return point_cam_hom


def project_points_img(points, proj_mat, width, height, points_orig):
    pixels = proj_mat.dot(points)
    pixels = np.divide(pixels[:2, :], pixels[2, :]).transpose().astype(np.int)

    # Remove pixels that are outside the image
    mask_x = (pixels[:, 0] < width) & (pixels[:, 0] > 0)
    mask_y = (pixels[:, 1] < height) & (pixels[:, 1] > 0)

    # Return the pixels and points that are inside the image
    pixels = pixels[mask_x & mask_y]
    points_orig = points_orig[mask_x & mask_y, :]
    return pixels, points_orig


def create_homogenous_transform(cam_pos, rot, invert=True):

    i_ = -1.0 if invert else 1.0
    inv_rot_q = np.quaternion(
        rot.w_val, i_ * rot.x_val, i_ * rot.y_val, i_ * rot.z_val)
    cam_pos = np.array([cam_pos.x_val, cam_pos.y_val, cam_pos.z_val])

    inv_rot_mat = quaternion.as_rotation_matrix(inv_rot_q)

    hom_tran = np.zeros(shape=(4, 4))
    hom_tran[:3, :3] = inv_rot_mat
    hom_tran[:3, 3] = -1 * inv_rot_mat.dot(cam_pos) if invert else cam_pos
    hom_tran[3, 3] = 1

    return hom_tran


def get_colors_from_image(pixels, img, normalize=True):
    """Extract pixel values from img

    Arguments:
        pixels {ndarray (N,2)} -- Pixel (N,2) array, (x,y)
        img {ndarray (M,N,4)} -- Img, Y,X,RGBA

    Returns:
        [type] -- [description]
    """
    # Notice the flip in axes as well as dividing by 255.0 to give floats
    if normalize:
        colors = np.squeeze(img[pixels[:, 1], pixels[:, 0], :]) / 255.0
    else:
        if len(img.shape) > 2:
            colors = np.squeeze(img[pixels[:, 1], pixels[:, 0], :])
        else:
            colors = np.squeeze(img[pixels[:, 1], pixels[:, 0]])
    return colors


def create_projection_matrix(height, width):
    f = width / 2.0
    cx = width / 2.0
    cy = height / 2.0
    proj_mat = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0]])
    return proj_mat


def set_all_to_zero(client, code=0):
    found = client.simSetSegmentationObjectID(REGEX_CATCH_ALL, code, True)
    if not found:
        LOGGER.warning(
            "Segmentation - Could not find %s in Unreal Environment to set to code %r", REGEX_CATCH_ALL, code)


def set_segmentation_ids(client, regex_codes):
    for regex_str, code in regex_codes:
        found = client.simSetSegmentationObjectID(regex_str, code, True)
        if not found:
            LOGGER.warning(
                "Segmentation - Could not find %s in Unreal Environment to set to code %r", regex_str, code)


def get_segmentation_codes(file_name):
    with open(file_name) as f:
        data = json.load(f)
    seg_codes = data.get('segmentation_codes', [])
    return seg_codes

# def unreal_to_ned(x, y, z, pitch, roll, yaw):
#     pos = Vector3r(x / 100, y / 100, -z / 100)
#     rot = to_quaternion(pitch, roll, yaw)
#     return pos, rot
