import time
import numpy as np
import cv2
import shapely
from skimage.transform import rescale
from astropy.convolution import convolve, Box2DKernel
from airsimcollect.helper.helper_transforms import colors2class, get_pixels_from_points, get_transforms, create_homogenous_transform, create_projection_matrix, affine_points_pixels
from shapely.geometry import Polygon


def create_fake_confidence_map_seg(img_seg, seg2rgb_map, ds=4, roof_class=4):
    img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
    if ds != 1:
        img_seg_rgb_ = rescale(img_seg_rgb, 1 / ds, multichannel=True,
                            mode='edge',
                            anti_aliasing=False,
                            anti_aliasing_sigma=None,
                            preserve_range=True,
                            order=0)
    else:
        img_seg_rgb_ = img_seg_rgb
    classes = colors2class(img_seg_rgb_, seg2rgb_map).astype(np.float32)
    classes[classes != roof_class] = 0.0
    classes[classes == roof_class] = 1.0

    return classes


def get_homogenous_projection_matrices(img_meta, airsim_settings, ds=4):
    img_meta_copy = dict(**img_meta)
    shape = np.array(
        (img_meta['height'], img_meta['width']), dtype=np.int) // ds
    img_meta_copy['height'] = shape[0]
    img_meta_copy['width'] = shape[1]
    transform_pos, transform_rot, invert = get_transforms(img_meta_copy, airsim_settings)
    hom_transform = create_homogenous_transform(cam_pos=transform_pos, cam_quat=transform_rot, invert=invert)
    proj_mat = create_projection_matrix(img_meta_copy['height'], img_meta_copy['width'])

    return hom_transform, proj_mat




def bounding_box_mask(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keeped or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

def bounding_box_points(points):
    box = []
    for dim in range(points.shape[1]):
        min_ = np.min(points[:, dim])
        max_ = np.max(points[:, dim])
        box.append(min_)
        box.append(max_)
    return box

def extended_bounding_box_points(points):
    box = bounding_box_points(points)
    size = [box[1] - box[0], box[3] - box[2]]

    # size of a quad tree box
    dim_index_min = np.argmin(size)
    dim_index_max = np.argmin(size)
    cell_size = size[dim_index_min]

    # additional space needed for longer dimension
    extra_size = size[dim_index_max]
    extra_boxes = 1
    while extra_size >=0:
        extra_size = extra_size - extra_boxes * cell_size
        extra_boxes += 1
    
    box[3] = box[2] + extra_boxes * cell_size
    return box

def bounding_box_polygon(poly: Polygon):
    points = np.asarray(poly.exterior)
    return bounding_box_points(points)

def extended_bounding_box_polygon(poly: Polygon):
    points = np.asarray(poly.exterior)
    return extended_bounding_box_points(points)

def create_affine_from_polygon(poly:Polygon, resolution=0.20):
    bbox = bounding_box_polygon(poly)
    affine = np.array([[bbox[0], bbox[2], resolution, resolution]], dtype=np.float64)
    return affine, bbox

def create_bbox_raster_from_polygon(poly:shapely, resolution=0.20):
    bbox = bounding_box_polygon(poly)
    size_world = [bbox[1] - bbox[0], bbox[3] - bbox[2]]
    size_pixels = [int(size_world[1] / resolution) + 1, int(size_world[0] / resolution) + 1]

    raster = np.full(size_pixels, np.nan, dtype=np.float32)
    x_off = bbox[0]
    y_off = bbox[2]
    x_scale = resolution
    y_scale = resolution
    affine = np.array([[x_off, y_off, x_scale, y_scale]], dtype=np.float64)
    # affine = np.zeros((3,3), dtype=np.float32)

    return raster, affine

def create_raster_from_bbox(bbox, resolution=0.20, value=np.nan):
    size_world = [bbox[1] - bbox[0], bbox[3] - bbox[2]]
    size_pixels = [int(size_world[1] / resolution) + 1, int(size_world[0] / resolution) + 1]
    raster = np.full(size_pixels, value, dtype=np.float64)
    return raster

def points_in_polygon(tri_mesh, poly, triangle_set, resolution=0.20):
    triangles_np = np.asarray(tri_mesh.triangles)
    vertices_np = np.asarray(tri_mesh.vertices)
    triangle_normals_np = np.asarray(tri_mesh.triangle_normals)
    triangles_planarity = np.abs(triangle_normals_np @ np.array([[0], [0], [-1]]))

    affine, bbox = create_affine_from_polygon(poly)
    bbox[4] = bbox[4] - 1 # z component
    bbox[5] = bbox[5] + 1 # z component
    # triangle_planarity_ = np.squeeze(triangles_planarity[triangle_set])
    # triangle_centroids_ = vertices_np[triangles_np[triangle_set, 0]]
    triangle_centroids = vertices_np[triangles_np[:, 0]]
    mask = bounding_box_mask(triangle_centroids, *bbox[:6])
    triangle_centroids_ = triangle_centroids[mask, :]
    triangle_planarity_ = np.squeeze(triangles_planarity[mask, :])

    size_world = [bbox[1] - bbox[0], bbox[3] - bbox[2]]
    size_pixels = [int(size_world[1] / resolution) + 1, int(size_world[0] / resolution) + 1]

    triangle_pixels, mask = affine_points_pixels(triangle_centroids_, affine, size_pixels[1], size_pixels[0])
    triangle_centroids_ = triangle_centroids_[mask, :]
    triangle_planarity_ = triangle_planarity_[mask]
    
    return dict(triangle_centroids=triangle_centroids_, triangle_pixels=triangle_pixels, poly_bbox=bbox, triangle_planarity=triangle_planarity_)


def create_confidence_map_planarity2(triangle_pixels, poly_bbox, triangle_planarity, **kwargs):
    box_kernel = Box2DKernel(3)
    raster_planarity = create_raster_from_bbox(poly_bbox)

    raster_planarity[triangle_pixels[:, 1], triangle_pixels[:, 0]] = triangle_planarity
    raster_planarity = convolve(raster_planarity, box_kernel) 
    raster_planarity[np.isnan(raster_planarity)] = 0.0

    return raster_planarity

def modify_img_meta(img_meta, ds=4):
    img_meta_copy = dict(**img_meta)
    img_meta_copy['height'] = img_meta_copy['height'] // ds
    img_meta_copy['width'] = img_meta_copy['width'] // ds
    return img_meta_copy

def create_confidence_map_segmentation(seg_img, img_meta, airsim_settings, triangle_centroids, triangle_pixels, poly_bbox, **kwargs):
    box_kernel = Box2DKernel(3)
    raster_seg = create_raster_from_bbox(poly_bbox)

    pixels_seg, mask = get_pixels_from_points(triangle_centroids, img_meta, airsim_settings) # slow
    seg_values = seg_img[pixels_seg[:, 1], pixels_seg[:, 0]]
    seg_img[pixels_seg[:, 1], pixels_seg[:, 0]] = 0.5
    triangle_pixels = triangle_pixels[mask, :]
    raster_seg[triangle_pixels[:, 1], triangle_pixels[:, 0]] = seg_values
    raster_seg = convolve(raster_seg, box_kernel) 
    raster_seg[np.isnan(raster_seg)] = 0.0
    return raster_seg


def create_confidence_map_planarity(tri_mesh, img_meta, airsim_settings, ds=4):
    img_meta_copy = dict(**img_meta)
    shape = np.array(
        (img_meta['height'], img_meta['width']), dtype=np.int) // ds
    img_meta_copy['height'] = shape[0]
    img_meta_copy['width'] = shape[1]
    conf_map_plan = np.full(
        (img_meta_copy['height'], img_meta_copy['width']), np.nan, dtype=np.float32)
    box_kernel = Box2DKernel(3)
    t1 = time.perf_counter()
    triangles_np = np.asarray(tri_mesh.triangles)
    vertices_np = np.asarray(tri_mesh.vertices)
    triangle_normals_np = np.asarray(tri_mesh.triangle_normals)
    t2 = time.perf_counter()
    triangle_centroids = vertices_np[triangles_np[:, 0]]
    # triangle_centroids = vertices_np[triangles_np].mean(axis=1) # slow, can be made faster
    t3 = time.perf_counter()
    triangles_planarity = triangle_normals_np @ np.array([[0], [0], [-1]])
    t4 = time.perf_counter()
    #
    pixels, mask = get_pixels_from_points(
        triangle_centroids, img_meta_copy, airsim_settings) # slow
    t5 = time.perf_counter()
    triangles_planarity_filt = np.squeeze(triangles_planarity[mask, :])
    t6 = time.perf_counter()
    conf_map_plan[pixels[:, 1], pixels[:, 0]] = triangles_planarity_filt
    conf_map_plan = convolve(conf_map_plan, box_kernel) 
    conf_map_plan[np.isnan(conf_map_plan)] = 0.0
    # conf_map_plan = uniform_filter(conf_map_plan, size=3)
    # grid_x, grid_y = np.mgrid[0:img_meta['height'], 0:img_meta['width']]
    # conf_map_plan = griddata(pixels, triangles_planarity_filt, (grid_y, grid_x), method='nearest')
    t7 = time.perf_counter()

    ms1 = (t2-t1) * 1000
    ms2 = (t3-t2) * 1000 # slow
    ms3 = (t4-t3) * 1000
    ms4 = (t5-t4) * 1000 # slow
    ms5 = (t6-t5) * 1000
    ms6 = (t7-t6) * 1000
    # print(ms1, ms2, ms3, ms4, ms5, ms6)

    return conf_map_plan


def create_confidence_map_combined(seg_map, conf_map):
    comb_map = (seg_map + conf_map) / 2.0
    return comb_map