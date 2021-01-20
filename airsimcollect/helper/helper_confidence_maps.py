import time
import numpy as np
import cv2
from skimage.transform import rescale
from astropy.convolution import convolve, Box2DKernel
from airsimcollect.helper.helper_transforms import colors2class, get_pixels_from_points, get_transforms, create_homogenous_transform, create_projection_matrix


def create_fake_confidence_map_seg(img_seg, seg2rgb_map, ds=4, roof_class=4):
    img_seg_rgb = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
    img_seg_rgb_ = rescale(img_seg_rgb, 1 / ds, multichannel=True,
                           mode='edge',
                           anti_aliasing=False,
                           anti_aliasing_sigma=None,
                           preserve_range=True,
                           order=0)
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