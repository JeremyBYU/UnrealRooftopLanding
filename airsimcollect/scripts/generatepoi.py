import sys
import logging
from os import path
import math
from functools import partial
import json

import numpy as np
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import Point

import click
click.option = partial(click.option, show_default=True)

from airsimcollect.helper.helper import import_world, plot_collection_points
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("GeneratePoi")
logger.setLevel(logging.INFO)



def num_collection_points(yaw_range, yaw_delta, pitch_range, pitch_delta):

    # Whether to include endpoint of the yaw and pitch range
    yaw_endpoint = True
    theta_endpoint = True

    if yaw_delta < .01:
        num_yaw = 1
    else:
        num_yaw = int((abs(yaw_range[0] - yaw_range[1])) / yaw_delta) + 1
        # Dont sample the last 0 and 360
        if int(abs(yaw_range[0] - yaw_range[1])) == 360:
            num_yaw -= 1
            yaw_endpoint = False


    if pitch_delta is None or pitch_delta < .01:
        num_phi = 1
    else:
        num_phi = int((abs(pitch_range[0] - pitch_range[1])) / pitch_delta) + 1


    return num_yaw, num_phi, num_phi * num_yaw, yaw_endpoint


def remove_collision(collection_points, collisions):
    x = collection_points[:, 0]
    y = collection_points[:, 1]
    z = collection_points[:, 2]
    obstacle_mask = np.zeros_like(x, dtype='bool')
    for (minx, miny, maxx, maxy), height in collisions:
        z_m = z < height
        x_m = (x > minx) & (x < maxx)
        y_m = (y > miny) & (y < maxy)
        obstacle_mask = obstacle_mask | (z_m & x_m & y_m)
    return collection_points[~obstacle_mask]


def sample_circle(focus_point, radius, yaw_range, yaw_delta):
    num_yaw, num_phi, _, yaw_endpoint = num_collection_points(yaw_range, yaw_delta, None, None)
    theta = np.linspace(math.radians(
        yaw_range[0]), math.radians(yaw_range[1]), num_yaw, endpoint=yaw_endpoint)

    phi = np.zeros_like(theta)
    roll = np.zeros_like(theta)

    x = np.cos(theta) * radius + focus_point[0]
    y = np.sin(theta) * radius + focus_point[1]
    z = np.ones_like(phi) * focus_point[2]

    collection_points = np.stack((x, y, z, phi, roll, theta), axis=1)
    collection_points = np.append(collection_points,[[*focus_point, 0, 0, 0]], axis=0)

    return collection_points

def sample_sphere(focus_point, radius, pitch_range, pitch_delta, yaw_range, yaw_delta):
    num_yaw, num_phi, _, yaw_endpoint = num_collection_points(yaw_range, yaw_delta, pitch_range, pitch_delta)
    theta = np.linspace(math.radians(
        yaw_range[0]), math.radians(yaw_range[1]), num_yaw, endpoint=yaw_endpoint)
    phi = np.linspace(math.radians(
        pitch_range[0]), math.radians(pitch_range[1]), num_phi)

    theta = np.repeat(theta, num_phi)
    phi = np.tile(phi, num_yaw)
    roll = np.zeros_like(phi)

    x = np.cos(theta) * np.sin(phi) * radius + focus_point[0]
    y = np.sin(theta) * np.sin(phi) * radius + focus_point[1]
    z = np.cos(phi) * radius + focus_point[2]

    collection_points = np.stack((x, y, z, phi, roll, theta), axis=1)

    return collection_points


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []
    while len(points) < num_points:
        random_point = Point(
            [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points


def genereate_radii(feature, radius_min=0.0, radius_increase=0.0, num_spheres=1, radius_delta=200.0):
    """Generates a list of radii for collection spheres

    Arguments:
        feature {GeoJSON} -- GeoJSON Feature

    Keyword Arguments:
        radius_min {float} -- Minimum Radius. If 0 takes on different defaults (default: {0.0})
        num_spheres {int} -- Number of collection spheres (default: {1})
        radius_delta {float} -- How much to expand each radi from the previous (default: {200.0})

    Returns:
        list -- list of radi
    """

    radius_min_default_point = 500
    geom = feature['geometry']
    if geom.geom_type == 'Point' or geom.geom_type == 'LineString':
        radius_min_ = radius_min_default_point if radius_min == 0.0 else radius_min
        radius_min_ += radius_increase
    else:
        minx, miny, maxx, maxy = geom.bounds
        radius_geom = min(maxx - minx, maxy - miny) / 2.0
        radius_min_ = radius_geom if radius_min == 0.0 else radius_min
        radius_min_ += radius_increase
    return [radius_min_ + radius_delta * i for i in range(num_spheres)]


def generate_line_points(geom, num_focus_points):
    sections = len(geom.coords) - 1
    point_per_section = max(int(math.floor(num_focus_points / sections)), 1)
    x_points = []
    y_points = []

    for i, (x_prev, y_prev) in enumerate(geom.coords[:-1]):
        x_next, y_next = geom.coords[i + 1]
        x_points.append(np.linspace(x_prev, x_next, num=point_per_section, endpoint=False))
        y_points.append(np.linspace(y_prev, y_next, num=point_per_section, endpoint=False))

    # Must add the last point
    last_point = geom.coords[-1]
    x_points.append(np.array([last_point[0]]))
    y_points.append(np.array([last_point[1]]))
    # Flattten and combine data
    x = np.concatenate(x_points)
    y = np.concatenate(y_points)
    points = np.column_stack((x, y))
    return points


def generate_focus_points(feature, focus_point, num_focus_points, height_offset=0.0):
    geom = feature['geometry']
    height = feature['properties']['height'] + height_offset
    # Check if LineString Feature, return early
    if geom.geom_type == 'LineString':
        points = generate_line_points(geom, num_focus_points)
        return [[point[0], point[1], height] for point in points]

    # Point or Polygon Feature
    if geom.geom_type == 'Point':
        points = [geom]
    else:
        if focus_point == 'random':
            points = random_points_within(geom, num_focus_points)
        elif focus_point == 'centroid':
            points = [geom.centroid]
        elif focus_point == 'pia':
            points = [polylabel(geom)]

    return [[point.x, point.y, height] for point in points]


@click.group()
def cli():
    """Generates points of interest from geojson file from unreal world"""
    pass


@cli.command()
@click.option('-m', '--map-path', type=click.Path(exists=True), required=True,
              help='GeoJSON map file of points of interests (features) in the UE4 world.')
@click.option('-pr', '--pitch-range', nargs=2, type=float, default=[30, 90],
              help='Range in pitch (phi) on a collection sphere to sample each collection point')
@click.option('-pd', '--pitch-delta', type=float, default=15.0,
              help='Change in pitch angle (degrees) on collection sphere for each collection point')
@click.option('-yr', '--yaw-range', nargs=2, type=float, default=[0, 360],
              help='Range in yaw (theta) on a collection sphere to sample each collection point')
@click.option('-yd', '--yaw-delta', type=float, default=15.0,
              help='Change in yaw angle (degrees) on collection sphere for each collection point')
@click.option('-ho', '--height-offset', type=float, default=0.0,
              help='Add a height offset to each feature')
@click.option('-ns', '--num-spheres', type=int, default=1,
              help='Number of collection spheres to generate and sample from.')
@click.option('-rm', '--radius-min', type=float, default=0.0,
              help="Fixed minimum radius of collection sphere (distance from the focus point). " +
              "If 0 and map feature is a polygon, will use smallest sized circle to circumscribe polygon. " +
              "If 0 and map feature is a point, set to 500.")
@click.option('-ri', '--radius-increase', type=float, default=0.0,
              help="Increase (bias) from minimum radius of collection sphere (distance from the focus point). ")
@click.option('-rd', '--radius-delta', type=float, default=500.0,
              help='Change in growing collection sphere radius. Only applicable for -ns > 1.')
@click.option('-fp', '--focus-point', type=click.Choice(['pia', 'centroid', 'random']), default='centroid',
              help='Only applicable to polygon features. Determines what point on a 2D polygon ' +
              'should be used as the center of the collection sphere')
@click.option('-nf', '--num-focus-points', type=int, default=1,
              help='Number of focus points to randomly generate on 2D polygon. Only applicable to -fp random.')
@click.option('-rfn', '--record-feature-name', type=str, default=None,
              help='Set to geojson property name if you want to record a label associated to each point')
@click.option('-o', '--out', type=click.Path(exists=False), default="collection_points.npy",
              help="Output numpy array of position and angles")
@click.option('-ao', '--append-out', is_flag=True,
              help="If output file already exists, just append to it")
@click.option('--seed', type=int, default=1, help="Random seed")
@click.option('-ic', '--ignore-collision', is_flag=True,
              help="By default this module ensures the collection point does not collide with any known features " +
              "in the map. Set this flag to ignore this check.")
@click.option('-sc', '--sampling-method', type=click.Choice(['sphere', 'circle']), default='sphere',
              help='Whether we are sampling on a sphere or on a 2D circle at a height offset from the focus point')
@click.option('-pp', '--plot-points', is_flag=True,
              help="Whether to plot points for viewing. Debug only.")
@click.option('-d', '--debug', is_flag=True,
              help="Whether to print debug statements")
def generate(map_path, pitch_range, pitch_delta, yaw_range, yaw_delta, height_offset, num_spheres, radius_min, radius_increase, radius_delta,
             focus_point, num_focus_points, record_feature_name, out, append_out, seed, ignore_collision, sampling_method, plot_points, debug):

    if debug:
        logger.setLevel(logging.DEBUG)
    logger.debug("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
        map_path, pitch_range, pitch_delta, yaw_delta, num_spheres, radius_min, radius_increase, radius_delta, focus_point,
        num_focus_points, ignore_collision, out, seed, sampling_method, plot_points))

    click.secho("Generating collection points...")

    # generate_collection_points(map_path, pitch_range, pitch_delta, yaw_delta, num_spheres, radius_min, radius_delta, focus_point, ignore_collision)
    try:
        features, collisions = import_world(map_path)
    except Exception as e:
        click.secho("Error parsing GeoJSON file. Is it valid?", fg='red')
        logger.exception(e)
        sys.exit()

    all_points = []
    all_feature_names = []
    for feature in features:
        logger.debug("Inspecting feature: %s", feature)
        focus_points = generate_focus_points(feature, focus_point, num_focus_points, height_offset=height_offset)
        radii = genereate_radii(feature, radius_min, radius_increase, num_spheres, radius_delta)
        for focus_point_ in focus_points:
            logger.debug("At focus point: %s", focus_point_)
            for radius in radii:
                if sampling_method == 'sphere':
                    collection_points = sample_sphere(focus_point_, radius, pitch_range,
                                                    pitch_delta, yaw_range, yaw_delta)
                else:
                    collection_points = sample_circle(focus_point_, radius, yaw_range, yaw_delta)
                                                    
                logger.debug("At radius level: %s", radius)
                if not ignore_collision:
                    prev_shape = collection_points.shape
                    collection_points = remove_collision(collection_points, collisions)
                    if collection_points.shape != prev_shape:
                        logger.debug("Collisions removed for feature %r", feature['properties']['class_label'])
                
                all_points.append(collection_points)
                if record_feature_name:
                    all_feature_names.extend([feature['properties'][record_feature_name]] * collection_points.shape[0])
                if plot_points:
                    plot_collection_points(collection_points, focus_point_, radius, feature, sampling_method)

    all_points = np.vstack(all_points)
    click.echo(
        "Finished generating {:d} collection points for {:d} points of interests!".format(
            all_points.shape[0],
            len(features)))
    if append_out and path.isfile(out):
        old_data = np.load(out)
        all_points = np.vstack((old_data, all_points))
    np.save(out, all_points)
    if all_feature_names:
        out_feature_names = out[:-4] + '.json'
        with open(out_feature_names, 'w') as f:
            json.dump(all_feature_names, f, indent=2)
