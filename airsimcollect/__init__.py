"""Defines AirSimCollect Class
"""

import logging
from os import path, makedirs
import sys
import time
import math
from itertools import count
from pprint import pprint, pformat
import json

import numpy as np
from PIL import Image

import airsim
from airsim import Vector3r, Pose, to_quaternion, ImageRequest

from airsimcollect.segmentation import set_all_to_zero, set_segmentation_ids
from airsimcollect.helper.helper import update, sensor_meta_data_json, update_airsim_settings

from airsimcollect.helper.helper_transforms import (
    parse_lidarData, create_projection_matrix, classify_points, get_seg2rgb_map)

from airsimcollect.helper.helper_logging import logger

class AirSimCollect(object):
    def __init__(
            self, name="AirSimCollector", sim_mode="ComputerVision", save_dir="AirSimData", collectors=None,
            segmentation_codes=[], collection_points=None, global_id_start=0, collector_file_prefix="", bar=None, collections_per_point=1,
            ignore_collision=False, collection_point_names=None, min_elapsed_time=0.01, color_codes=None, start_offset_unreal=[0, 0, 0]):
        self.name = name
        self.sim_mode = sim_mode
        self.save_dir = save_dir
        self.collectors = collectors
        self.collection_points = collection_points
        self.collector_file_prefix = collector_file_prefix
        self.collections_per_point = collections_per_point
        self.global_id_start = global_id_start
        self.segmentation_codes = segmentation_codes
        self.collect_data = len(self.collectors) > 0
        self.ignore_collision = ignore_collision
        self.collection_point_names = collection_point_names
        self.min_elapsed_time = min_elapsed_time
        self.color_codes = color_codes
        self.start_offset_unreal = start_offset_unreal
        self.bar = None if logger.getEffectiveLevel() == logging.DEBUG else bar
        self.airsim_settings = update_airsim_settings()
        

        if self.collection_points is None or self.collectors is None:
            logger.error(
                "Need collection points and collectors. Exiting early..")
            sys.exit()
        self.connect_airsim()
        self.prepare_collectors()
        self.global_id_counter = count(start=global_id_start)
        self.current_global_id = 0

        self.num_classes = len(
            set([code[1] for code in self.segmentation_codes]))
        if color_codes:
            _, self.seg2rgb_map = get_seg2rgb_map(
                self.num_classes, color_codes, normalized=False)
        else:
            self.seg2rgb_map = None

    def prepare_collectors(self):
        """Prepares Collectors for Data Collection"""
        set_segmentation_ids(self.client, self.segmentation_codes)
        for collector in self.collectors:
            collector['save_dir'] = path.normpath(
                path.join(self.save_dir, collector['type']))
            makedirs(collector['save_dir'], exist_ok=True)

    def connect_airsim(self):
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            return True
        except Exception as e:
            logger.exception(
                "Can not connect to AirSim vehicle! Is AirSim running? Exiting early")
            sys.exit()

    def get_next_global_id(self):
        return next(self.global_id_counter)

    def save_records(self, records=[], fname="records.json"):
        fpath = path.normpath(path.join(self.save_dir, fname))
        # import ipdb; ipdb.set_trace()
        # Have to put in a form that is JSON serializable
        ltcp = self.airsim_settings['lidar_to_camera_pos']
        ltcp = [ltcp.x_val, ltcp.y_val, ltcp.z_val]
        ltcq = self.airsim_settings['lidar_to_camera_quat']
        ltcq = [ltcq.w, ltcq.x, ltcq.y, ltcq.z]

        simple_airsim_settings = dict(lidar_local_frame=self.airsim_settings['lidar_local_frame'],
                                      lidar_to_camera_pos=ltcp, lidar_to_camera_quat=ltcq,
                                      lidar_z_col=self.airsim_settings['lidar_z_col'])

        records = dict(airsim_settings=simple_airsim_settings, start_offset_unreal=self.start_offset_unreal, records=records)
        with open(fpath, 'w') as outfile:
            json.dump(records, outfile, indent=2)

    def begin_collection(self):
        """Collect data. Iterate through collection points
        Notes - Position elements need to be divided by 100
                Z Position must be negated (NED coordinate System)
                Yaw (Theta) must be added with PI
                Pitch (Phi) must be subtracted by PI/2
        """
        num_collections = 0
        records = []
        for point in self.collection_points:
            point[:3] = point[:3] - self.start_offset_unreal
            # update progress bar if passed
            if self.bar:
                self.bar.update(1)
            # Get pos and rot
            pos = [point[0] / 100, point[1] / 100, - (point[2] / 100 + 1)]
            rot = [point[3], 0, point[5] + math.pi]
            logger.debug(
                "x: {:.2f}, y: {:.2f}, z: {:.2f}, pitch: {:.2f}, roll: {:.2f}, yaw: {:.2f}".format(*pos, *rot))
            # Begin Timer
            t0 = time.time()
            self.client.simSetVehiclePose(
                Pose(Vector3r(*pos), to_quaternion(*rot)), self.ignore_collision)
            # Check collision, only works with actual vehicles
            if not self.ignore_collision and self.client.simGetCollisionInfo().has_collided:
                logger.debug("Collision at point %r, skipping..", pos)
                continue

            elapsed = time.time() - t0
            if elapsed < self.min_elapsed_time:
                time.sleep(self.min_elapsed_time - elapsed)

            if self.collect_data:
                try:
                    self.current_global_id = self.get_next_global_id()
                    for sub_uid in range(self.collections_per_point):
                        record = self.collect_data_at_point(pos, rot, self.current_global_id, sub_uid)
                        records.append(record)
                except Exception:
                    logger.exception("Error collection data!")
                    records.append(uid=self.current_global_id, error=True)

            logger.debug("Time Elapsed: %.2f", elapsed)

        if records:
            self.save_records(records)
            # save records

        return records

    def get_file_name(self, global_id, sub_uid, sensor_id, ext):
        sensor_id_ = sensor_id if sensor_id != "" else "0"
        name = ""
        if self.collector_file_prefix:
            name = "{}-{}-{}-{}".format(self.collector_file_prefix,
                                     global_id, sub_uid, sensor_id_)
        else:
            name = "{}-{}-{}".format(global_id, sub_uid, sensor_id_)

        return "{}.{}".format(name, ext) if ext is not None else name

    def collect_images(self, image_requests, image_collectors, global_id, sub_uid):
        image_responses = self.client.simGetImages(image_requests)
        images_meta = []
        for i, response in enumerate(image_responses):
            img_collector = image_collectors[i]
            img_meta = {"sensor_name": img_collector['camera_name'], "sensor_type": 'camera',
                        "position": response.camera_position, "rotation": response.camera_orientation,
                        "height": response.height, "width": response.width,
                        "type": img_collector['type']}
            if response.pixels_as_float:
                file_path = path.join(img_collector['save_dir'], self.get_file_name(
                    global_id, sub_uid, img_collector['camera_name'], 'pfm'))
                airsim.write_pfm(file_path, airsim.get_pfm_array(response))
                logger.debug("Image Global ID: %d, Type %d, size %d, pos %s", global_id, response.image_type,
                             len(response.image_data_float), pformat(response.camera_position))
            else:
                file_path = path.join(img_collector['save_dir'], self.get_file_name(
                    global_id, sub_uid, img_collector['camera_name'], 'png'))
                if img_collector['compress']:
                    airsim.write_file(file_path, response.image_data_uint8)
                else:
                    img1d = np.fromstring(
                        response.image_data_uint8, dtype=np.uint8)
                    img_rgba = img1d.reshape(
                        (response.height, response.width, 3))
                    # bgr to rgb
                    img_rgba[:, :, [0, 2]] = img_rgba[:, :, [2, 0]]
                    img = Image.fromarray(img_rgba)
                    img.save(file_path, "PNG")
                # logger.("Image Global ID: %d, Type %d, size %d, pos %s", global_id, response.image_type,
                #              len(response.image_data_uint8), pformat(response.camera_position))
                # if we are asked to retain this data, possibly used for lidar data later
                if img_collector.get('retain_data'):
                    img_meta['data'] = img_rgba
            images_meta.append(img_meta)
        return images_meta

    def collect_lidar(self, collector, img_meta=None, global_id=1, sub_uid=0):

        # Concern, this lidar data is collected AFTER multiple image requests and saved to disk (IO bottleneck)
        # It may be out of date. Should we find a way to query the lidar data before getting the images?
        lidar_data = self.client.getLidarData(
            collector['lidar_name'], collector['vehicle_name'])
        if (len(lidar_data.point_cloud) < 3):
            logger.debug("No lidar points received")
            return None
        points = parse_lidarData(lidar_data)
        if points.size < 10:
            logger.warn("Missing lidar data")
        # Project points into segmentation image if available
        if collector['segmented']:
            if img_meta is None:
                logger.warn(
                    "Attempting to project lidar points but missing image data!")
                return

            # Transform and project point cloud into segmentation image
            cam_ori = img_meta['rotation']
            cam_pos = img_meta['position']
            height = img_meta['height']
            width = img_meta['width']

            point_classes, _, _ = classify_points(
                img_meta['data'], points, img_meta, self.airsim_settings)

            # proj_mat = create_projection_matrix(height, width)
            # # Transform NED points to camera coordinate system (not NED)
            # points_transformed = transform_to_cam(
            #     points, cam_pos, cam_ori, points_in_unreal=False)
            # # Project Points into image, filter points outside of image
            # pixels, points = project_points_img(
            #     points_transformed, proj_mat, width, height, points)
            # # Ensure we have valid points
            # if points.shape[0] < 1:
            #     logger.warn("No points for lidar in segmented image")
            #     return
            # color = get_colors_from_image(
            #     pixels, img_meta['data'], normalize=False)
            # # converts colors to numbered class
            # color = colors2class(color, self.seg2rgb_map)
            points = np.column_stack((points, point_classes))

        # Save point data as numpy
        if collector['save_as'] == 'numpy':
            file_path = path.join(collector['save_dir'], self.get_file_name(
                global_id, sub_uid, collector['lidar_name'], 'npy'))
            np.save(file_path, points)
        else:
            file_path = path.join(collector['save_dir'], self.get_file_name(
                global_id, sub_uid, collector['lidar_name'], 'csv'))
            np.savetxt(file_path, points, delimiter=',')

        lidar_meta = dict(sensor_type="lidar", sensor_name=collector['lidar_name'],
                          position=lidar_data.pose.position, rotation=lidar_data.pose.orientation)

        return lidar_meta

    def collect_lidars(self, lidar_collectors, images_meta, global_id, sub_uid):
        lidars_meta = []
        for collector in lidar_collectors:
            corresponding_camera = collector['camera_name']
            # Search for corresponding segmentation camera image
            camera_img_meta = next((item for item in images_meta if (
                item["sensor_name"] == corresponding_camera) and (item["type"] == 'Segmentation')), None)
            lidar_meta = self.collect_lidar(
                collector, camera_img_meta, global_id, sub_uid)
            if lidar_meta is not None:
                lidars_meta.append(lidar_meta)
        return lidars_meta

    def collect_data_at_point(self, pos, rot, global_id, sub_uid):
        """Collect data from each collector

        Arguments:
            pos {list} -- X,Y,Z
            rot {rot} -- pitch, roll, yaw
        """
        image_requests = []
        image_collectors = []
        lidar_collectors = []

        logger.debug("Global ID: %r; Sub ID: %s", global_id, sub_uid)
        for collector in self.collectors:
            if collector['sensor'] == 'Image':
                image_collectors.append(collector)
                image_requests.append(
                    ImageRequest(
                        collector['camera_name'],
                        collector['image_type'],
                        collector['pixels_as_float'],
                        collector['compress']))
            if collector['sensor'] == 'Lidar':
                lidar_collectors.append(collector)

        images_meta = self.collect_images(
            image_requests, image_collectors, global_id, sub_uid)
        lidars_meta = []
        if lidar_collectors:
            lidars_meta = self.collect_lidars(
                lidar_collectors, images_meta, global_id, sub_uid)

        sensor_meta_data = sensor_meta_data_json(images_meta, lidars_meta)
        label = self.collection_point_names[global_id] if self.collection_point_names else ''
        record = {"uid": global_id, "sub_uid": sub_uid,
                  'sensors': sensor_meta_data, 'label': label}
        return record
