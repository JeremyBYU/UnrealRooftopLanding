import argparse
import json
import logging
import time
from os import path

import numpy as np
from numpy.core.fromnumeric import squeeze
from airsim import Vector3r, Pose, to_quaternion, ImageRequest

from airsimcollect.helper.helper import update, update_collectors, DEFAULT_CONFIG
from airsimcollect import AirSimCollect
from airsimcollect.helper.helper_logging import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Decision Point Data")
    parser.add_argument('--config', type=str, help='Configuration file for collection',
                        default='./assets/config/collect_lidar_decision_point.json')
    args = parser.parse_args()
    return args


def setup_airsimcollect(config_file):
    with open(config_file) as file:
        config = json.load(file)
    config = update(DEFAULT_CONFIG, config)
    config['collectors'] = update_collectors(config['collectors'])
    asc = AirSimCollect(
        **config, manual_collect=True)  # pylint: disable=E1132,

    fpath = path.normpath(path.join(asc.save_dir, 'records.json'))
    if path.exists(fpath):
        with open(fpath, 'r') as outfile:
            data = json.load(outfile)
        try:
            global_id = data['records'][-1]['uid'] + 1
        except:
            global_id = 0
        records = data['records']
    else:
        records = []
        global_id = 0
    return asc, records, global_id


def create_square_pose(altitude=-5, pose=np.array([0, 0, 0]), square_size=4, grid_size=4):
    half_square = square_size / 2.0
    poses = []
    delta = square_size / grid_size
    for x in range(grid_size + 1):
        half_square_x = half_square - x * delta
        for y in range(grid_size + 1):
            half_square_y = half_square - y * delta
            new_pose = pose + \
                np.array([half_square_x, half_square_y, altitude])
            poses.append(new_pose.tolist())
    return poses


def main(config_file):
    heights = [5, 10, 15, 20, 25, 30]
    # heights= [5]
    asc, records, global_id = setup_airsimcollect(config_file)
    for height in heights:
        poses = create_square_pose(-height)
        for sub_id, pose in enumerate(poses):
            asc.client.simSetVehiclePose(
                Pose(Vector3r(*pose), to_quaternion(0, 0, 0)), True)
            time.sleep(asc.min_elapsed_time)
            extra_data = dict(lidar_beams=asc.airsim_settings['lidar_beams'],
                              range_noise=asc.airsim_settings['range_noise'],
                              horizontal_noise=asc.airsim_settings['horizontal_noise'],
                              height=height)
            record = None
            while record is None:
                try:
                    record = asc.collect_data_at_point(
                        global_id, sub_id, label='Building2_Example3', extra_data=extra_data)
                    records.append(record)
                except:
                    logger.exception("Error getting data from point, retrying..")
                    time.sleep(asc.min_elapsed_time)
        global_id += 1
    asc.save_records(records)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
