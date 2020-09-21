"""AirSim Collection Script

"""

import sys
import json
import logging
import time
from pprint import pprint

import click
import numpy as np

from airsimcollect.helper.helper import update, update_collectors, DEFAULT_CONFIG
from airsimcollect import AirSimCollect

logger = logging.getLogger("AirSimCapture")
logger.setLevel(logging.INFO)

def validate_collect_config(config, segmentation_only=False):
    """Validates the configuration file

    Arguments:
        config {dict} -- Dictionary of configuration
    """
    if segmentation_only:
        points = np.array([])
    else:
        collection_points_file = config['collection_points']
        points = np.load(collection_points_file)
    return points


@click.group()
def cli():
    """Collects data from UE4 AirSim World"""
    pass


@cli.command()
@click.option('-c', '--config-file', type=click.Path(exists=True), required=True,
              help='File to configure collection')
@click.option('-so', '--segmentation-only', is_flag=True,
              help="Only apply segmentation codes")
@click.option('-d', '--debug', is_flag=True,
              help="Set debug mode. Verbose logs")
def collect(config_file, segmentation_only, debug):
    """Collects AirSim data"""

    if debug:
        logger.setLevel(logging.DEBUG)

    with open(config_file) as file:
        config = json.load(file)
    config = update(DEFAULT_CONFIG, config)
    config['collectors'] = update_collectors(config['collectors'])
    # pprint(config, indent=4)
    collection_points = None
    try:
        collection_points = validate_collect_config(config, segmentation_only)
        del config['collection_points']
    except Exception as e:
        click.secho("Error in validating config file", fg='red')
        logger.exception(e)
        sys.exit()
    if config.get('collection_point_names'):
        with open(config['collection_point_names']) as f:
            config['collection_point_names'] = json.load(f)

    num_collections = collection_points.shape[0]
    click.secho("Collecting {:d} data snapshots".format(num_collections))
    with click.progressbar(length=num_collections, label='Collecting data') as bar: # pylint: disable=C0102,
        asc = AirSimCollect(**config, collection_points=collection_points, bar=bar) # pylint: disable=E1132,
        start_time = time.time()
        records = asc.begin_collection()
        end_time = time.time()
    logger.info("%.2f seconds elapsed to take %d data snapshots", end_time - start_time, num_collections)
    # logger.info("%d data snapshots taken", num_collections)
