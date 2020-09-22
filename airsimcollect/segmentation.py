"""Segmentation Module for airsim

"""
import logging
from os import path
import json

logger = logging.getLogger("AirSimCapture")
logger.setLevel(logging.DEBUG)

# airsim.wait_key('Press any key to set all object IDs to 0')
# found = client.simSetSegmentationObjectID("[\w*. ]*", 0, True);
# print("Done: %r" % (found))

REGEX_CATCH_ALL = "[\w*. ]*"

BASE_DIR = path.dirname(path.dirname(__file__))
DATA_DIR = path.join(BASE_DIR, 'assets', 'data')
REGEXCODES_FILE = path.join(DATA_DIR, 'segmentation_codes.json')
DEFAULT_REGEX_CODES = []
# Load the segmentation codes
with open(REGEXCODES_FILE) as f:
    data = json.load(f)
    DEFAULT_REGEX_CODES = data['segmentation_codes']

def set_all_to_zero(client, code=0):
    """Will set every mesh to the 0 id

    Args:
        client (AirSimClient): AirSimClient object
        code (int, optional): Code label for semantic segmentation. Defaults to 0.
    """
    found = client.simSetSegmentationObjectID(REGEX_CATCH_ALL, code, True)
    if not found:
        logger.warning(
            "Segmentation - Could not find %s in Unreal Environment to set to code %r", REGEX_CATCH_ALL, code)


def set_segmentation_ids(client, regex_codes):
    """Will take a list of RegexCodes and set them with the client

    Args:
        client (AirSimClient): AirSimClient object
        regex_codes (List[[regex string, int]]): A list of RegexCodes. A RegexCode is simply a list of [str, int].
    """
    for regex_str, code in regex_codes:
        found = client.simSetSegmentationObjectID(regex_str, code, True)
        if not found:
            logger.warning(
                "Segmentation - Could not find %s in Unreal Environment to set to code %r", regex_str, code)
