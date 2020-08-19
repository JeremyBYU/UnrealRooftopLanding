# Sanity check to overwrite the label codes for scene meshes
import json
import airsim
from os import path

from airsimcollect.segmentation import set_segmentation_ids


BASE_DIR = path.dirname(path.dirname(__file__))
DATA_DIR = path.join(BASE_DIR, 'assets', 'data')
REGEXCODES_FILE = path.join(DATA_DIR, 'segmentation_codes.json')

# Load the segmentation codes
with open(REGEXCODES_FILE) as f:
    data = json.load(f)
    codes = data['segmentation_codes']

print("Loaded Segmentation Codes")
print(codes)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# Write the codes
set_segmentation_ids(client, codes)
# Release Control
client.enableApiControl(False)

