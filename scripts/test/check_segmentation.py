# Sanity check to overwrite the label codes for scene meshes
import json
import airsim
from os import path

from airsimcollect.segmentation import set_segmentation_ids, DEFAULT_REGEX_CODES



print("Loaded Segmentation Codes")
print(DEFAULT_REGEX_CODES)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# Write the codes
set_segmentation_ids(client, DEFAULT_REGEX_CODES)
# Release Control
client.enableApiControl(False)

