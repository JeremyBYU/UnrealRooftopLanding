"""Will check all the lidar returned
"""
import logging
from rich.logging import RichHandler
import open3d as o3d
from pathlib import Path
from os import listdir
from os.path import isfile, join
import numpy as np
from airsimcollect.helper.helper_transforms import seg2rgb


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("UnrealLanding")

directory = Path(r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\AirSimCollectData\LidarRoofManualTest")

def remove_nans(a):
    return a[~np.isnan(a).any(axis=1)]

def natural_keys(text):
    text = text.split('.npy')[0]
    return float(text.split('-')[0]) + float(int(text.split('-')[1]) / 100)

lidar_directory = directory / Path("Lidar")
all_lidar_file_paths = [lidar_directory / f for f in sorted(listdir(lidar_directory), key=natural_keys)]

colors_mapping = seg2rgb()
for lidar_path in all_lidar_file_paths:
    log.info("Inspecting %s", lidar_path)
    pc_np = np.load(str(lidar_path))
    pc_vis = remove_nans(pc_np)
    label = pc_vis[:, 3].astype(np.int)

    colors = colors_mapping(label)[:, :3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_vis[:, :3])
    pc.colors =  o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc])



