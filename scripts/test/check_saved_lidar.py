import pptk
import numpy as np

data_file = r"C:\Users\Jeremy\Documents\UMICH\Research\UnrealRooftopLanding\AirSimCollectData\LidarRoofManualTest\Lidar\0-0.npy"
xyz = np.load(data_file)
v = pptk.viewer(xyz[:, :3])
label = xyz[:, 3].astype(np.uint64)
print(label)
v.attributes(label)
v.set(point_size=0.01)