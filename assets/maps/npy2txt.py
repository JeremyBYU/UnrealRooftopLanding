import numpy as np


pts = np.load('point_cloud.npy')
np.savetxt('point_cloud_manual_map.txt', pts)