import numpy as np
import struct
from open3d import *

def bin_to_pcd(binFileName):
    size_float = 8
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


# source
bin_file = 'human.bin'

# convert
pcd_ = bin_to_pcd(bin_file)

# show
print(pcd_)
print(np.asarray(pcd_.points))
open3d.visualization.draw_geometries([pcd_])

# save
open3d.io.write_point_cloud('human.pcd', pcd_, write_ascii=False, compressed=False, print_progress=False)
