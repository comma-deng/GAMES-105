import os
import numpy as np
import pickle
from answer_task1 import *
from smooth_utils import *

class MotionKey:
    def __init__(self) -> None:
        self.positions = np.zeros((6,3))
        self.rotations = np.zeros((6,4))
        self.velocities = np.zeros((6,3))
        self.avelocities = np.zeros((6,3))
        pass


bvh_folder_path = r'motion_material\kinematic_motion'

for file in os.listdir(bvh_folder_path):
    if not file.endswith('bvh'):
        continue
    print(file)
    motion = BVHMotion(os.path.join(bvh_folder_path, file))
    all_offset = [0,20,40,60,80,100]
    motion_keys = []
    for i in range(0, motion.motion_length-101):
        motion_key = MotionKey()
        for j in range(0, 6):
            offset = all_offset[j]
            motion_key.positions[j] = motion.joint_position[i+offset, 0, :]
            next_position = motion.joint_position[i+offset+1, 0, :]
            motion_key.velocities[j] = (next_position - motion_key.positions[j]) * 60
            motion_key.rotations[j] = motion.joint_rotation[i+offset, 0, :]
            motion_key.avelocities[j] = quat_to_avel(motion.joint_rotation[i+offset: i+offset+2, 0, :], 1/60)
        # print(motion_key.positions)
        # print(motion_key.velocities)
        # print(motion_key.rotations)
        # print(motion_key.avelocities)
        for j in range(0,6):
            motion_key.positions[5-j] -= motion_key.positions[0]
        motion_keys.append(motion_key)
    with open(os.path.join(bvh_folder_path, file.replace('.bvh', '.keys')), 'wb') as f:
        pickle.dump(motion_keys, f)