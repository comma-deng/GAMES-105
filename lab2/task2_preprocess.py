import os
import numpy as np
import pickle
from answer_task1 import *
from smooth_utils import *

bvh_folder_path = r'motion_material\kinematic_motion'
class MotionKey:
    def __init__(self, num) -> None:
        self.positions = np.zeros((num,6,3))
        self.rotations = np.zeros((num,6,4))
        self.velocities = np.zeros((num,6,3))
        self.avelocities = np.zeros((num,6,3))
        self.tracking_joints = ['lToeJoint', 'rToeJoint']
        self.joint_postions = np.zeros((len(self.tracking_joints), 3))
        self.joint_velocities = np.zeros((len(self.tracking_joints), 3))
        pass

def main():
    for file in os.listdir(bvh_folder_path):
        if not file.endswith('bvh'):
            continue
        print(file)
        motion = BVHMotion(os.path.join(bvh_folder_path, file))
        all_offset = [0,20,40,60,80,100]
        motion_key = MotionKey(motion.motion_length-101)
        joint_translation, joint_orientation = motion.batch_forward_kinematics()
        for i in range(0, motion.motion_length-101):
            # trajectory part
            for j in range(0, 6):
                offset = all_offset[j]
                motion_key.positions[i][j] = motion.joint_position[i+offset, 0, :]
                next_position = motion.joint_position[i+offset+1, 0, :]
                motion_key.velocities[i][j] = (next_position - motion_key.positions[i][j]) * 60
                motion_key.rotations[i][j] = motion.joint_rotation[i+offset, 0, :]
                motion_key.avelocities[i][j] = quat_to_avel(motion.joint_rotation[i+offset: i+offset+2, 0, :], 1/60)
            # print(motion_key.positions)
            # print(motion_key.velocities)
            # print(motion_key.rotations)
            # print(motion_key.avelocities)
            for j in range(0,6):
                motion_key.positions[i][5-j] -= motion_key.positions[i][0]

            #joint part
            for k in range(len(motion_key.tracking_joints)):
                joint = motion_key.tracking_joints[k]
                idx = motion.joint_name.index(joint)
                # calc joint position relative to root position
                motion_key.joint_postions[k] = joint_translation[i, idx, :] - joint_translation[i, 0, :]
                motion_key.joint_velocities[k] = (joint_translation[i+1, idx, :] - joint_translation[i, idx, :]) * 60

        with open(os.path.join(bvh_folder_path, file.replace('.bvh', '.keys')), 'wb') as f:
            pickle.dump(motion_key, f)

if __name__ == '__main__':
    main()