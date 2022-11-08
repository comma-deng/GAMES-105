import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    joint_index_stack = []

    with open(bvh_file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('ROOT'):
                joint_name.append(line.split()[1])
            elif line.startswith('JOINT'):
                joint_name.append(line.split()[1])
            elif line.startswith('End'):
                joint_name.append(joint_name[joint_index_stack[-1]] + "_end")
            elif line.startswith("{"):
                joint_parent.append(joint_index_stack[-1] if len(joint_index_stack)>0 else -1)
                joint_index_stack.append(len(joint_name)-1)
            elif line.startswith("}"):
                joint_index_stack.pop()
            elif line.startswith("OFFSET"):
                offset = line.split()
                joint_offset.append([float(offset[1]), float(offset[2]), float(offset[3])])

    # print(joint_name)
    # print(joint_parent)
    # print(joint_offset)
    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = []
    joint_orientations = None
    joint_rot = []
    # the rotation of a node is parent.rotation * self.rotation
    # the postion of a node is parent.rotation * self.offset + parent.position
    motion_frame = motion_data[frame_id]
    joint_positions.append(motion_frame[0:3])
    joint_rot.append(R.from_euler('xyz', motion_frame[3:6], degrees=True) )
    data_idx=2
    for name_idx in range(1,len(joint_name)):
        cur_name = joint_name[name_idx]
        cur_rot_parent = joint_rot[joint_parent[name_idx]]
        cur_pos_parent = joint_positions[joint_parent[name_idx]]
        cur_position = cur_rot_parent.apply(joint_offset[name_idx]) + cur_pos_parent
        joint_positions.append(cur_position)
        if cur_name.endswith("_end"):
            joint_rot.append(joint_rot[joint_parent[name_idx]])
        else:
            cur_rot = R.from_euler('xyz', motion_frame[data_idx*3:data_idx*3+3], degrees=True)       
            joint_rot.append(cur_rot_parent * cur_rot)
            data_idx += 1

    # reshape joint_positions and joint_orientations
    joint_positions = [pos.reshape(1,-1) for pos in joint_positions]
    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = [rot.as_quat().reshape(1,-1) for rot in joint_rot]
    joint_orientations = np.concatenate(joint_orientations, axis=0)
    # print(joint_positions)
    # print(joint_orientations)
    return joint_positions, joint_orientations 


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    motion_data = None
    return motion_data