import numpy as np
from scipy.spatial.transform import Rotation as R

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    idx_end = meta_data.joint_name.index(meta_data.end_joint)
    it = 0
    max_it = 1e4
    alpha = 1e-4
    #这里假定了第一个一定是根节点
    joint_eulers = []
    joint_offset = []
    joint_path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_parent_new = meta_data.joint_parent.copy()
    for i in range(len(path2)-1):
        idx_l = path2[i]
        idx_r = path2[i+1]
        joint_parent_new[idx_r] = idx_l
    joint_parent_new[path2[0]] = -1
    end_pos = joint_positions[idx_end]
    for i in range(len(joint_offset)):
        joint_offset.append(meta_data.joint_initial_position[i] -  meta_data.joint_initial_position[joint_parent_new[i]] if joint_parent_new[i]>=0 else meta_data.joint_initial_position[i])
        if joint_parent_new[i]>=0:
            Q_p = R.from_quat(joint_orientations[joint_parent_new[i]])
            Q_cur = R.from_quat(joint_orientations[i])
            relative_euler = (Q_p.inv() * Q_cur).as_euler()
        else:
            relative_euler = R.from_euler("XYZ",[0,0,0])
        joint_eulers.append(relative_euler)
    while np.linalg.norm(end_pos - target_pose) > 0.01:
        if it > max_it:
            break
        delta = end_pos - target_pose
        for i in range(len(joint_path)):
            cur_idx = joint_path[i]
            Q_p = joint_orientations[joint_parent_new[cur_idx]]
            relative_euler = joint_eulers[cur_idx].as_euler()
            cord_x = R.from_quat(Q_p).apply(np.array([1,0,0]))
            cord_y = (R.from_quat(Q_p) * R.from_euler(relative_euler[0],"X")).apply(np.array([0,1,0]))
            cord_z = (R.from_quat(Q_p) * R.from_euler(relative_euler[0],"X") * R.from_euler(relative_euler[1],"Y")).apply(np.array([0,0,1]))

            r = end_pos - joint_positions[cur_idx]
            joint_eulers[i][0] -= alpha * np.dot(np.cross(cord_x, r) , delta)
            joint_eulers[i][1] -= alpha * np.dot(np.cross(cord_y, r), delta)
            joint_eulers[i][2] -= alpha * np.dot(np.cross(cord_z, r), delta)
        
        #更新joint_path当中的joint_orientation, joint_position
        for i in range(1, len(joint_path)):
            cur_idx = joint_path[i]
            joint_orientations[cur_idx] = joint_orientations[joint_parent_new[cur_idx]] * R.from_euler("XYZ", joint_eulers)
            joint_positions[cur_idx] = joint_positions[joint_parent_new[cur_idx]] + R.from_quat(joint_orientations[joint_parent_new[cur_idx]]).apply(joint_offset[cur_idx])
        end_pos = joint_positions[idx_end]
    #根据joint offset 和 joint_eulers更新剩余的joint
    #遇到在joint_path内的要跳过
    #不能使用joint_parent_new,得用meta_data.joint_parent
    #需要建立树，方便从根节点开始更新
    joint_orientations = R.from_euler(joint_eulers).as_quat()
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations