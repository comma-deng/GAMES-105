# 以下部分均为可更改部分

from answer_task1 import *
from task2_preprocess import MotionKey
import os
import pickle

position_weight = 1
vel_weight = 1
joint_position_weight = 10
joint_vel_weight = 10
bvh_folder_path = r'motion_material\kinematic_motion'

def find_min_cost(motion_key_desire, motion_key_real):
    pos_cost = np.sum(np.linalg.norm(motion_key_desire.positions - motion_key_real.positions, axis = 2), axis=1)
    vel_cost = np.sum(np.linalg.norm(motion_key_desire.velocities - motion_key_real.velocities, axis = 2), axis=1)
    joint_position_cost = np.sum(np.linalg.norm(motion_key_desire.joint_postions - motion_key_real.joint_postions, axis = 2), axis=1)
    joint_vel_cost = np.sum(np.linalg.norm(motion_key_desire.joint_postions - motion_key_real.joint_postions, axis = 2), axis=1)
    res = pos_cost * position_weight + vel_cost * vel_weight + joint_position_weight * joint_position_cost + joint_vel_weight * joint_vel_cost
    # print("pos_cost {0}, vel_cost {1}, joint_position_cost {2}, joint_vel_cost {3}".format(pos_cost, vel_cost, joint_position_cost, joint_vel_cost))
    # res = joint_position_weight * joint_position_cost + joint_vel_weight * joint_vel_cost

    return np.argmin(res), res[np.argmin(res)]

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        # self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        self.cur_seq = 0
        self.motion_keys = []
        for file in os.listdir(bvh_folder_path):
            if not file.endswith('keys'):
                continue
            f = open(os.path.join(bvh_folder_path ,file), 'rb')
            self.motion_keys.append(pickle.load(f))
            bvh_file_name = os.path.join(bvh_folder_path ,file.replace('keys', 'bvh'))
            print(bvh_file_name)
            self.motions.append(BVHMotion(bvh_file_name))
            f.close()
        pass
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        # joint_name = self.motions[0].joint_name
        # joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics()
        # joint_translation = joint_translation[self.cur_frame]
        # joint_orientation = joint_orientation[self.cur_frame]

        #TDOD 自己的代码
        real_key = MotionKey(1)
        real_key.positions[0] = desired_pos_list
        for i in range(6):
            real_key.positions[0][5-i] -= desired_pos_list[0]
        real_key.velocities[0] = desired_vel_list
        real_key.tracking_joints = self.motion_keys[0].tracking_joints
        for k in range(len(real_key.tracking_joints)):
            joint = real_key.tracking_joints[k]
            idx = self.motions[self.cur_seq].joint_name.index(joint)
            joint_translation, joint_orientation = self.motions[self.cur_seq].forward_kinematics_at_index([self.cur_frame, self.cur_frame+1])
            real_key.joint_postions[0][k] = joint_translation[0, idx, :] - joint_translation[0, 0, :]
            real_key.joint_velocities[0][k] = (joint_translation[1, idx, :] - joint_translation[0, idx, :]) * 60

        min_cost = 1e20
        min_seq_id = -1
        min_frame_id = -1
        for i in range(len(self.motion_keys)):
            motion_key = self.motion_keys[i]
            cur_id, cur_cost = find_min_cost(motion_key, real_key)
            if cur_cost < min_cost:
                min_cost = cur_cost
                min_seq_id = i
                min_frame_id = cur_id

        if min_seq_id == self.cur_seq and abs(min_frame_id-self.cur_frame) < 10:
            self.cur_frame = (self.cur_frame + 1) % self.motions[0].motion_length
        else:
            self.cur_seq = min_seq_id
            self.cur_frame = min_frame_id
        joint_name = self.motions[self.cur_seq].joint_name
        frame_position = self.motions[self.cur_seq].joint_position[self.cur_frame]
        frame_position[0, [0,2]] = desired_pos_list[0, [0,2]]
        frame_position = frame_position.reshape(1, frame_position.shape[0], frame_position.shape[1])
        frame_rotation = self.motions[self.cur_seq].joint_rotation[self.cur_frame]
        frame_rotation = frame_rotation.reshape(1, frame_rotation.shape[0], frame_rotation.shape[1])
        joint_translation, joint_orientation = self.motions[self.cur_seq].batch_forward_kinematics(frame_position, frame_rotation)
        joint_translation = joint_translation[0]
        joint_orientation = joint_orientation[0]
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        # self.cur_frame = (self.cur_frame + 1) % self.motions[0].motion_length
        # print(self.cur_frame, self.cur_seq)
        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)
        
        return character_state
    # 你的其他代码,state matchine, motion matching, learning, etc.