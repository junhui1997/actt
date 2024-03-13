import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE, START_ARM_POSE_S
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PANDA_GRIPPER_POSITION_NORMALIZE_FN, PANDA_GRIPPER_POSITION_UNNORMALIZE_FN, PANDA_GRIPPER_POSITION_OPEN, PANDA_GRIPPER_POSITION_CLOSE

from my_util.my_func import check_collision
from utils import sample_box_pose, sample_insertion_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython

e = IPython.embed


def make_ee_sim_env(task_name):
    """
    obs是在joint space里面的，act是在ee的
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (positive: opening, negative: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (positive: opening, negative: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_single_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'new_panda/single_cube_ee.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = SingleCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env


class SingleEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    # 用于每个仿真步来更新数据，每个step都会去执行一下 #是四元数
    def before_step(self, action, physics):
        # set mocap position and quat # 这里是因为xml里面写了两个mocap所以写成这样
        # left
        np.copyto(physics.data.mocap_pos[0], action[:3])  # target，source
        np.copyto(physics.data.mocap_quat[0], action[3:7])

        # set gripper #actuator里面对应data.ctrl
        # 原本夹爪action对应的是[0-1]
        g_left_ctrl = PANDA_GRIPPER_POSITION_UNNORMALIZE_FN(action[7])
        np.copyto(physics.data.ctrl[-1:], np.array([g_left_ctrl]))  # 一共只需要改两个地方就行

    # init robot和before step执行的基本一致，唯一的区别就是init robot更新时候不传参数进去，每次设定为一个固定的量
    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:9] = START_ARM_POSE_S  # 设定所有关节角的初始位姿，和action space一致 # 在关节空间和这里是七个自由度

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side

        a = physics.named.data.xpos['hand']
        b = physics.named.data.xquat['hand']
        np.copyto(physics.data.mocap_pos[0], [-0.45,    0.6,   0.83])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PANDA_GRIPPER_POSITION_CLOSE
        ])
        np.copyto(physics.data.ctrl[-1:], close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    # qpos和qvel返回的都是8个数值
    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()  # 这里len包含了机械臂和环境中的其他东西，比如盒子，qpos是0～0.04,下面那个norm放缩的是control range，每次使用norm是为了归一，un_norm是为了变回使用的尺度
        left_qpos_raw = qpos_raw[:9]  # 7个关节+两个开关量 # 这里没错，虽然只需要一个控制信号，但是夹爪这里是两个姿态值
        left_arm_qpos = left_qpos_raw[:7]  # 这里取前7个
        left_gripper_qpos = [PANDA_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])*(255/0.04)]  # 每次这里都是第八个 # 夹爪这里的obs是归一化了的，从control range映射回position range
        return np.concatenate([left_arm_qpos, left_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:9]
        left_arm_qvel = left_qvel_raw[:7]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]  # 直接按照位置比例进行缩放， x/(open-close) # 这里暂时没管因为我训练时候也不用qval
        return np.concatenate([left_arm_qvel, left_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')  # 这里本来就只写了一个top的camera，在scene里面写的
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')  # 这里本来就只写了一个top的camera，在scene里面写的
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()

        # used when replaying joint trajectory # 对于panda来说这里是8 # 因为是7个转动加一个split
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class SingleCubeEETask(SingleEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()  # 在设定范围内随机生成一个box
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')  # 获取关节名称为red_box_joint中的joint的id，这里和mojoco里面用法还不一样
        np.copyto(physics.data.qpos[box_start_idx: box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[9:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        all_geom = []
        # physics.data.ncon: num of contract,现有接触的pair的数目
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            # 转化为name
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_geom.append((id_geom_1, id_geom_2))
            all_contact_pairs.append(contact_pair)  # 分别是左右手的左右finger，形成两个pair

        # print(all_geom)
        # print(all_contact_pairs)
        # 下面三个分别是左边手夹住，右边手夹住，以及离地
        # touch_left_gripper = ("red_box", "left_finger_collision") in all_contact_pairs  # 返回一个bool，是判断这个pair有没有在里面,验证过了
        touch_left_gripper = check_collision(['fpc1', 'fpc2', 'fpc3', 'fpc4', 'fpc5', ], 'red_box', all_contact_pairs)  # 这个在panda的xml里面
        touch_table = ("red_box", "table") in all_contact_pairs

        # 下面是分成了四个阶段
        reward = 0
        if touch_left_gripper:  # toucher
            reward = 1
        if touch_left_gripper and not touch_table:  # successful pick
            reward = 2
        return reward


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    # 用于每个仿真步来更新数据，每个step都会去执行一下
    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat # 这里是因为xml里面写了两个mocap所以写成这样
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])  # target，source
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper #qqqq：actuator里面对应data.ctrl么
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    # init robot和before step执行的基本一致，唯一的区别就是init robot更新时候不传参数进去，每次设定为一个固定的量
    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE  # 设定所有关节角的初始位姿，和action space一致

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side

        # 感觉哪个都对不上号来者
        a = physics.named.data.xpos['vx300s_right/gripper_link']
        b = physics.named.data.xquat['vx300s_right/gripper_link']
        np.copyto(physics.data.mocap_pos[0], [-0.31718881 + 0.1, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881 - 0.1, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()  # 这里len是23不太懂
        left_qpos_raw = qpos_raw[:8]  # 这里是前8个，因为是6+2,2是因为左右夹抓是对陈的，对方的负数 # 观察了一下确实这样
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]  # 这里取前6六个
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]  # 每次这里都是normalize之后的
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]  # 直接按照位置比例进行缩放， x/(open-close)
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')  # 这里本来就只写了一个top的camera，在scene里面写的
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()  # 在设定范围内随机生成一个box
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')  # 获取关节名称为red_box_joint中的joint的id，这里和mojoco里面用法还不一样
        np.copyto(physics.data.qpos[box_start_idx: box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        # physics.data.ncon: num of contract,现有接触的pair的数目
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            # 转化为name
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)  # 分别是左右手的左右finger，形成两个pair

        # 下面三个分别是左边手夹住，右边手夹住，以及离地
        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs  # 返回一个bool，是判断这个pair有没有在里面,验证过了
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        # 下面是分成了四个阶段
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7  # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx: peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx: socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward
