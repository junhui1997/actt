import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from constants import PANDA_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy
from script_single_policy import SinglePickPolicy

import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_single_cube':
        policy_cls = SinglePickPolicy
    else:
        raise NotImplementedError

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()  # 初始时候return的是一个timestamp
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        view = 'top'  # 对于我的写angle
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            # print(action)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])  # episode_return反映的是奖励的水平
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])  # 获取最大的reward
        # env需要写max_reward
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control # gripper一个夹爪上面有两个，是另一个负数，我们取正数的那个
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        # 将开启的初始值到关闭值给变换到1-0,0对应着关闭 # 这里修改的是qpos的量
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            if task_name == 'sim_single_cube':
                left_ctrl = PANDA_GRIPPER_POSITION_NORMALIZE_FN(ctrl[7])
                joint[7] = left_ctrl
            else:
                left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
                right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
                joint[6] = left_ctrl
                joint[6+7] = right_ctrl  # joint[13],刚好左右分别是index[6,13]

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()
        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            plt.ion()
        # ts = env.step(action)
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            # print(action)
            #
            ts = env.step(action)
            # print((np.array(ts.observation['qpos'])-np.array(action))[4:])
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.02)
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        print(episode_max_reward, env.task.max_reward)
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        #




        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)  # action这里又回到了关节空间，上面使用qpos里面获取到的
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])  # qpos不太确定是哪个，要看env里面写的究竟是啥
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        if task_name == 'sim_single_cube':
            dim = 8
        else:
            dim = 14
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')  # create group相当于新建folder
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )  # create dataset相当于先预留一个空间，size要对的上
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, dim))
            qvel = obs.create_dataset('qvel', (max_timesteps, dim))
            action = root.create_dataset('action', (max_timesteps, dim))
            # 最后将数据给提案进去
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

