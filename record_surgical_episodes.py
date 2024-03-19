import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import gym

from surrol.const import ROOT_DIR_PATH
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS, surgical_tasks
from collections import OrderedDict
from utils import parse_ts
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
    render_cam_name = 'ecm'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']


    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = gym.make(task_name)
        seeds = np.random.randint(0, 1001)
        # 同时设定这两个就可以保证可复现性
        env.seed(seeds)
        np.random.seed(seeds)
        ts = env.reset()  # 初始时候return的是一个observation
        obs = ts
        ts = parse_ts(ts, env)
        episode = []  # 和普通机械臂不一样，这里第一步就不加了因为没有相应的action
        # setup plotting
        view = 'ecm'  # 对于我的写angle
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            plt.ion()
        for step in range(episode_len):
            action = env.get_oracle_action(obs)
            # print(action)
            ts = env.step(action)
            obs = ts[0]
            ts = parse_ts(ts, env, action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode])  # episode_return反映的是奖励的水平
        episode_max_reward = np.max([ts.reward for ts in episode])  # 获取最大的reward
        # env需要写max_reward
        if episode_max_reward == 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        action_traj = [ts.observation['action'] for ts in episode]



        # clear unused variables
        del episode


        # setup the environment
        print('Replaying joint commands')
        # 这里因为用的同一个env所以就不用再make一遍了，但是为了保证数据一致还得重新设置一下seeds
        env.seed(seeds)
        np.random.seed(seeds)
        ts = env.reset()
        ts = parse_ts(ts, env)
        episode_replay = [ts]  # 这里是为了有第一步的obs来实现，然后才是action
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][view])
            plt.ion()
        # ts = env.step(action)
        for t in range(len(action_traj)):
            action = action_traj[t]
            # print(action)
            #
            ts = env.step(action)
            ts = parse_ts(ts, env, action)
            # print((np.array(ts.observation['qpos'])-np.array(action))[4:])
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][view])
                plt.pause(0.02)
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        print(episode_max_reward, 0)
        if episode_max_reward == 0:
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
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        # joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while action_traj:
            action = action_traj.pop(0)  # action这里又回到了关节空间，上面使用qpos里面获取到的
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])  # qpos不太确定是哪个，要看env里面写的究竟是啥
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        if task_name == 'sim_single_cube':
            dim = 8
        elif task_name in surgical_tasks:
            robo_state_dim = 7
            dim = 5
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
            qpos = obs.create_dataset('qpos', (max_timesteps, robo_state_dim))  # 对于其他任务是dim
            action = root.create_dataset('action', (max_timesteps, dim))
            # 最后将数据给提案进去
            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        env.close()
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

