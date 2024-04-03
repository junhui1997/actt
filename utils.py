import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from collections import OrderedDict
import IPython

e = IPython.embed


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0)  # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        '''
        这里定位的是第多少个episode，以及以0开始计数的index
        '''
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        # print(index) # index就是[0~19600]
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        # print(dataset_path)
        with h5py.File(dataset_path, 'r') as root:
            try:  # some legacy data does not have this attribute
                is_sim = root.attrs['sim']
            except:
                is_sim = False
            compressed = root.attrs.get('compress', False)
            # 这里又处理了一次
            # to_modify
            if '/base_action' in root:
                base_action = root['/base_action'][()]
                base_action = preprocess_base_action(base_action)
                action = np.concatenate([root['/action'][()], base_action], axis=-1)
            else:
                action = root['/action'][()]
                if action.shape[1] > 14:
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
            original_action_shape = action.shape
            episode_len = original_action_shape[0]
            # get observation at start_ts only # [dim]获取当前时刻的obs
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

            # get all actions after and including start_ts #在本episode里面剩余的值
            if is_sim:
                action = action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        # self.is_sim = is_sim
        # 这里是没有pad的，action：[400,16],padded_action:[400,16]， act数值可能是一个小于400的值，所以要给pad，按照最大的episode_len
        padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1  # 给新填充的部分加了一个index就是1

        padded_action = padded_action[:self.chunk_size]  # 取前chunk_size,pad保证了永远可以取到前chunk_size的数值
        is_pad = is_pad[:self.chunk_size]  # is pad这里也被截取了

        # new axis for different cameras
        # 【num_camera,weigh,height,channel]
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last # to channel second
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # augmentation
        # 这里图像的变换并不会影响策略，因为transform之后相当于从随机的摄像头位置进行学习
        # 但是是不是会对双目视觉造成一定干扰呢？？
        # 如果没有规定transform则用这个
        if self.transformations is None:
            print('Initializing transformations')
            original_size = image_data.shape[2:]
            ratio = 0.95
            self.transformations = [
                transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                transforms.Resize(original_size, antialias=True),
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
            ]

        if self.augment_images:
            for transform in self.transformations:
                image_data = transform(image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        if self.policy_class == 'Diffusion':
            # normalize to [-1, 1] # diffusion policy有些特殊
            action_data = ((action_data - self.norm_stats["action_min"]) / (
                    self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
        else:
            # normalize to mean 0 std 1
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        # 【num_camera,channel,height,width],[dim],[chunk_size,dim],[chunk_size] #根据当前帧数据来预测未来100步
        return image_data, qpos_data, action_data, is_pad

    # def __getitem__(self, index):
    #     episode_id, start_ts = self._locate_transition(index)
    #     dataset_path = self.dataset_path_list[episode_id]
    #     try:
    #         # print(dataset_path)
    #         with h5py.File(dataset_path, 'r') as root:
    #             try:  # some legacy data does not have this attribute
    #                 is_sim = root.attrs['sim']
    #             except:
    #                 is_sim = False
    #             compressed = root.attrs.get('compress', False)
    #             if '/base_action' in root:
    #                 base_action = root['/base_action'][()]
    #                 base_action = preprocess_base_action(base_action)
    #                 action = np.concatenate([root['/action'][()], base_action], axis=-1)
    #             else:
    #                 action = root['/action'][()]
    #                 dummy_base_action = np.zeros([action.shape[0], 2])
    #                 action = np.concatenate([action, dummy_base_action], axis=-1)
    #             original_action_shape = action.shape
    #             episode_len = original_action_shape[0]
    #             # get observation at start_ts only
    #             qpos = root['/observations/qpos'][start_ts]
    #             qvel = root['/observations/qvel'][start_ts]
    #             image_dict = dict()
    #             for cam_name in self.camera_names:
    #                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
    #
    #             if compressed:
    #                 for cam_name in image_dict.keys():
    #                     decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
    #                     image_dict[cam_name] = np.array(decompressed_image)
    #
    #             # get all actions after and including start_ts
    #             if is_sim:
    #                 action = action[start_ts:]
    #                 action_len = episode_len - start_ts
    #             else:
    #                 action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
    #                 action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
    #
    #         # self.is_sim = is_sim
    #         padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
    #         padded_action[:action_len] = action
    #         is_pad = np.zeros(self.max_episode_len)
    #         is_pad[action_len:] = 1
    #
    #         padded_action = padded_action[:self.chunk_size]
    #         is_pad = is_pad[:self.chunk_size]
    #
    #         # new axis for different cameras
    #         all_cam_images = []
    #         for cam_name in self.camera_names:
    #             all_cam_images.append(image_dict[cam_name])
    #         all_cam_images = np.stack(all_cam_images, axis=0)
    #
    #         # construct observations
    #         image_data = torch.from_numpy(all_cam_images)
    #         qpos_data = torch.from_numpy(qpos).float()
    #         action_data = torch.from_numpy(padded_action).float()
    #         is_pad = torch.from_numpy(is_pad).bool()
    #
    #         # channel last
    #         image_data = torch.einsum('k h w c -> k c h w', image_data)
    #
    #         # augmentation
    #         if self.transformations is None:
    #             print('Initializing transformations')
    #             original_size = image_data.shape[2:]
    #             ratio = 0.95
    #             self.transformations = [
    #                 transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
    #                 transforms.Resize(original_size, antialias=True),
    #                 transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
    #                 transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)  # , hue=0.08)
    #             ]
    #
    #         if self.augment_images:
    #             for transform in self.transformations:
    #                 image_data = transform(image_data)
    #
    #         # normalize image and change dtype to float
    #         image_data = image_data / 255.0
    #
    #         if self.policy_class == 'Diffusion':
    #             # normalize to [-1, 1]
    #             action_data = ((action_data - self.norm_stats["action_min"]) / (
    #                         self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
    #         else:
    #             # normalize to mean 0 std 1
    #             action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
    #
    #         qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
    #
    #     except:
    #         print(f'Error loading1 {dataset_path} in __getitem__')
    #         quit()
    #
    #     # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
    #     return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                # qvel = root['/observations/qvel'][()]  # to_modify
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    # 被注释掉的是原本的
                    # action = root['/action'][()]  # 如果没有底盘的话，根据其他action的数目用0进行填充
                    # # dummy_base_action：【400,2】， action：【400,14】
                    # dummy_base_action = np.zeros([action.shape[0], 2])
                    # action = np.concatenate([action, dummy_base_action], axis=-1)

                    # to_modify
                    action = root['/action'][()]  # 如果没有底盘的话，根据其他action的数目用0进行填充
                    if action.shape[1] > 14:  # 双臂
                        # dummy_base_action：【400,2】， action：【400,14】
                        dummy_base_action = np.zeros([action.shape[0], 2])
                        action = np.concatenate([action, dummy_base_action], axis=-1)


        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        # episode len都是400
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    # state里面反应了一个episode里面的所有数值的统计量，以及给了一个示例，为了整体的正则化
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps, "action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size,
              skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None,
              train_ratio=0.99):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for
                                                   idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(
        f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list)
    # train_episode_len [list:49] 注意这是被flatten之后,带l的是没有flatten的[list:1]实际是[[]]结构
    # val_episode_len   [list:49]
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(
        flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')
    # 如果有batchsampler就不需要再获取len了，实际下面的采样也是在每个dataset中进行的采样
    # 下面的操作是为了混合训练应该，sample weight形式类似于【0.5,0.5】和你的数据集种类有关，这里代表的是分别使用两个数据集内的参数，按照各50%的比例进行采样
    # 如果里面有yield的话则函数不会立刻执行，而是会生成一个生成器，使用next调用之后才会执行一次，yeild特殊的地方在于会resume上次的位置，而不是直接从while最开始地方执行
    # prob = [1]
    # episode_idx = np.random.choice(len(train_episode_len_l), p=prob)
    # sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in train_episode_len_l]) #【0,19600】直接是400*49所有episode的长度
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len,
                                    chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len,
                                  chunk_size, policy_class)
    train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(
        f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True,
                                  num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True,
                                num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0  # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action


def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5) / 5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)


def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action


def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions
# 按照key计算均值
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class time_step:
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward


def parse_ts(ts, env, action=None, is_joint=False, is_bi=False):
    new_ts = time_step(OrderedDict(), -1)
    # 使用关节空间进行训练
    if is_joint:
        # 使用关节空间进行训练时候qpos即是指令也是观测值， 不需要使用action这条信息
        if isinstance(ts, dict):
            new_ts.reward = 0
            if is_bi:
                new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + [-0.8] + env.psm2.get_current_joint_position() + [-0.8]# 六个关节数值和夹爪，夹爪信息开
            else:
                new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + [0.5]  # 六个关节数值和夹爪，夹爪信息开
        else:
            new_ts.reward = env.get_reward()
            if is_bi:
                new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + [action[4]] + env.psm2.get_current_joint_position() + [action[-1]]
            else:
                new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + [action[-1]]
    else:
        if is_bi:
            state_dim = 14
        else:
            state_dim = 7
        if isinstance(ts, dict):
            new_ts.reward = 0
            new_ts.observation['qpos'] = ts['observation'][:state_dim]
        else:
            new_ts.reward = env.get_reward()
            new_ts.observation['qpos'] = ts[0]['observation'][:state_dim]  # 对应robot state：位置+欧拉角+夹爪
            new_ts.observation['action'] = action  # .tolist()
    ecm_img, mask = env.ecm.render_image(640, 480)  # 注意这里是反着写的
    front_img, front_mask = env.ecm.render_image_front(640, 480)  # 注意这里是反着写的
    top_img, top_mask = env.ecm.render_image_top(640, 480)  # 注意这里是反着写的
    # human_img = env.render('rgb_array')
    new_ts.observation['images'] = {}
    new_ts.observation['images']['ecm'] = ecm_img
    new_ts.observation['images']['top'] = top_img
    new_ts.observation['images']['front'] = front_img
    # new_ts.observation['images']['human'] = human_img
    return new_ts


def parse_ts_abs(ts, env, action=None, is_ee=False):
    new_ts = time_step(OrderedDict(), -1)
    # 使用关节空间进行训练
    if is_ee:
        # 使用关节空间进行训练时候qpos即是指令也是观测值， 不需要使用action这条信息
        if isinstance(ts, dict):
            new_ts.reward = -1
            new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + ts['observation'][6]  # 六个关节数值和夹爪，夹爪信息从robot state中获取
        else:
            new_ts.reward = ts[1]
            new_ts.observation['qpos'] = env.psm1.get_current_joint_position() + ts[0]['observation'][6]
    else:
        if isinstance(ts, dict):
            new_ts.reward = -1
            new_ts.observation['qpos'] = ts['observation'][:7]
        else:
            new_ts.reward = ts[1]
            new_ts.observation['qpos'] = ts[0]['observation'][:7]  # 对应robot state：位置+欧拉角+夹爪
            new_ts.observation['action'] = action  # .tolist()
    ecm_img, mask = env.ecm.render_image(640, 480)  # 注意这里是反着写的
    front_img, front_mask = env.ecm.render_image_front(640, 480)  # 注意这里是反着写的
    top_img, top_mask = env.ecm.render_image_top(640, 480)  # 注意这里是反着写的
    # human_img = env.render('rgb_array')
    new_ts.observation['images'] = {}
    new_ts.observation['images']['ecm'] = ecm_img
    new_ts.observation['images']['top'] = top_img
    new_ts.observation['images']['front'] = front_img
    # new_ts.observation['images']['human'] = human_img
    return new_ts
