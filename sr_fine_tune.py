from absl import app, flags
import os
from typing import Tuple
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"
import ml_collections
import time
import random
import numpy as np
from tqdm import trange
import collections
import jax
import jax.numpy as jnp
import pdb
import dmc
from pathlib import Path
from sr_models import SR_SACLearner
import optax
from ml_collections.config_dict import config_dict
Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "discounts", "next_observations"])

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = int(2e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((max_size, obs_dim))
        self.actions = np.zeros((max_size, act_dim))
        self.next_observations = np.zeros((max_size, obs_dim))
        self.rewards = np.zeros(max_size)
        self.discounts = np.zeros(max_size)

    def load_utds(self, replay_dir_list):
        for i in range(len(replay_dir_list)):
            _replay_dir = replay_dir_list[i]
            eps_fns = sorted(_replay_dir.glob('*.npz'))
            for eps_fn in eps_fns:
                # load a npz file to represent an episodic sample. The keys include 'observation', 'action', 'reward', 'discount', 'physics'
                with eps_fn.open('rb') as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()} # dict
                eps_len = episode['observation'].shape[0]
                self.observations[self.size:self.size+eps_len-1] = episode['observation'][:-1]
                self.actions[self.size:self.size+eps_len-1] = episode['action'][1:]
                self.rewards[self.size:self.size+eps_len-1] = episode['reward'][1:].squeeze()
                self.next_observations[self.size:self.size+eps_len-1] = episode['observation'][1:]
                self.discounts[self.size:self.size+eps_len-1] = episode['discount'][:-1].squeeze()
                self.size += eps_len -1
                if self.size % 10000 == 0:
                    print('load ', eps_fn, ' ', self.size)

    def add(self, observation: np.ndarray, action: np.ndarray,
            next_observation: np.ndarray, reward: float, done: float):
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.next_observations[self.ptr] = next_observation
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = 1 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = Batch(observations=self.observations[idx],
                      actions=self.actions[idx],
                      rewards=self.rewards[idx],
                      discounts=self.discounts[idx],
                      next_observations=self.next_observations[idx])
        return batch

    def normalize_obs(self, eps: float = 1e-3):
        mean = self.observations.mean(0)
        std = self.observations.std(0) + eps
        self.observations = (self.observations - mean)/std
        self.next_observations = (self.next_observations - mean)/std
        return mean, std

def eval_policy(agent, env, eval_episodes = 10) -> Tuple[float, float]:
    avg_reward = 0.
    for _ in range(eval_episodes):
        time_step = env.reset()
        while not time_step.last():
            action = agent.eval_actions(time_step.observation)
            time_step = env.step(action)
            avg_reward += time_step.reward
    avg_reward /= eval_episodes
    return {"reward": avg_reward}

Envs = ['quadruped_walk', 'quadruped_run', 'quadruped_roll_fast', 'quadruped_jump',
        'jaco_reach_top_left', 'jaco_reach_top_right', 'jaco_reach_bottom_right', 'jaco_reach_bottom_left',
        'walker_walk', 'walker_run', 'walker_flip']
Data_types = ['medium', 'medium-replay', 'expert', 'replay']

def get_config():
    config = ml_collections.ConfigDict()
    config.offline_env = 2
    config.online_env = 0
    config.data_type_indx = 3
    config.seed = 42
    config.batch_size = 256
    config.checkpoint_model = True
    config.offline_timesteps = 500000
    config.max_timesteps = 250000
    config.utd_ratio = 1

    sub = ml_collections.ConfigDict()
    sub.actor_lr = 3e-4
    sub.critic_lr = 3e-4
    sub.temp_lr = 3e-4
    sub.backup_entropy = True
    sub.critic_layer_norm = True
    sub.critic_weight_decay = config_dict.placeholder(float)
    sub.target_entropy = config_dict.placeholder(float)
    sub.discount = 0.99
    sub.hidden_dims = (256, 256)
    sub.init_temperature = 1.0
    sub.tau = 0.005
    sub.num_qs = 10
    sub.sr_num_qs = 10
    sub.sr_use_LN = False
    sub.use_q_msg = False
    sub.use_sr_msg = True
    config.sub = sub

    return config

import wandb

def train_and_evaluate(configs: ml_collections.ConfigDict): 
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    exp_name = f'sr_{Data_types[configs.data_type_indx]}_seed{configs.seed}_{timestamp}'
    exp_info = f'# Running experiment for: {exp_name}_{Envs[configs.offline_env]} #'
    print('#'*len(exp_info) + f'\n{exp_info}\n' + '#'*len(exp_info))

    # wandb.init(project='exp_utds_for_paper', name=f'{exp_name}', 
            #    group=f'Offline-{Envs[configs.offline_env]}', config=dict(configs))
    # wandb.init(project='exp_utds_for_paper_fine_tune', name=f'{exp_name}', 
            #    group=f'{Envs[configs.offline_env]}-to-{Envs[configs.online_env]}', config=configs)
    wandb.init(project='exp_utds_test', name=f'{exp_name}', 
               group=f'{Envs[configs.offline_env]}-to-{Envs[configs.online_env]}', config=configs)

    # initialize the environment 
    offline_env = dmc.make(Envs[configs.offline_env], seed=configs.seed)
    offline_eval_env = dmc.make(Envs[configs.offline_env], seed=configs.seed+42)
    online_env = dmc.make(Envs[configs.online_env], seed=configs.seed)
    online_eval_env = dmc.make(Envs[configs.online_env], seed=configs.seed+42)
    obs_dim = offline_env.observation_spec().shape[0]
    act_dim = offline_env.action_spec().shape[0]
    time_step = offline_env.reset()

    np.random.seed(configs.seed)
    random.seed(configs.seed)

    kwargs = dict(configs.sub)
    print(kwargs)
    agent = SR_SACLearner.create(configs.seed, act_dim, time_step, **kwargs)
    # replay buffer
    data_type = Data_types[configs.data_type_indx]
    max_len = int(2e6) if data_type == 'replay' else int(1e6)
    if (configs.offline_env == 8 or configs.offline_env == 9) and data_type == 'medium-replay':
        max_len = 1100000
    replay_buffer = ReplayBuffer(obs_dim, act_dim, max_size=max_len)
    replay_dir_list = []
    datasets_dir =  '/mnt/disk1/yxd/CORL/UTDS/collect'
    replay_dir = datasets_dir / Path(Envs[configs.offline_env]+"-td3-"+str(data_type)) / 'data'
    print(f'replay dir: {replay_dir}')
    replay_dir_list.append(replay_dir)
    replay_buffer.load_utds(replay_dir_list) # 从UTDS数据集中加载数据，存为[s,a,s',r,gamma]

    for i in trange(0, configs.offline_timesteps+1):
        batch = replay_buffer.sample(configs.batch_size)
        agent, update_info = agent.update(batch, utd_ratio=1, bc_w = 1, fix_m = False)

        if i % 1000 == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-train/{k}": v}, step=i) # 上传wandb

        if i % 5000 == 0:
            eval_info = eval_policy(agent, offline_eval_env) # 评估离线训练的agent
            for k, v in eval_info.items():
                wandb.log({f"offline-eval/{k}": v}, step=i)
    
    # online fine-tuning
    time_step = online_env.reset()
    online_buffer = ReplayBuffer(obs_dim, act_dim, max_size=configs.max_timesteps)
    for t in trange(0, configs.max_timesteps+1, desc='online'):
        # interaction
        if not time_step.last():
            obs = time_step.observation
            action, _ = agent.sample_actions(obs)
            time_step = online_env.step(action)
            online_buffer.add(obs, action, time_step.observation, time_step.reward, time_step.last())
        else:
            time_step = online_env.reset()

        # train
        if t > configs.batch_size * configs.utd_ratio:
            batch = online_buffer.sample(configs.batch_size * configs.utd_ratio)
            agent, update_info = agent.update(batch, utd_ratio=configs.utd_ratio, bc_w = 0, fix_m = True)
        
        if t % 1000 == 0:
            for k, v in update_info.items():
                wandb.log({f"online-train/{k}": v}, step=t+configs.offline_timesteps) # 上传wandb

        if t % 5000 == 0:
            eval_info = eval_policy(agent, online_eval_env) # 评估离线训练的agent
            eval1_info = eval_policy(agent, offline_env)
            print('eval: ', eval_info['reward'], ' ', eval1_info['reward'])
            for k, v in eval_info.items():
                wandb.log({f"online-eval/{k}": v}, step=t+configs.offline_timesteps)

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("offline_env", 4, "Offline env index.")
flags.DEFINE_integer("online_env", 5, "Online env index.")
flags.DEFINE_integer("data_type_indx", 3, "Data type index.")

def main(argv):
    configs = get_config()
    configs.offline_env = FLAGS.offline_env
    configs.online_env = FLAGS.online_env
    configs.seed = FLAGS.seed
    configs.data_type_indx = FLAGS.data_type_indx
    train_and_evaluate(configs)

if __name__ == '__main__':
    app.run(main)
