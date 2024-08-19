from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Any
import pdb
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn
from sr_rlpd_utils import TanhNormal, MLP, Ensemble, TwoInput, OneInput, DatasetDict
import numpy as np
from dmc import ExtendedTimeStep
PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]

@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=new_rng)

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))

class SR_SACLearner(Agent):
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    sr: TrainState
    target_sr: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    # num_min_qs: Optional[int] = struct.field(pytree_node=False)  # See M in RedQ https://arxiv.org/abs/2101.05982
    sr_num_qs: int = struct.field(pytree_node=False)
    # sr_num_min_qs: int = struct.field(pytree_node=False)
    sr_use_LN: bool = struct.field(pytree_node=False)
    use_q_msg: bool = struct.field(pytree_node=False)
    use_sr_msg: bool = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod    
    def create(
        cls, # cls代表类本身，相当于self
        seed: int,  # 随机数种子
        action_dim: int,  # 动作空间
        timestep: ExtendedTimeStep, # env reset的结果，包括observation, action, reward等
        actor_lr: float = 3e-4,  # actor的学习率
        critic_lr: float = 3e-4,  # critic的学习率
        temp_lr: float = 3e-4,  # temperature的学习率
        hidden_dims: Sequence[int] = (256, 256),  # 隐藏层维度
        discount: float = 0.99,  # 折扣因子
        tau: float = 0.005,  # 软更新的系数
        num_qs: int = 2,  # critic的数量
        # num_min_qs: Optional[int] = None,  # 最小的critic数量
        critic_dropout_rate: Optional[float] = None,  # critic的dropout率
        critic_weight_decay: Optional[float] = None,  # critic的权重衰减率
        critic_layer_norm: bool = False,  # 是否使用层归一化
        target_entropy: Optional[float] = None,  # 目标熵
        init_temperature: float = 1.0,  # 初始温度
        backup_entropy: bool = True,  # 是否备份熵
        use_pnorm: bool = False,  # 是否使用p范数
        sr_num_qs: int = 10, # sr的ensemble的数量
        # sr_num_min_qs: int = 2, # subsample时用的sr的ensemble的数量
        sr_use_LN: bool = True, # SR网络是否用LayerNorm
        use_q_msg: bool = True, # use independent targets for q
        use_sr_msg: bool = True, # use independent targets for sr
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        observations = timestep.observation
        actions = timestep.action

        if target_entropy is None:
            target_entropy = -action_dim / 2 # 将目标熵设置为动作空间维度的负一半

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, sr_key = jax.random.split(rng, 5)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm)
        actor_def = TanhNormal(actor_base_cls, action_dim) # 定义一个TanhNormal策略网络
        actor_params = actor_def.init(actor_key, observations)["params"] # 初始化策略网络的参数
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(learning_rate=actor_lr)) # 创建一个策略网络的训练状态

        if sr_use_LN:
            sr_cls = partial(MLP, hidden_dims=(256, 256), activate_final=True, use_layer_norm=True)
        else:
            sr_cls = partial(MLP, hidden_dims=(256, 256), activate_final=True)
        print('sr use ensemble')
        sr_cls = partial(TwoInput, base_cls=sr_cls, output_dim=observations.shape[0])
        sr_def = Ensemble(sr_cls, num=sr_num_qs)
        sr_params = sr_def.init(sr_key, observations, actions)["params"]
        target_sr_def = Ensemble(sr_cls, num=sr_num_qs) # 网络结构的num取多少没有关系，重要的是参数的ensemble是多少，多少组参数对应多少个估计
        
        sr = TrainState.create(apply_fn=sr_def.apply, params=sr_params, tx=optax.adam(learning_rate=critic_lr))
        target_sr = TrainState.create(apply_fn=target_sr_def.apply, params=sr_params, tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        critic_base_cls = partial(MLP, hidden_dims=(256, 256), activate_final=True, dropout_rate=critic_dropout_rate,
                                  use_layer_norm=critic_layer_norm, use_pnorm=use_pnorm)
        critic_cls = partial(OneInput, net_cls=critic_base_cls, output_dim=1) # (256,256)后面加一层Dense，输出(1)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(learning_rate=critic_lr,weight_decay=critic_weight_decay,mask=decay_mask_fn,)
        else: 
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply,params=critic_params,tx=tx) # 创建一个状态-动作值函数的训练状态
        target_critic_def = Ensemble(critic_cls, num=num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply, params=critic_params,
                                        tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(apply_fn=temp_def.apply,params=temp_params,tx=optax.adam(learning_rate=temp_lr))

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            sr=sr,
            target_sr=target_sr,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            # num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            sr_num_qs=sr_num_qs,
            # sr_num_min_qs=sr_num_min_qs,
            sr_use_LN=sr_use_LN,
            use_q_msg=use_q_msg,
            use_sr_msg=use_sr_msg,
        )
    
    def update_actor(self, batch: DatasetDict, bc_w: int, key) -> Tuple[Agent, Dict[str, float]]:
        # key, rng = jax.random.split(self.rng, 2)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch.observations) # pi(s)
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            # 计算Q值
            m_sa = self.sr.apply_fn({"params": self.sr.params}, batch.observations, actions)
            qs = self.critic.apply_fn({"params": self.critic.params}, m_sa)  # training=True
            m_sa = m_sa.min(axis=0)
            q = qs.min(axis=0)
            # m_sa = m_sa.mean(axis=0)
            # q = qs.mean(axis=0)
            q_loss = (log_probs * self.temp.apply_fn({"params": self.temp.params}) - q).mean() # 计算actor的损失, alpha * log_prob - Q
            bc_loss = ((actions - batch.actions) ** 2).mean() # add bc loss
            actor_loss = q_loss + bc_w * bc_loss
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean(), "bc_loss": bc_loss, "q_loss": q_loss}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params) # 计算actor_loss_fn的梯度和actor_info
        actor = self.actor.apply_gradients(grads=grads) # 使用梯度更新actor

        return self.replace(actor=actor), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict, key) -> Tuple[TrainState, Dict[str, float]]:
        dist = self.actor.apply_fn({"params": self.actor.params}, batch.next_observations) # pi(s')
        # key, rng = jax.random.split(self.rng)
        next_actions = dist.sample(seed=key)

        # key, rng = jax.random.split(rng)
        # range = 0.5
        # re = jnp.repeat(jnp.expand_dims(batch.rewards, axis=0), repeats=self.num_qs, axis=0)
        # noise = jax.random.uniform(key, shape=re.shape, minval=-range, maxval=range)
        # re += noise

        next_m_sa = self.sr.apply_fn({"params": self.sr.params}, batch.next_observations, next_actions)#.mean(axis=0)
        m_sa = self.sr.apply_fn({"params": self.sr.params}, batch.observations, batch.actions)#.mean(axis=0)

        # 计算Q(s', a')
        next_q = self.target_critic.apply_fn({"params": self.target_critic.params}, next_m_sa)
        if not self.use_q_msg:
            next_q = next_q.min(axis=0) # min(Q(s',a'))

        target_q = batch.rewards + self.discount * next_q # target_q = r + gamma * Q(s',a')
        # target_q = re + self.discount * next_q # target_q = r + gamma * Q(s',a')

        if self.backup_entropy: # from SAC
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (self.discount * self.temp.apply_fn({"params": self.temp.params})* next_log_probs)

        # key, rng = jax.random.split(rng)
        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn({"params": critic_params}, m_sa)
            critic_loss = ((qs - target_q) ** 2).mean() # TD error
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean(), "q_std": qs.std(0).mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params) # 计算梯度
        critic = self.critic.apply_gradients(grads=grads) # 进行梯度更新

        target_critic_params = optax.incremental_update(critic.params, self.target_critic.params, self.tau) # soft update
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic), info
   
    def update_sr(self, batch: DatasetDict, key) -> Tuple[TrainState, Dict[str, float]]:
        # dynamic amplitude sampling, refer to Augmented World Models
        # key, rng = jax.random.split(rng)
        # next_obs = jnp.repeat(jnp.expand_dims(batch.observations, axis=0), repeats=10, axis=0)
        # noise = jax.random.uniform(key, shape=next_obs.shape, minval=0.0, maxval=1.0) * \
        #     (batch.next_observations - batch.observations)
        # next_obs += noise
        dist = self.actor.apply_fn({"params": self.actor.params}, batch.next_observations) # pi(s')
        # dist = self.actor.apply_fn({"params": self.actor.params}, next_obs) # pi(s')
        # key, rng = jax.random.split(self.rng)
        next_actions = dist.sample(seed=key)
        
        next_m = self.target_sr.apply_fn({"params": self.target_sr.params}, batch.next_observations, next_actions)
        # next_m = self.target_sr.apply_fn({"params": self.target_sr.params}, next_obs, next_actions)
        if not self.use_sr_msg:
            next_m = next_m.min(axis=0)
        target_m = batch.observations + self.discount * next_m
        # target_m = next_obs + self.discount * next_m

        def sr_loss_fn(sr_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            m = self.sr.apply_fn({"params": sr_params}, batch.observations, batch.actions)
            sr_loss = ((m - target_m) ** 2).mean() # TD error
            return sr_loss, {"sr_loss": sr_loss, "m": m.mean(), "m_std": m.std(0).mean()}

        grads, info = jax.grad(sr_loss_fn, has_aux=True)(self.sr.params) # 计算梯度
        sr = self.sr.apply_gradients(grads=grads) # 进行梯度更新

        target_sr_params = optax.incremental_update(sr.params, self.target_sr.params, self.tau) # soft update
        target_sr = self.target_sr.replace(params=target_sr_params)

        return self.replace(sr=sr, target_sr=target_sr), info
 
    @partial(jax.jit, static_argnames=["utd_ratio", "bc_w", "fix_m"])
    def update(self, batch: DatasetDict, utd_ratio: int, bc_w: int, fix_m: bool):

        key1, key2, key3 = jax.random.split(self.rng, 3)
        # self.replace(rng=rng)
        new_agent = self        
        for i in range(utd_ratio):

            def slice(x): # 定义一个切片函数，用于将数据集切分成多个小批次
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)] # 每个小批次的起始和结束索引

            mini_batch = jax.tree_util.tree_map(slice, batch)
            '''
            jax.tree_util.tree_map函数的作用是将一个函数应用于数据结构中的每个元素,并返回一个与原始数据结构具有相同结构的新数据结构。
            '''
            if not fix_m:
                new_agent, sr_info = new_agent.update_sr(mini_batch, key1)
            new_agent, critic_info = new_agent.update_critic(mini_batch, key2)

        new_agent, actor_info = new_agent.update_actor(mini_batch, bc_w, key3)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])
        if not fix_m:
            return new_agent, {**actor_info, **critic_info, **sr_info}
        else:
            return new_agent, {**actor_info, **critic_info}
