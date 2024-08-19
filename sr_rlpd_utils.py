import functools
import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence, Type, Dict, Optional, Union, Any
import pdb
default_init = nn.initializers.xavier_uniform
import numpy as np

DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]

class TwoInput(nn.Module):
    '''
    input could be (obs, actions) or (embed, actions)
    for example, Q(s,a), psi(s,a), psi(phi(s),a), forward(phi(s),a)
    the frist operation is concat the state or embedding with the action'''
    base_cls: nn.Module
    output_dim: int=17 # output_dim, could be encoder_dim, state_dim, or 1
    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)
        value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)
        return value

class OneInput(nn.Module):
    '''
    single input
    for example, Q(phi(s,a)), phi(s), d(phi(s))
    just a dense layer, no concatnation'''
    net_cls: Type[nn.Module]
    output_dim: int=17 # output_dim, could be encoder_dim, state_dim, or 1
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.net_cls()(inputs, *args, **kwargs)
        value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)
        if self.output_dim == 1:
            return jnp.squeeze(value, -1) # important!
        else:
            return value

class Ensemble(nn.Module):
    '''
    By using variable_axes={'params': 0}, we indicate that the parameters themselves are mapped over 
    and therefore not shared along the mapped axis. Consequently, we also split the 'params' RNG, 
    otherwise the parameters would be initialized identically along the mapped axis
    
    split_rngs -- Split PRNG sequences will be different for each index of the batch dimension.
    Unsplit PRNGs will be broadcasted.
    '''
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)

class MLP(nn.Module):
    hidden_dims: Sequence[int]  # 隐藏层的维度，是一个整数序列
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu  # 激活函数，默认为ReLU
    activate_final: bool = False  # 是否对最后一层进行激活
    use_layer_norm: bool = False  # 是否使用层归一化
    scale_final: Optional[float] = None  # 最后一层的缩放因子，可选
    dropout_rate: Optional[float] = None  # dropout 的比率，可选
    use_pnorm: bool = False  # 是否使用 p 范数

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray: # x 是输入，training 表示是否在训练模式下
        for i, size in enumerate(self.hidden_dims): # 遍历每一层
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None: # 如果是最后一层且有缩放因子
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final: # 如果不是最后一层或者需要对最后一层进行激活
                if self.dropout_rate is not None and self.dropout_rate > 0: # 如果 dropout 比率不为空且大于 0
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training) # 对 x 进行 dropout
                if self.use_layer_norm: # 如果使用层归一化
                    x = nn.LayerNorm()(x) # 对 x 进行层归一化
                x = self.activations(x) # 对 x 进行激活
        if self.use_pnorm: # 如果使用 p 范数
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10) # 对 x 进行 p 范数归一化
        return x

class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties

class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)