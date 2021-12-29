import numpy as np
import gym
from gym.spaces import Box, Discrete, MultiDiscrete
from typing import Dict, List, Union
from gym.envs.classic_control import CartPoleEnv

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.torch_utils import one_hot as torch_one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy.sample_batch import SampleBatch


class StatelessCartPole(CartPoleEnv):
    """Partially observable variant of the CartPole gym environment.

    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    cartpole.py

    We delete the x- and angular velocity components of the state, so that it
    can only be solved by a memory enhanced model (policy).
    """

    def __init__(self, config=None):
        super().__init__()

        # Fix our observation-space (remove 2 velocity components).
        high = np.array(
            [
                self.x_threshold * 2,
                self.theta_threshold_radians * 2,
            ],
            dtype=np.float32)

        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        next_obs, reward, done, info = super().step(action)
        # next_obs is [x-pos, x-veloc, angle, angle-veloc]
        return np.array([next_obs[0], next_obs[2]]), reward, done, info

    def reset(self):
        init_obs = super().reset()
        # init_obs is [x-pos, x-veloc, angle, angle-veloc]
        return np.array([init_obs[0], init_obs[2]])


torch, nn = try_import_torch()


class TorchFrameStackingCartPoleModel(TorchModelV2, nn.Module):
    """A simple FC model that takes the last n observations as input."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 num_frames=3):
        nn.Module.__init__(self)
        super(TorchFrameStackingCartPoleModel, self).__init__(
            obs_space, action_space, None, model_config, name)

        self.num_frames = num_frames
        self.num_outputs = num_outputs

        # Construct actual (very simple) FC model.
        assert len(obs_space.shape) == 1
        in_size = self.num_frames * (obs_space.shape[0] + action_space.n + 1)
        self.layer1 = SlimFC(
            in_size=in_size, out_size=256, activation_fn="relu")
        self.layer2 = SlimFC(in_size=256, out_size=256, activation_fn="relu")
        self.out = SlimFC(
            in_size=256, out_size=self.num_outputs, activation_fn="linear")
        self.values = SlimFC(in_size=256, out_size=1, activation_fn="linear")

        self._last_value = None

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs",
            shift="-{}:0".format(num_frames - 1),
            space=obs_space)
        self.view_requirements["prev_n_rewards"] = ViewRequirement(
            data_col="rewards", shift="-{}:-1".format(self.num_frames))
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space)

    def forward(self, input_dict, states, seq_lens):
        import pdb;pdb.set_trace()
        obs = input_dict["prev_n_obs"]
        obs = torch.reshape(obs,
                            [-1, self.obs_space.shape[0] * self.num_frames])
        rewards = torch.reshape(input_dict["prev_n_rewards"],
                                [-1, self.num_frames])
        actions = torch_one_hot(input_dict["prev_n_actions"],
                                self.action_space)
        actions = torch.reshape(actions,
                                [-1, self.num_frames * actions.shape[-1]])
        input_ = torch.cat([obs, actions, rewards], dim=-1)
        features = self.layer1(input_)
        features = self.layer2(features)
        out = self.out(features)
        self._last_value = self.values(features)
        return out, []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)

ray.init(num_cpus=0 or None)

ModelCatalog.register_custom_model("frame_stack_model", TorchFrameStackingCartPoleModel)
register_env("StatelessPendulum", lambda _: StatelessCartPole())

num_frames = 20

config = {
    "env": 'StatelessPendulum',
    "gamma": 0.9,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_workers": 0,
    "num_envs_per_worker": 20,
    "entropy_coeff": 0.001,
    "num_sgd_iter": 5,
    "vf_loss_coeff": 1e-5,
    "model": {
        "custom_model": "frame_stack_model",
        "custom_model_config": {
            "num_frames": num_frames,
            "fc_size" : [128, 64],
            "lstm_size": 256,
        },
    },
    "framework": 'torch',
}

stop = {
    "training_iteration": 10,
    "timesteps_total": 4000000,
    "episode_reward_mean": 510.,
}

results = tune.run('PPO',
                   config=config,
                   stop=stop,
                   verbose=2)

ray.shutdown()
