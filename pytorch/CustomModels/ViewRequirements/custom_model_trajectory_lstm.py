import numpy as np
import gym
from gym.spaces import Box, Discrete, MultiDiscrete
from typing import Dict, List, Union
from gym.envs.classic_control import CartPoleEnv

import torch
from torch import nn
from torch.autograd import Variable

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
                 num_frames=20,
                 lstm_size=16):
        nn.Module.__init__(self)
        super(TorchFrameStackingCartPoleModel, self).__init__(
            obs_space, action_space, None, model_config, name)

        self.num_frames = num_frames
        self.num_outputs = num_outputs
        self.lstm_size = lstm_size
        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        # Construct actual (very simple) FC model.
        assert len(obs_space.shape) == 1
        fc_size = self.num_frames * self.lstm_size

        self.lstm = nn.LSTM(self.obs_size, self.lstm_size, batch_first=True)
        self.out = SlimFC(in_size=fc_size, out_size=self.num_outputs, activation_fn="linear")
        self.values = SlimFC(in_size=fc_size, out_size=1, activation_fn="linear")

        self._last_value = None

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs",
            shift="-{}:0".format(num_frames - 1),
            space=obs_space)
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space)

    def forward(self, input_dict, states, seq_lens):
        # import pdb;pdb.set_trace()
        obs = input_dict["prev_n_obs"]
        s1 = torch.unsqueeze(states[0], 0)
        s2 = torch.unsqueeze(states[1], 0)
        self.features, [h, c] = self.lstm(obs, [s1, s2])
        self.features = torch.reshape(self.features, [-1, self.features.shape[1] * self.features.shape[2]])
        out = self.out(self.features)
        self._last_value = self.values(self.features)
        t1 = torch.squeeze(h, 0)
        t2 = torch.squeeze(c, 0)
        return out, [t1, t2]

    def value_function(self):
        return torch.squeeze(self._last_value, -1)

    def get_initial_state(self):
        # h = [
        #     torch.zeros(self.lstm_size),
        #     torch.zeros(self.lstm_size)
        # ]
        # h = [
        #     torch.zeros(1, self.lstm_size),
        #     torch.zeros(1, self.lstm_size)
        # ]
        h = self.lstm.weight_hh_l0.data.fill_(0)
        return h

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

# To run the Trainer without tune.run, using our LSTM model and
# manual state-in handling, do the following:

# Example (use `config` from the above code):

"""
trainer = PPOTrainer(config)
lstm_cell_size = config["model"]["custom_model_config"]["cell_size"]
env = StatelessCartPole()
obs = env.reset()

# range(2) b/c h- and c-states of the LSTM.
init_state = state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)]
prev_a = 0
prev_r = 0.0

while True:
    #a, state_out, _ = trainer.compute_single_action(obs, state, prev_a, prev_r)
    a, state_out, _ = trainer.compute_single_action(obs, state)
    obs, reward, done, _ = env.step(a)
    if done:
        obs = env.reset()
        state = init_state
        prev_a = 0
        prev_r = 0.0
    else:
        state = state_out
        prev_a = a
        prev_r = reward
"""

results = tune.run('PPO',
                   config=config,
                   stop=stop,
                   verbose=2)

ray.shutdown()
