import gym
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.utils import EzPickle
import numpy as np
import os

class HalfCheetahSoftEnv(HalfCheetahEnv):
  """HalfCheetah-v3 but with options for a different xml file and frame_skip"""
  _xml_path:str

  def __init__(self,
              xml_path = None,
              frame_skip = 5,
              forward_reward_weight=1.0,
              ctrl_cost_weight=0.1,
              reset_noise_scale=0.,
              exclude_current_positions_from_observation=True):
    # Completely replacing parent init function, it doesn't let us choose frame_skip
    EzPickle.__init__(**locals())
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'half_cheetah_soft.xml')
    else:
      self._xml_path = xml_path
    MujocoEnv.__init__(self, self._xml_path, frame_skip=frame_skip)

  def step(self, action):
    x_position_before = self.sim.data.qpos[0]
    contacts = self.sim.data.contact
    y_ori_before = self.sim.data.qpos[2]
    self.do_simulation(action, self.frame_skip)
    x_position_after = self.sim.data.qpos[0]
    y_ori_after = self.sim.data.qpos[2]

    x_velocity = ((x_position_after - x_position_before)
                  / self.dt)

    orientation_cost = 1* np.abs(y_ori_after) + 0.5 * ((y_ori_after - y_ori_before)/self.dt)

    ctrl_cost = self.control_cost(action)

    forward_reward = self._forward_reward_weight * x_velocity

    observation = self._get_obs()
    reward = forward_reward - ctrl_cost - orientation_cost
    done = False
    info = {
      'x_position': x_position_after,
      'x_velocity': x_velocity,
      'reward_run': forward_reward,
      'reward_ctrl': -ctrl_cost
    }

    return observation, reward, done, info
