import numpy as np
from gym.utils import EzPickle
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
import os

class Walker2dObstacleEnv(Walker2dEnv):
  def __init__(self,
              xml_path=None,
              frame_skip = 2,
              forward_reward_weight=1.0,
              ctrl_cost_weight=1e-3,
              healthy_reward=1.0,
              terminate_when_unhealthy=True,
              healthy_z_range=(0.8, 2.0),
              healthy_angle_range=(-1.0, 1.0),
              reset_noise_scale=0.,
              exclude_current_positions_from_observation=True):
    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      xml_path = os.path.join(curr_dir, 'assets', 'walker2d_obstacle.xml')

    EzPickle.__init__(**locals())

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight

    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy

    self._healthy_z_range = healthy_z_range
    self._healthy_angle_range = healthy_angle_range

    self._reset_noise_scale = reset_noise_scale

    self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

    MujocoEnv.__init__(self, xml_path, frame_skip)
    self.init_qpos = self.sim.model.key_qpos[0]
    self.init_qvel = self.sim.model.key_qvel[0]

  def step(self, action, output=True):
    x_position_before = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    if output:
      x_position_after = self.sim.data.qpos[0]
      x_velocity = ((x_position_after - x_position_before)
                    / self.dt)

      ctrl_cost = self.control_cost(action)

      forward_reward = self._forward_reward_weight * x_velocity
      healthy_reward = self.healthy_reward

      rewards = forward_reward + healthy_reward
      costs = ctrl_cost

      observation = self._get_obs()
      reward = rewards - costs
      done = self.done
      info = {
          'x_position': x_position_after,
          'x_velocity': x_velocity,
      }

      return observation, reward, done, info
    else:
      return
