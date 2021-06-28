import numpy as np
from gym.utils import EzPickle
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os
from utils.misc import quat2rpy, quat_dist # get from mujoco

DEFAULT_CAMERA_CONFIG = {'distance': 3.0, 'trackbodyid': 1, 'elevation': 0}

class LaikagoEnv(MujocoEnv, EzPickle):
  def __init__(self,
               xml_path=None,
               frame_skip=5,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.,
               bad_contact_cost = 1.):
    EzPickle.__init__(**locals())
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._bad_contact_cost = bad_contact_cost

    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago.xml')
    else:
      self._xml_path = xml_path
    MujocoEnv.__init__(self, self._xml_path, frame_skip=frame_skip)
    self.init_qpos = self.sim.model.key_qpos[0]
    self.init_qvel = self.sim.model.key_qvel[0]


  def step(self, action, output=True):
    if action.size == 6: # symmetric
      action = np.hstack([action, action])

    prev_state = self._get_obs()
    self.do_simulation(action, self.frame_skip)
    curr_state = self._get_obs()

    # Velocity
    x_after = curr_state[0]
    x_before = prev_state[0]
    x_vel = (x_after - x_before)/(self.dt)
    forward_reward = self._forward_reward_weight * x_vel

    # Penalty for bad contacts
    bad_contact_cost = 0 # default
    for c in range(len(self.sim.data.contact)):
      contact = self.sim.data.contact[c]
      if contact.dist == 0.0: # end of contacts
        break
      if ('calf' not in self.env.sim.model.geom_id2name(contact.geom1)) and \
          ('calf' not in self.env.sim.model.geom_id2name(contact.geom2)):
        # print(f"Bad contact found between {self.env.sim.model.geom_id2name(contact.geom1)} and {self.env.sim.model.geom_id2name(contact.geom2)}")
        bad_contact_cost = self._bad_contact_cost
        break # skip checking other contacts

    # Orientation    
    # x_ori_before, y_ori_before, z_ori_before = quat2rpy(prev_state[3:7])
    # x_ori_after, y_ori_after, z_ori_after = quat2rpy(curr_state[3:7])
    # y_orientation_cost = 1* np.abs(y_ori_after) + 0.5 * ((y_ori_after - y_ori_before)/self.dt)
    # x_orientation_cost = 1* np.abs(x_ori_after) + 0.5 * ((x_ori_after - x_ori_before)/self.dt)
    # z_orientation_cost = 1* np.abs(z_ori_after) + 0.5 * ((z_ori_after - z_ori_before)/self.dt)
    # orientation_cost = y_orientation_cost\
    #                   + 0.1 * x_orientation_cost\
    #                   + 0.1 * z_orientation_cost
    orientation_cost = quat_dist(curr_state[3:7], np.array([0.5, 0.5, 0.5, 0.5]))


    # Action
    ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

    # Total    
    reward = forward_reward - ctrl_cost - orientation_cost - bad_contact_cost # from gym's half cheetah

    obs = curr_state
    done = False
    info = {
      'x_position': x_after,
      'x_velocity': x_vel,
      'reward_run': forward_reward,
      'reward_ctrl': -ctrl_cost,
      'reward_orientation': -orientation_cost,
      'reward_bad_contact': -bad_contact_cost,
    }
    return obs, reward, done, info


  def _get_obs(self):
    position = self.sim.data.qpos.flat.copy()
    velocity = self.sim.data.qvel.flat.copy()
    observation = np.concatenate((position, velocity)).ravel()
    return observation


  def reset_model(self):
    noise_low = -self._reset_noise_scale
    noise_high = self._reset_noise_scale

    qpos = self.init_qpos + self.np_random.uniform(
      low=noise_low, high=noise_high, size=self.model.nq)
    qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
      self.model.nv)

    self.set_state(qpos, qvel)

    observation = self._get_obs()
    return observation


  def viewer_setup(self):
    for key, value in DEFAULT_CAMERA_CONFIG.items():
      if isinstance(value, np.ndarray):
        getattr(self.viewer.cam, key)[:] = value
      else:
        setattr(self.viewer.cam, key, value)

  def get_obs(self):
    return self._get_obs()