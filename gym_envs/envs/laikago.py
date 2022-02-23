import numpy as np
import gym
from gym.utils import EzPickle
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import error
import traceback

import os
from os import path

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )
from mujoco_py import functions



DEFAULT_CAMERA_CONFIG = {'distance': 3.0, 'trackbodyid': 1, 'elevation': 0}

def quat_dist(q1: np.ndarray, q2: np.ndarray) -> float:
  """Computes the distance between two unit quaternions."""
  assert np.isclose(np.linalg.norm(q1), 1.)
  assert np.isclose(np.linalg.norm(q2), 1.)
  assert q1.size == 4
  assert q2.size == 4

  q2_inv = np.zeros(4)
  q_diff = np.zeros(4)
  functions.mju_negQuat(q2_inv, q2)
  functions.mju_mulQuat(q_diff, q1, q2_inv)
  if np.isclose(np.fabs(q_diff[0]), 1.):
    return 0.
  else:
    return 2 * np.arctan2(np.linalg.norm(q_diff[1:]), q_diff[0])

def get_stage(step_height = 0.0):
  assert np.isclose((step_height*20)%(0.05*20), 0., atol=0.001), f"Step ht ({step_height}) has to be a multiple of 0.05!!"
  stage = int(np.rint(step_height/0.05))
  return stage

class LaikagoEnv(MujocoEnv, EzPickle):
  def __init__(self,
               xml_path=None,
               frame_skip=5,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.,
               bad_contact_cost = 0.):
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
      if ('calf' not in self.sim.model.geom_id2name(contact.geom1)) and \
          ('calf' not in self.sim.model.geom_id2name(contact.geom2)):
        # print(f"Bad contact found between {self.sim.model.geom_id2name(contact.geom1)} and {self.sim.model.geom_id2name(contact.geom2)}")
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

    h_after = curr_state[2]
    h_before = prev_state[2]
    ht_cost = 0#0.02*np.abs(h_after - h_before)


    # Total
    reward = forward_reward - ht_cost - ctrl_cost - orientation_cost - bad_contact_cost # from gym's half cheetah

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
    position = self.sim.data.qpos.flat.copy()[:19] # only for first LK
    velocity = self.sim.data.qvel.flat.copy()[:18] # only for first LK
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


class LaikagoOverlayEnv(LaikagoEnv):
  def __init__(self,
               xml_path=None,
               frame_skip=5,
               forward_reward_weight=1.0,
               ctrl_cost_weight=0.1,
               reset_noise_scale=0.,
               bad_contact_cost = 1.):
    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago_overlay.xml')
    else:
      self._xml_path = xml_path
    super(LaikagoOverlayEnv, self).__init__(xml_path=self._xml_path,
                                            reset_noise_scale=reset_noise_scale,
                                            frame_skip=frame_skip,
                                            forward_reward_weight=forward_reward_weight,
                                            ctrl_cost_weight=ctrl_cost_weight,
                                            bad_contact_cost=bad_contact_cost)

  ### NOT NEEDED. We only give actions to main LK, not overlay LK
  # def step(self, action):
  #   if action.size == 12: # symmetric
  #     actpol = action[:6]
  #     acttraj = action[6:]
  #     action = np.hstack([actpol, actpol, acttraj, acttraj])

  #   self.do_simulation(action, self.frame_skip)

  #   reward = 0
  #   obs = self._get_obs()
  #   done = False
  #   info = {
  #     'x_position': 0,
  #     'x_velocity': 0,
  #     'reward_run': 0,
  #     'reward_ctrl': 0,
  #     'reward_orientation': 0,
  #     'reward_bad_contact': 0,
  #   }
  #   return obs, reward, done, info

  def _get_obs(self):
    position = self.sim.data.qpos.flat.copy()[:19] # only for first LK
    velocity = self.sim.data.qvel.flat.copy()[:18] # only for first LK
    observation = np.concatenate((position, velocity)).ravel()
    return observation


class Laikagov2Env(MujocoEnv, EzPickle):
  """This is a class for Laikago which has the overlay running all the time.
  The states for the overlay environment are picked from the reference trajectories.
  """
  def __init__(self,
              xml_path=None,
              frame_skip=5,

              forward_reward_weight=1.0,
              ctrl_cost_weight=0.1,
              orientation_cost_weight=0.3,
              reset_noise_scale=0.,
              bad_contact_cost = 0.,

              task_name='jumpup',
              path_to_trajs=None,

              randomize_step_height=True,
              randomize_step_location=True,

              sensing_threshold = 4.0,
              enable_overlay = True,
              max_episode_steps = 400,

              sparse_reward = False,
              fwd_reward_only = False,
              ):
    EzPickle.__init__(**locals())

    # set the weights and scales
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._bad_contact_cost = bad_contact_cost
    self._orientation_cost_weight = orientation_cost_weight
    self.fwd_reward_only = fwd_reward_only

    # set the task name
    # can be ['jumpup', 'hurdle', 'stairs'], default is 'hurdle'
    assert task_name in ['jumpup', 'hurdle', 'stairs'], 'task_name should be one of jumpup, hurdle, stairs!!'
    self._task_name = task_name

    # set other class variables
    self._sensing_threshold = sensing_threshold
    self._enable_overlay = enable_overlay
    self._max_episode_steps = max_episode_steps
    self._sparse_reward = sparse_reward

    # set the path to the trajectories
    if path_to_trajs is None:
      raise ValueError('path_to_trajs is None!, For this env, it should be set!')
    self._path_to_trajs = path_to_trajs

    # load the trajectories
    self.load_trajs()

    # set the xml path based on the task name
    if xml_path is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      if self._task_name == 'hurdle':
        self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago_hurdle_1.xml')
      elif self._task_name == 'jumpup':
        self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago_jumpup_1.xml')
      elif self._task_name == 'stairs':
        self._xml_path = os.path.join(curr_dir, 'assets', 'laikago', 'laikago_stairs_1.xml')
    else:
      if self._task_name is not None:
        print(f"Note: task_name is {self._task_name}. Be sure the provided xml is for this task!")
      self._xml_path = xml_path

    # initialize the Mujoco env
    # MujocoEnv.__init__(self, self._xml_path, frame_skip=frame_skip)
    if self._xml_path.startswith("/"):
      fullpath = self._xml_path
    else:
      fullpath = os.path.join(os.path.dirname(__file__), "assets", self._xml_path)
    if not path.exists(fullpath):
      raise IOError("File %s does not exist" % fullpath)
    self.frame_skip = frame_skip
    self.model = mujoco_py.load_model_from_path(fullpath)
    self.sim = mujoco_py.MjSim(self.model)
    self.data = self.sim.data
    self.viewer = None
    self._viewers = {}

    # initialize the task obstacle indices
    self._init_task_object_indices()

    # initialize the robot indices
    self._init_robot_indices()

    self.init_qpos = self.sim.model.key_qpos[0]
    self.init_qvel = self.sim.model.key_qvel[0]

    # The default location of the obstacle
    if self._task_name == 'hurdle':
      self.cma_step_loc = self.model.body_pos[self.step0_idx][0] # same as step0_xpos
    elif self._task_name == 'jumpup':
      self.cma_step_loc = self.model.body_pos[self.step0_idx][0] # same as step0_xpos
    elif self._task_name == 'stairs':
      self.cma_step_loc = self.model.body_pos[self.step0_idx][0] # same as step0_xpos
      # raise NotImplementedError('stairs is not implemented yet!')

    # set obstacle height and location parameters
    self._randomize_step_height = randomize_step_height
    self._randomize_step_location = randomize_step_location
    self._set_obstacle_distribution()

    # set success thresholds
    self._set_obstacle_success_parameters()

    # reset the obstacle height and location
    self.reset_obstacle()

    # reset the model, this also sets a lot of other cumulative variables
    self.reset_model()

    # reset the overlay
    self.reset_overlay()

    # Note: By default, the robot will never zero reset to a location with x_offset of the step location.
    self.robot_has_crossed_x_offset = False

    self.metadata = {
        "render.modes": ["human", "rgb_array", "depth_array"],
        "video.frames_per_second": int(np.round(1.0 / self.dt)),
    }

    self._set_action_space()

    action = self.action_space.sample()
    observation, _reward, done, _info = self.step(action)
    assert not done

    self._set_observation_space(observation)

    self.seed()


  def _set_obstacle_success_parameters(self):
    """ Sets the x offset the robot must get to ahead of the obstacle in order to succeed.
    """
    if self._task_name == 'hurdle':
      self.x_beyond_obstacle = 1
    elif self._task_name == 'jumpup':
      self.x_beyond_obstacle = 1
    elif self._task_name == 'stairs':
      self.x_beyond_obstacle = 5
      # raise NotImplementedError('stairs is not implemented yet!')


  def _set_obstacle_distribution(self):
    """ Set the max and min values for obstacle height and location
    """
    # get the obstacle height and location
    if self._task_name == 'hurdle':
      self.step_min_height = 0.0
      self.step_max_height = 0.4
      self.step_default_height = 0.2
      self.step_min_location = 0.0
      self.step_max_location = 4.0
      self.step_default_location = 0.0
      self.step_inc = 0.05
    elif self._task_name == 'jumpup':
      self.step_min_height = 0.0
      self.step_max_height = 0.4
      self.step_default_height = 0.2
      self.step_min_location = 0.0
      self.step_max_location = 4.0
      self.step_default_location = 0.0
      self.step_inc = 0.05
    elif self._task_name == 'stairs':
      self.step_min_height = 0.0
      self.step_max_height = 0.4
      self.step_default_height = 0.2
      self.step_min_location = 0.0
      self.step_max_location = 4.0
      self.step_default_location = 0.0
      self.step_inc = 0.05
      # raise NotImplementedError('stairs task is not implemented yet!')


  def _init_robot_indices(self):
    self.env_qpos_idxs = np.array(range(int(self.sim.model.nq/2))) # qpos of first LK
    self.env_qvel_idxs = self.sim.model.nq - len(self.env_qpos_idxs) + np.array(range(int(self.sim.model.nv/2))) # qvel of first LK
    self.env_pos_idxs = np.array(range(3)) # xyz distance
    self.robot_height_index = 2



  def _init_task_object_indices(self):
    """ This function initializes the task obstacle indices. So that these can used to set the stage of the task.
    """
    if self._task_name == 'hurdle':
      self.step0_idx = self.model.body_name2id('step0')
      self.step0geom_idx = self.model.geom_name2id('step0_geom')
      self.step0_xpos = self.model.body_pos[self.step0_idx][0]
    elif self._task_name == 'jumpup':
      self.step0_idx = self.model.body_name2id('step0')
      self.step0geom_idx = self.model.geom_name2id('step0_geom')
      self.step0_xpos = self.model.body_pos[self.step0_idx][0]
    elif self._task_name == 'stairs':
      # step 0
      self.step0_idx = self.model.body_name2id('step0')
      self.step0geom_idx = self.model.geom_name2id('step0_geom')
      self.step0_xpos = self.model.body_pos[self.step0_idx][0]

      # step 1
      self.step1_idx = self.model.body_name2id('step1')
      self.step1geom_idx = self.model.geom_name2id('step1_geom')
      self.step1_xpos = self.model.body_pos[self.step1_idx][0]

      # ...
      # raise NotImplementedError('Get the indices of the stairs task from the XML!')


  def load_trajs(self):
    """
    Load the trajectories from the file, depending on the task name. By default, only state trajectories are loaded.
    In the path_to_trajs, we have the tasks solved at increasing levels of difficulty, denoted by stage. For each task, we have a given number of stages, as defined below.
    """
    if self._task_name == 'hurdle':
      self.num_sub_tasks = 9
    elif self._task_name == 'jumpup':
      self.num_sub_tasks = 9
    elif self._task_name == 'stairs':
      self.num_sub_tasks = 9
    # load the trajectories
    self.datasets = [Dataset(base_folder=self._path_to_trajs, robot_name = 'laikago', task = self._task_name, stage = i, load_actions=True) for i in range(self.num_sub_tasks)]


  def set_stage(self, stage):
    """
    Set the stage of the task. By doing this, we can set the heights of the given obstacles
    """
    assert stage in range(self.num_sub_tasks), f"stage should be in {range(self.num_sub_tasks)} for task {self._task_name}"
    self.stage = stage

    if self._task_name == 'hurdle':
      self.set_hurdle_stage(stage)
    elif self._task_name == 'jumpup':
      self.set_jumpup_stage(stage)
    elif self._task_name == 'stairs':
      self.set_stairs_stage(stage)

  def set_hurdle_height(self, height):
    """ Set the height of the hurdle obstacle
    """
    self.model.body_pos[self.step0_idx, 2] = height
    self.sim.forward() # won't increment timestep by 1


  def set_hurdle_stage(self, stage):
    """ Set the stage of the hurdle task by setting the height of the Step0 obstacle. Each stage increases the height of the obstacle by 0.05 m.
    """
    # caclulate the height of the obstacle
    step_height = 0.05 * stage
    self.set_hurdle_height(step_height)


  def set_jumpup_height(self, height):
    """ Set the height of the jumpup obstacle
    """
    self.model.body_pos[self.step0_idx, 2] = height
    self.sim.forward() # won't increment timestep by 1


  def set_jumpup_stage(self, stage):
    """ Set the stage of the jumpup task by setting the height of the Step0 obstacle. Each stage increases the height of the obstacle by 0.05 m.
    """
    # caclulate the height of the obstacle
    step_height = 0.05 * stage
    self.set_jumpup_height(step_height)


  def set_stairs_height(self, height):
    """ Set the height of the stairs obstacle
    """
    self.model.body_pos[self.step0_idx, 2] = height
    self.model.body_pos[self.step1_idx, 2] = height + 0.1
    self.sim.forward() # won't increment timestep by 1

    # raise NotImplementedError('Set the height of the stairs obstacle!')


  def set_stairs_stage(self, stage):
    step_height = 0.05 * stage
    self.set_stairs_height(step_height)
    # raise NotImplementedError('Set the stage of the stairs task!')


  def set_obstacle_height(self, height):
    """ Set the height of the Obstacle.
    For the hurdle task, this is the height of the Step0 obstacle.
    For the jumpup task, this is the height of the Step0 obstacle.
    For the stairs task, this is the height of each step the robot has to take.
    """
    if self._task_name == 'hurdle':
      self.set_hurdle_height(height)
    elif self._task_name == 'jumpup':
      self.set_jumpup_height(height)
    elif self._task_name == 'stairs':
      self.set_stairs_height(height)


  def set_jumpup_location(self, location):
    """ Set the location of the jumpup obstacle
    """
    self.model.body_pos[self.step0_idx, 0] = location
    self.sim.forward() # won't increment timestep by 1


  def set_hurdle_location(self, location):
    """ Set the location of the hurdle obstacle
    """
    self.model.body_pos[self.step0_idx, 0] = location
    self.sim.forward() # won't increment timestep by 1


  def set_stairs_location(self, location):
    """ Set the location of the stairs obstacle
    """
    self.model.body_pos[self.step0_idx, 0] = location
    self.model.body_pos[self.step1_idx, 0] = location + (self.step1_xpos - self.step0_xpos)
    self.sim.forward() # won't increment timestep by 1
    # raise NotImplementedError('Set the location of the stairs obstacle!')


  def set_obstacle_location(self, location):
    """ Set the location of the obstacle.
    For the hurdle task, this is the location of the Step0 obstacle.
    For the jumpup task, this is the location of the Step0 obstacle.
    For the stairs task, this is the location of staircase.
    """
    if self._task_name == 'hurdle':
      self.set_hurdle_location(location)
    elif self._task_name == 'jumpup':
      self.set_jumpup_location(location)
    elif self._task_name == 'stairs':
      self.set_stairs_location(location)


  def _set_action_space(self):
    """ Override the default action space. The Default action space is 12 DOF, since there are 12 actuators.
      We have a symmetric action space, so we only need 6 DOF. Same actions will be applied to both Left and Right
      legs of the LK.

      Returns:
        action_space (gym.spaces.Box): The action space of the environment.
    """
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,)) # Hardcoded for Laikago

    return self.action_space

  def _get_obs(self):
    """ Gets the observation as the qpos and qvel of the robot only.

    Returns:
        numpy array: Observation array of the robot.
    """
    position = self.sim.data.qpos.flat.copy()[:self.env_qpos_idxs.shape[0]] # only for first 19 are for the real LK, the rest are from overlay LK
    velocity = self.sim.data.qvel.flat.copy()[:self.env_qvel_idxs.shape[0]] # only for first 18 are for the real LK, the rest are from overlay LK
    observation = np.concatenate((position, velocity)).ravel()
    return observation


  def set_robot_state(self, state):
    """ Set the state of the robot only.
    """
    overlay_qpos = self.sim.data.qpos.flat.copy()[self.env_qpos_idxs.shape[0]:]
    overlay_qvel = self.sim.data.qvel.flat.copy()[self.env_qvel_idxs.shape[0]:]

    new_robot_qpos = state[:self.env_qpos_idxs.shape[0]]
    new_robot_qvel = state[:self.env_qvel_idxs.shape[0]]

    new_qpos = np.concatenate((new_robot_qpos, overlay_qpos))
    new_qvel = np.concatenate((new_robot_qvel, overlay_qvel))
    self.set_state(new_qpos, new_qvel)


  def reset_model(self):
    """ Resets the robot to its initial state after adding a little noise if needed.

    Returns:
        numpy array: Observation array of the robot after reset.
    """
    noise_low = -self._reset_noise_scale
    noise_high = self._reset_noise_scale

    qpos = self.init_qpos + np.random.uniform(
      low=noise_low, high=noise_high, size=self.model.nq)
    qvel = self.init_qvel + self._reset_noise_scale * np.random.randn(
      self.model.nv)

    self.set_state(qpos, qvel)

    self.step_number = 0
    self.max_height = 0
    self.min_height = 99999

    self.robot_has_crossed_x_offset = False
    self.cumulative_dense_reward = 0

    self.gym_rew_fwd = 0
    self.gym_rew_ctrl = 0
    self.gym_rew_orientation = 0


    observation = self._get_obs()
    return observation


  def reset_obstacle(self, step_height=None, step_location = None):
    """ Resets the Obstale

    Args:
      step_height (float): The height of the obstacle.
      step_location (float): The location of the obstacle.
    """
    if step_height is not None:
      self.step_height = step_height
    else:
      if self._randomize_step_height:
        self.step_height = np.random.uniform(self.step_min_height, self.step_max_height)
      else:
        self.step_height = self.step_default_height

    if step_location is not None:
      self.step_location = step_location + self.step0_xpos
    else:
      if self._randomize_step_location:
        self.step_location = np.random.uniform(self.step_min_location, self.step_max_location) + self.step0_xpos
      else:
        self.step_location = self.step_default_location + self.step0_xpos

    self.set_obstacle_height(self.step_height)
    self.set_obstacle_location(self.step_location)

    # store the offset between the obstacles new location and the original location
    self.x_offset = self.model.body_pos[self.step0_idx, 0] - self.cma_step_loc

    # deduce what should be the stage of the obstacle in the overlay
    nearest_larger_step = np.ceil(self.step_height/self.step_inc) * self.step_inc
    self.stage = get_stage(nearest_larger_step)
    self.dataset = self.datasets[self.stage]


  def reset(self, step_height = None, step_location = None):
    """ Reset the robot to its inital state and set the height of the obstacle if needed.

    Args:
        step_height (float): The height of the obstacle.
        step_location (float): The location of the obstacle, relative to the xpos of the obstacle in the xml file.

    Returns:
        dict: Observation dictionary of the robot after reset.
    """
    # reset the simulator
    self.sim.reset()

    # reset the model and get the joint states
    joint_states = self.reset_model()

    # reset the obstacle
    self.reset_obstacle(step_height, step_location)

    # reset the overlay
    self.reset_overlay()


    return self.get_augmented_observation()


  def get_augmented_observation(self):
    """ Get the observation of the robot.

    Returns:
        dict: Observation dictionary of the robot.
    """
    joint_states = self._get_obs()

    # get the observation as a dictionary of the joint states and it's sensing of the obstacle
    if self._task_name == 'hurdle':
      self.relative_location_of_left_edge = (self.model.body_pos[self.step0_idx, 0] - self.model.geom_size[self.step0geom_idx, 0]) - joint_states[0]
    elif self._task_name == 'jumpup':
      self.relative_location_of_left_edge = (self.model.body_pos[self.step0_idx, 0] - self.model.geom_size[self.step0geom_idx, 0]) - joint_states[0]
    elif self._task_name == 'stairs':
      self.relative_location_of_left_edge = (self.model.body_pos[self.step0_idx, 0] - self.model.geom_size[self.step0geom_idx, 0]) - joint_states[0]
      # raise NotImplementedError('Sensing not defined for stairs!')

    augmented_obs = {}
    augmented_obs['obs'] = joint_states
    augmented_obs['step_rel_loc'] = np.array([self.relative_location_of_left_edge]).clip(-np.inf, self._sensing_threshold)
    if augmented_obs['step_rel_loc'] > self._sensing_threshold:
      augmented_obs['stepht'] = 0
    else:
      augmented_obs['stepht'] = np.array([self.step_height])

    augmented_obs['overlay_time'] = np.array([self.overlay_step_number])
    return augmented_obs


  def viewer_setup(self):
    """ Setup the viewer.
    """
    for key, value in DEFAULT_CAMERA_CONFIG.items():
      if isinstance(value, np.ndarray):
        getattr(self.viewer.cam, key)[:] = value
      else:
        setattr(self.viewer.cam, key, value)


  def get_obs(self):
    """ Returns the observation.

    Returns:
        numpy array: Observation array of the robot.
    """
    return self._get_obs()


  def set_overlay_state(self, imi_qpos, imi_qvel):
    """ Set the overlay to the desired state

    Args:
        qpos (numpy array): The position of the overlay.
        qvel (numpy array): The velocity of the overlay.
    """
    curr_state = self._get_obs()
    qpos = curr_state[self.env_qpos_idxs]
    qvel = curr_state[self.env_qvel_idxs]
    total_qpos = np.concatenate([qpos, imi_qpos])
    total_qvel = np.concatenate([qvel, imi_qvel])
    self.set_state(total_qpos, total_qvel)


  def step_overlay(self):
    """ Updates the overlay based on the current overlay_step_number
    """
    self.overlay_state = self.dataset.reference_states[self.overlay_step_number].copy()
    self.overlay_state[0] += self.x_offset
    if self._enable_overlay:
      imi_qpos = self.overlay_state[self.env_qpos_idxs]
      imi_qvel = self.overlay_state[self.env_qvel_idxs]
      self.set_overlay_state(imi_qpos, imi_qvel)

  def move_overlay(self,):
    """ Step the overlay model.
    """
    if self.overlay_step_number < (self.dataset.reference_states.shape[0] - 1):
      self.overlay_step_number += 1

    self.step_overlay()


  def reset_overlay(self):
    """ Resets the overlay to its starting position
    """
    # we set it to 1 since we want the state to which we arrive at each step.
    self.overlay_step_number = 1

    self.step_overlay()


  def should_move_overlay(self):
    """ The overlay will move if the robot has crossed the x_offset of if it has ever crossed it before
    """
    if self.robot_has_crossed_x_offset:
      return True

    if self._get_obs()[0] >= self.dataset.reference_states[0].copy()[0] + self.x_offset:
      self.robot_has_crossed_x_offset = True
      return True
    else:
      return False



  def should_terminate(self):
    """ Check if the episode should terminate.
    The episode will terminate if the environment counter has reached max_steps.
    """
    if self.step_number >= self._max_episode_steps:
      return True
    else:
      return False


  def calculate_sparse_reward(self, prev_state, curr_state, action):
    """ Calculate the sparse reward for the current state.
    """
    self.cumulative_dense_reward += self.calculate_dense_reward(prev_state, curr_state, action)
    # If robot is on the left of the success x position, the reward is 0
    if (self.relative_location_of_left_edge < -self.x_beyond_obstacle):
      # and (self.relative_location_of_left_edge > (-self.x_beyond_obstacle-1)):
      return 1 - (self.orientation_cost * self._orientation_cost_weight/np.pi) - (self.ctrl_cost)#self.cumulative_dense_reward
    else:
      return 0 - (self.orientation_cost * self._orientation_cost_weight/np.pi) - (self.ctrl_cost)



  def calculate_dense_reward(self, prev_state, curr_state, action):
    """ Calculate the Dense reward of the current step.

    Args:
        prev_state (numpy array): The state of the robot at the previous step.
        curr_state (numpy array): The state of the robot at the current step.
        action (numpy array): The action of the robot at the current step.

    Returns:
        float: The reward of the current step.
    """
    # Velocity
    self.x_after = curr_state[0]
    x_before = prev_state[0]
    self.x_vel = (self.x_after - x_before)/(self.dt)
    self.forward_reward = self._forward_reward_weight * self.x_vel

    # Orientation
    # TODO: Check if the indices used for the state are correct
    self.orientation_cost = quat_dist(curr_state[3:7], np.array([0.5, 0.5, 0.5, 0.5]))

    # Action
    self.ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

    # Total
    if self.fwd_reward_only:
        reward = self.forward_reward
    else:
        reward = self.forward_reward - self.ctrl_cost - self.orientation_cost

    self.gym_rew_fwd += self.forward_reward
    self.gym_rew_ctrl += (-self.ctrl_cost)
    self.gym_rew_orientation += (-self.orientation_cost)

    return reward


  def calculate_reward(self, prev_state, curr_state, action):
    """ Calculate the reward of the current step.

    Args:
        prev_state (numpy array): The state of the robot at the previous step.
        curr_state (numpy array): The state of the robot at the current step.
        action (numpy array): The action of the robot at the current step.

    Returns:
        float: The reward of the current step.
    """
    if self._sparse_reward:
      return self.calculate_sparse_reward(prev_state, curr_state, action)
    else:
      return self.calculate_dense_reward(prev_state, curr_state, action)


  def get_info(self):
    """ Returns the info of the environment.
    """
    info = {
      'x_position': self.x_after,
      'x_velocity': self.x_vel,
      'reward_run': self.forward_reward,
      'reward_ctrl': -self.ctrl_cost,
      'reward_orientation': -self.orientation_cost,
      'cumulative_reward': self.cumulative_dense_reward,

      'episode_extra_stats': {
        'episode_max_height': self.max_height,
        'episode_min_height': self.min_height,
        'step_height': self.step_height,
      }
    }
    return info


  def step(self, action):
    """ Step the environment.

    Args:
        action (numpy array): Action array of the robot.

    Returns:
        dict: Observation dictionary of the robot.
        float: Reward.
        bool: Whether the episode is done.
        dict: Additional information.
    """
    action = np.hstack([action, action]) # duplicate the action for the left and right legs
    prev_state = self._get_obs()
    try:
      self.do_simulation(action, self.frame_skip)
    except mujoco_py.MujocoException:
      print(f"Mujoco Exception!\n for state: {prev_state} \n and action: {action}")
      traceback.print_exc()
      return self.get_augmented_observation(), 0, True, self.get_info()
    curr_state = self._get_obs()

    # update some info
    self.max_height = max(self.max_height, curr_state[self.robot_height_index])
    self.min_height = min(self.min_height, curr_state[self.robot_height_index])


    # increase the step counter
    self.step_number += 1

    # move the overlay model if required
    if self.should_move_overlay():
        self.move_overlay()

    observation = self.get_augmented_observation()

    # calculate the reward
    reward = self.calculate_reward(prev_state, curr_state, action)

    done = self.should_terminate()

    info = self.get_info()

    return observation, reward, done, info


class Dataset(object):
    def __init__(self, base_folder = "CMAES_helper/data", robot_name = "laikago", task = None, stage = 0, load_actions = False):
        assert task in ["jumpup", "stairs", "hurdle"], f"Unknown task {task}. Choices are [jumpup, stairs, hurdle]"
        self.base_folder = os.path.join(*[base_folder, robot_name, task])
        self.robot_name = robot_name
        self.stage = stage
        self.steps_per_episode = 0
        self._load_actions = load_actions


        self.folders = os.listdir(self.base_folder)
        if ".DS_Store" in self.folders: self.folders.remove(".DS_Store") # macOS hack

        try:
          self.folders = [folder for folder in self.folders if folder.split('_')[2] == str(self.stage)]
        except:
          raise NameError(f"Error in parsing some folder in this list\n{self.folders}")

        assert len(self.folders)>0 , "Stage {} data does not exist in {}!!!".format(self.stage, self.base_folder)

        self.load_trajs()
        # self.state_shape = self.trajs[self.costs[0]][0].shape[1]
        # self.action_shape = self.trajs[self.costs[0]][1].shape[1]
        self.sort_trans()

    def load_trajs(self):
        # all_trajectories = {}
        all_transitions = []
        self.start_x_states = []
        self.start_states = []
        # all_transitions_dict = {}

        for folder_name in self.folders:
            folder_path = os.path.join(*[self.base_folder, folder_name])
            npz_file = [file for file in os.listdir(folder_path) if '_trajs.npz' in file][0]
            path_to_npz = os.path.join(*[folder_path, npz_file])
            trajectories = np.load(path_to_npz)
            # print([k for k in trajectories.keys()])
            # for k,cost in enumerate (trajectories['fitness_cost']):
            #     self.start_x_states.append(trajectories['states'][k][0][0])
            #     self.start_states.append(trajectories['states'][k][0])
            #     # all_trajectories[cost] = (trajectories['states'][k][1:], trajectories['actions'][k])
            #     for i in range(trajectories['actions'][k].shape[0]):
            #         state = trajectories['states'][k][i]
            #         next_state = trajectories['states'][k][i+1]
            #         action = trajectories['actions'][k][i]
            #         action = np.clip(action, -1, 1)
            #         all_transitions.append((state, action, next_state))
            #         # all_transitions_dict[state[0]] = (state, action)

        # self.trajs = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))
        self.trans = all_transitions # TODO: not needed?
        if self.robot_name == "half_cheetah":
            self.reference_states = trajectories['states'][0] # first motion = reference
            if self._load_actions:
              self.reference_actions = trajectories['actions'][0]
            self.traj_reward = -trajectories['fitness_cost'][0] # this should be a reward now
        else:
            self.reference_states = trajectories['states'][0][:401] # first motion = reference
            if self._load_actions:
              self.reference_actions = trajectories['actions'][0][:400]
            self.traj_reward = -trajectories['fitness_cost'][0] # this should be a reward now
        # self.trans_dict = all_transitions_dict
        # self.costs = list(self.trajs.keys())
        self.steps_per_episode = self.reference_states.shape[0] - 1

        # Make states and actions non-writeable after everything is loaded
        if self._load_actions:
          self.reference_actions.setflags(write=0)
        self.reference_states.setflags(write=0)

    def sort_trans(self):
        self.trans.sort(key=lambda x:x[2][0])
        # print("All transitions are now sorted!!")

    def sample_random_state(self):
        return np.random.choice(self.trans)[0]

    def get_closest_tuple_for_next_state(self, state):
        closest_state, action_state, closest_next_state = self.binarySearch(state[0], 2)
        return closest_state, action_state, closest_next_state

    def get_closest_tuple_for_current_state(self, state):
        closest_state, action_state, closest_next_state = self.binarySearch(state[0], 0)
        return closest_state, action_state, closest_next_state

    def binarySearch(self, val, tup_index):
        lo, hi = 0, len(self.trans) - 1
        best_ind = lo
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if self.trans[mid][tup_index][0] < val:
                lo = mid + 1
            elif self.trans[mid][tup_index][0] > val:
                hi = mid - 1
            else:
                best_ind = mid
                break
            # check if data[mid] is closer to val than data[best_ind]
            if abs(self.trans[mid][tup_index][0] - val) < abs(self.trans[best_ind][tup_index][0] - val):
                best_ind = mid
        return self.trans[best_ind]

    def sample_for_mimic(self, batch_size = 64):
        indices = np.random.permutation(len(self.trans))
        batch_indices = np.random.choice(indices,batch_size,replace=False)

        state_batch = np.array([self.trans[index][0] for index in batch_indices])
        action_batch = np.array([self.trans[index][1] for index in batch_indices])

        return state_batch, action_batch

