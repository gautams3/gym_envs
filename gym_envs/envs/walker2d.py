import numpy as np
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
import os

class Walker2dObstacleEnv(Walker2dEnv):
  def __init__(self,
              xml_file=None,
              reset_noise_scale=0.):
    if xml_file is None:
      curr_dir = os.path.dirname(os.path.realpath(__file__))
      xml_file = os.path.join(curr_dir, 'assets', 'walker2d_obstacle.xml')
    super(Walker2dObstacleEnv, self).__init__(xml_file=xml_file, reset_noise_scale=reset_noise_scale)
