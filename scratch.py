#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict
os.environ['MUJOCO_GL'] = 'egl'  # or try 'osmesa' if egl doesn't work
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Full tracebacks
os.environ['JAX_DEBUG_NANS'] = 'True'  # Raise errors on NaNs
os.environ['JAX_ENABLE_X64'] = 'True'  # Use double precision
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'  # More detailed XLA info
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Force CPU execution

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model



#make model
mj_model = mujoco.MjModel.from_xml_path("bitcraze_crazyflie_2/scene_mjx.xml")
# Make model, data, and renderer
#mj_model = mujoco.MjModel.from_xml_string(xml)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

# Option 1: Isometric view (recommended for general viewing)
cam.lookat = [0, 0, 0.1]    # Look at where the Crazyflie is (z=0.1)
cam.distance = 1.5          # Much closer than 10
cam.elevation = -30         # Look down at an angle
cam.azimuth = 45           # Rotate around for nice perspective

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3  # (seconds)
framerate = 60  # (Hz)


frames = []
mujoco.mj_resetData(mj_model, mj_data)

# Initial control values (on ground)
initial_thrust = 0.0
hover_thrust = 0.265  # From your keyframe data
takeoff_duration = 1.5  # seconds to reach hover

# Print actuator info to see the control structure
print(f"Number of actuators: {mj_model.nu}")
print("Actuator names:")
for i in range(mj_model.nu):
    print(f"  {i}: {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

while mj_data.time < duration:
  # Calculate gradual takeoff thrust
  if mj_data.time < takeoff_duration:
    thrust_progress = mj_data.time / takeoff_duration
    current_thrust = initial_thrust + (hover_thrust - initial_thrust) * thrust_progress
  else:
    # Maintain hover thrust with small variations for realism
    current_thrust = hover_thrust + 0.01 * np.sin(2 * np.pi * 0.5 * mj_data.time)
  
  # Control all 3 drones (12 actuators total: 4 per drone)
  # Drone 1 (actuators 0-3): thrust, x_moment, y_moment, z_moment
  mj_data.ctrl[0] = 2  # cf2_1_thrust
  mj_data.ctrl[1] = 0.0            # cf2_1_xmoment
  mj_data.ctrl[2] = 0.0            # cf2_1_ymoment
  mj_data.ctrl[3] = 0.0            # cf2_1_zmoment
  
  # Drone 2 (actuators 4-7)
  mj_data.ctrl[4] = 2  # cf2_2_thrust
  mj_data.ctrl[5] = 0.0            # cf2_2_xmoment
  mj_data.ctrl[6] = 0.0            # cf2_2_ymoment
  mj_data.ctrl[7] = 0.0            # cf2_2_zmoment
  
  # Drone 3 (actuators 8-11)
  mj_data.ctrl[8] = 2   # cf2_3_thrust
  mj_data.ctrl[9] = 0.0             # cf2_3_xmoment
  mj_data.ctrl[10] = 0.0            # cf2_3_ymoment
  mj_data.ctrl[11] = 0.0            # cf2_3_zmoment
  
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    renderer.update_scene(mj_data, scene_option=scene_option, camera=cam)
    pixels = renderer.render()
    frames.append(pixels)

# Simulate and display video.
try:
    media.write_video(path="gifs/mjx_video.mp4", images=frames, fps=framerate)
    print("done!")
    # Explicitly delete renderer before exit
    #del renderer
except Exception as e:
    print("done!")

#mjx
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
  mjx_data = jit_step(mjx_model, mjx_data)
  if len(frames) < mjx_data.time * framerate:
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)
    print(mjx_data.time)

try:
    media.write_video(path="gifs/mjx_video_jit.mp4", images=frames, fps=framerate)
    print("done!")
    # Explicitly delete renderer before exit
    del renderer
except Exception as e:
    print("done!")


