#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict
os.environ['MUJOCO_GL'] = 'egl'  # or try 'osmesa' if egl doesn't work
# os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Full tracebacks
# os.environ['JAX_DEBUG_NANS'] = 'True'  # Raise errors on NaNs
# os.environ['JAX_ENABLE_X64'] = 'True'  # Use double precision
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'  # More detailed XLA info
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Force CPU execution

import jax
from jax import numpy as jnp
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





class CarryBoxEnv(PipelineEnv):
  """Three Crazyflies carry a box to 1 m and keep it stable."""

  def __init__(self,
               target_height: float = 1.0,
               height_reward_weight: float = 5.0,
               ctrl_cost_weight: float = 0.1,
               stability_cost_weight: float = 0.5,
               reset_noise_scale: float = 1e-2,
               **kwargs):
    # load your MJX model
    mj_model = mujoco.MjModel.from_xml_path("bitcraze_crazyflie_2/scene_mjx.xml")   # or full path
    mj_model.opt.solver       = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations   = 6
    mj_model.opt.ls_iterations= 6

    sys = mjcf.load_model(mj_model)

    # number of physics steps per control action
    physics_steps = 5
    kwargs['n_frames'] = kwargs.get('n_frames', physics_steps)
    kwargs['backend']  = 'mjx'

    super().__init__(sys, **kwargs)

    self.target_height          = target_height
    self.height_reward_weight   = height_reward_weight
    self.ctrl_cost_weight       = ctrl_cost_weight
    self.stability_cost_weight  = stability_cost_weight
    self.reset_noise_scale      = reset_noise_scale

    # cache body index for the payload box

  def reset(self, rng: jnp.ndarray) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    # add a little noise around default qpos0 / qvel0
    low, hi = -self.reset_noise_scale, self.reset_noise_scale
    qpos = self.sys.qpos0# + jax.random.uniform(rng1, (self.sys.nq,), low, hi)
    qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=-0.01, maxval=0.01)
    data = self.pipeline_init(qpos, qvel)

    # initial observation, zero reward/done/metrics
    obs    = self._get_obs(data, jnp.zeros(self.sys.nu))
    reward = jnp.array(0.0)
    done   = jnp.array(0.0)
    metrics = {
      'height'       : data.qpos[6],
      'height_error' : jnp.abs(data.qpos[6] - self.target_height),
      'reward_height' : 0.0,
      'reward_ctrl' : 0.0,
      'reward_stability' : 0.0
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jnp.ndarray) -> State:
    data0 = state.pipeline_state
    data1 = self.pipeline_step(data0, action)

    # 1) height reward: encourage box_z ≈ target_height
    box_z = data1.qpos[6]
    height_err = box_z - self.target_height
    reward_height = self.height_reward_weight * (1.0 - jnp.abs(height_err))

    # 2) control cost
    ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))

    # 3) stability cost: penalize box angular velocity (rough proxy)
    #    we can get body angular vel from data1.cvel (twist) for each body:
    ang_vel = data1.cvel[0, 3:]   # shape (3,)
    stab_cost = self.stability_cost_weight * jnp.sum(jnp.square(ang_vel))

   # reward = reward_height  - ctrl_cost - stab_cost
    reward = ctrl_cost
    done   = 0.0  # you can define a termination if box falls too low/high

    obs = self._get_obs(data1, action)
    # update metrics for logging
    state.metrics.update(
      height=box_z,
      height_error=jnp.abs(height_err),
      reward_height=reward_height,
      reward_ctrl=-ctrl_cost,
      reward_stability=-stab_cost,
    )

    return state.replace(pipeline_state=data1,
                         obs=obs,
                         reward=reward,
                         done=done)

  def _get_obs(self, data, action):
    # you can observe:
    #  - box xpos & orientation  (7 dims)
    #  - cf2_i qpos & qvel for each drone
    #  - cable stretch (if desired)
    #  - last action
    box_qpos = data.qpos[:7]            # [quat, xyz] of payload_box
    cf_qpos  = data.qpos[7:]            # the rest: 3×7 dims for each cf
    obs = jnp.concatenate([box_qpos,
                           cf_qpos,
                           data.qvel,
                           action])
    return obs



cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

# Option 1: Isometric view (recommended for general viewing)
cam.lookat = [0, 0, 0.1]    # Look at where the Crazyflie is (z=0.1)
cam.distance = 1.5          # Much closer than 10
cam.elevation = -30         # Look down at an angle
cam.azimuth = 45   

envs.register_environment('carry_box', CarryBoxEnv)

# instantiate the environment
env_name = 'carry_box'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state.pipeline_state]

# # grab a trajectory
# for i in range(10):
#   ctrl = -0.1 * jnp.ones(env.sys.nu)
#   state = jit_step(state, ctrl)
#   rollout.append(state.pipeline_state)

# media.write_video(path="gifs/train.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)




env_name = 'carry_box'
env    = envs.get_environment(env_name)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=10000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=50,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=24,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-5,
    entropy_cost=1e-3,
    num_envs=3072,
    batch_size=512,
    seed=0,
)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0
xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  print(num_steps, metrics)


print('training...')
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print('done training')

#@title Save Model
model_path = 'models/mjx_brax_policy'
model.save_params(model_path, params)

#@title Load Model and Define Inference Function
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 50
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

  if state.done:
    break

media.write_video(path="gifs/train.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)
