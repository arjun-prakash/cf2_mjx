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





class SimpleEnv(PipelineEnv):
  """Crazyflie 2.0 to 1 m and keep it stable."""

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
    mj_model.opt.iterations   = 10
    mj_model.opt.ls_iterations= 10

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
    self.target_pos = jnp.array([0.0, 0.0, 1.0])

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
      'x'       : data.qpos[0],
      'y'       : data.qpos[1],
      'z'       : data.qpos[2],
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jnp.ndarray) -> State:
    data0 = state.pipeline_state
    data1 = self.pipeline_step(data0, action)

    # 1) height reward: encourage box_z ≈ target_height
    xyz = data1.qpos[0:3]
    xyz_err = xyz - self.target_pos
    reward_xyz = -jnp.sum(jnp.square(xyz_err)) + 1e-8

    # 2) control cost
    ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))

    # 3) stability cost: penalize box angular velocity (rough proxy)
    #    we can get body angular vel from data1.cvel (twist) for each body:

    obs = self._get_obs(data1, action)
    # update metrics for logging
    state.metrics.update(
      x=xyz[0],
      y=xyz[1],
      z=xyz[2],
    )

    reward = reward_xyz 
    #done = jnp.where((height <= 0.0) | (height >= 2.0), 1.0, 0.0)
    # --- done (termination) signal ---------------------------------------------
    # contact with the floor?
    hard_contact = data1.ncon > 0           # 1 if any contact pair is active

    # body orientation: roll / pitch must stay within ±45 deg
    root_quat = data1.xquat[1]              # body 1 is the quad itself (world=0)
    roll, pitch, _ = math.quat_to_euler(root_quat)
    flipped = (jnp.abs(roll) > jnp.pi/4) | (jnp.abs(pitch) > jnp.pi/4)

    #done = jnp.where(flipped, 1.0, 0.0)
    done = jnp.array(0.0)

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
    obs = jnp.concatenate([data.qpos,
                           data.qvel,
                           action])
    return obs



cam = mujoco.MjvCamera()

mujoco.mjv_defaultCamera(cam)

# Option 1: Isometric view (recommended for general viewing)
cam.lookat = [0, 0, 0.1]    # Look at where the Crazyflie is (z=0.1)
cam.distance = 3          # Much closer than 10
cam.elevation = -30         # Look down at an angle
cam.azimuth = 45   

envs.register_environment('simple', SimpleEnv)

# instantiate the environment
env_name = 'simple'
env = envs.get_environment(env_name)

print("# actuators (sys.nu):", env.sys.nu)   # should print 12
print("action_shape:", env.action_size)  
print("position:", env.sys.qpos0)  

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




env_name = 'simple'
env    = envs.get_environment(env_name)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=5000000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=200,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=24,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=512,
    batch_size=512,
    seed=0,
)

y_data = []
ydataerr = []
times = [datetime.now()]

xdata, ydata, ydataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  print(num_steps, metrics['eval/episode_reward'])
  print('position:', metrics['eval/episode_x'], metrics['eval/episode_y'], metrics['eval/episode_z'])
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])
  times.append(datetime.now())
 




print('training...')
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)


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

# initialize the stateS
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 200
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)



print(len(rollout))


media.write_video(path="gifs/simple_train.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)
print('done training')
plt.plot(xdata, ydata, label='Episode Reward')
plt.xlabel('Number of Steps')
plt.ylabel('Episode Reward')
plt.title('Episode Reward vs. Number of Steps')
plt.legend()
plt.savefig('plots/train.png')
