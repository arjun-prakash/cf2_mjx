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





import jax.numpy as jnp
from brax import math

class CarryBoxEnv(PipelineEnv):
  def __init__(self,
               target_height: float = 1.0,
               # — new reward hyper-parameters —
               reward_distance_scale: float    = 1.6,
               reward_align_weight: float      = 0.5,
               reward_swing_weight: float      = 0.1,
               reward_effort_weight: float     = 0.1,
               reward_smoothness_weight: float = 0.05,
               **kwargs):
    # load MJX model & system as before
    mj_model = mujoco.MjModel.from_xml_path("bitcraze_crazyflie_2/scene_mjx.xml")
    sys      = mjcf.load_model(mj_model)
    super().__init__(sys, **kwargs)

    self.target_height           = target_height
    # store new reward weights
    self.reward_distance_scale   = reward_distance_scale
    self.reward_align_weight     = reward_align_weight
    self.reward_swing_weight     = reward_swing_weight
    self.reward_effort_weight    = reward_effort_weight
    self.reward_smoothness_weight= reward_smoothness_weight

  def reset(self, rng: jnp.ndarray) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    qpos = self.sys.qpos0
    qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=-0.01, maxval=0.01)
    data = self.pipeline_init(qpos, qvel)

    # zero-out last_action so smoothness has a baseline
    obs = self._get_obs(data, jnp.zeros(self.sys.nu))
     # **Initialize every metric key that step() will ever write:**
    metrics = {
      'reward_pose'      : jnp.array(0.0),
      'reward_align'     : jnp.array(0.0),
      'reward_swing'     : jnp.array(0.0),
      'reward_effort'    : jnp.array(0.0),
    }
    return State(data, obs, 0.0, 0.0, metrics)

  def step(self, state: State, action: jnp.ndarray) -> State:
    data0 = state.pipeline_state
    data1 = self.pipeline_step(data0, action)

    # 1) pose error → r_pose = exp(-scale · ‖pos - target‖)
    box_quat = data1.qpos[:4]
    box_pos  = data1.qpos[4:7]
    target   = jnp.array([0., 0., self.target_height])
    pos_err  = jnp.linalg.norm(box_pos - target)
    r_pose   = jnp.exp(- self.reward_distance_scale * pos_err)

    # 2) orientation alignment → r_align = ((up·world_up + 1)/2)^2
    up_vec   = math.rotate(jnp.array([0., 0., 1.]), box_quat)
    world_up = jnp.array([0., 0., 1.])
    r_align  = ((jnp.dot(up_vec, world_up) + 1.) / 2.)**2

    # 3) swing/spin penalty → r_swing = 1/(1 + ω^2)
    ang_vel  = data1.cvel[0, 3:]                  # box’s angular velocity
    spin     = jnp.linalg.norm(ang_vel)
    r_swing  = 1. / (1. + spin**2)

    # 4) effort cost
    r_effort = 0.0# self.reward_effort_weight * jnp.exp(-jnp.sum(jnp.square(action)))

    # 5) action smoothness
    # delta    = action - prev_act
    # r_smooth = self.reward_smoothness_weight * jnp.sum(jnp.square(delta))

    # combine them
    reward = (
      r_pose
      + r_pose * (r_align + r_swing)
      + r_effort
    )

    # termination logic (unchanged)
    done = jnp.where((box_pos[2] <= 0.0) | (box_pos[2] >= 2.0), 1.0, 0.0)

    # build new observation
    obs = self._get_obs(data1, action)

    # update metrics for logging / smoothness
    state.metrics.update({
      'reward_pose'      : r_pose,
      'reward_align'     : r_align,
      'reward_swing'     : r_swing,
      'reward_effort'    : r_effort,
    })

    return state.replace(
      pipeline_state = data1,
      obs            = obs,
      reward         = reward,
      done           = done,
    )


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

print("# actuators (sys.nu):", env.sys.nu)   # should print 12
print("action_shape:", env.action_size)  

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
    num_timesteps=50000000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=500,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=24,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=3072,
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
n_steps = 500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)



print(len(rollout))


media.write_video(path="gifs/train.mp4", images=env.render(rollout, camera=cam), fps=1.0 / env.dt)
print('done training')
plt.plot(xdata, ydata, label='Episode Reward')
plt.xlabel('Number of Steps')
plt.ylabel('Episode Reward')
plt.title('Episode Reward vs. Number of Steps')
plt.legend()
plt.savefig('plots/train.png')
