import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from utils import rollout, policy

np.random.seed(0)

class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01*high) # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

env = myPendulum()

# Settings, are explained in the rest of the notebook
SUBS=3 # subsampling rate
T = 30 # number of timesteps (for planning, training and testing here)
J = 3 # rollouts before optimisation starts

max_action=2.0 # used by the controller, but really defined by the environment

# Reward function parameters
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])

# Environment defined
m_init = np.reshape([-1.0, 0.0, 0.0], (1,3))
S_init = np.diag([0.01, 0.01, 0.01])



X,Y, _, _ = rollout(env, None, timesteps=T, verbose=False, random=True, SUBS=SUBS, render=False)
for i in range(1,J):
    X_, Y_, _, _ = rollout(env, None, timesteps=T, verbose=False, random=True, SUBS=SUBS, render=False)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10, max_action=max_action)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
pilco = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

pilco.optimize_models(maxiter=100)
pilco.optimize_policy(maxiter=20)

X_new, Y_new, _, _ = rollout(env, pilco, timesteps=T, SUBS=SUBS, render=False)
