import sys
import math

import pandas as pd
import numpy as np
import gpflow
import pilco
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward

import dill
import pickle

try:
    dat = pd.read_csv(sys.argv[1])
except:
    dat = pd.read_csv("./pilco_train_data30hz.csv")

try:
    fname = sys.argv[2]
except:
    fname = 'pilco_model_30hz_experience.pkl' 


# dat['position_lower'] = dat['position_lower'] + math.pi
# dat['position_upper'] = dat['position_upper'] + math.pi

for colname in ['position_lower', 'position_upper', 'velocity_lower', 'velocity_upper']:
    dat[colname + '_diff'] = np.append(dat[colname][1:].values - dat[colname][:-1].values, np.nan)

print(dat)
# new function delta t mapping between prev and new state

target = np.array([3.14, 3.14, 0., 0.])
state_dim = 4
control_dim = 2
weights = np.diag([2.0, 1.0, 9.0, 8.0])

stride = dat.shape[0] // 270
startk = np.random.randint(dat.shape[0] - 270)

X = pd.DataFrame(dat[startk:startk+270], columns=['position_lower', 'position_upper', 'velocity_lower', 'velocity_upper', 'effort_lower', 'effort_upper']).values
Y = pd.DataFrame(dat[startk:startk+270], columns=['position_lower_diff', 'position_upper_diff', 'velocity_lower_diff', 'velocity_upper_diff']).values

print('startk = ', startk)
print('X = ', X)
print('Y =', Y)

# X = X.T; Y = Y.T
T =  int(30 * 3)

m_init = np.reshape([0.0, 0.0, 0.0, 0.0], (1,4))
S_init = np.diag([0.01, 0.01, 0.01, 0.01])

controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
pilco_model = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

pilco_model.optimize_models(maxiter=100)
pilco_model.optimize_policy(maxiter=20)


with open(fname, 'wb') as wf:
    frozen_model = gpflow.utilities.freeze(pilco_model)
    dill.dump(frozen_model, wf)
