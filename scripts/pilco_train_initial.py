import os
import math
import sys
from glob import glob

import pandas as pd
import numpy as np
import gpflow
import pilco
import rospkg
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward

import dill
import pickle

rospack = rospkg.RosPack()

DATA_PATH = os.path.join(rospack.get_path('pilco_control'), 'data')
fnames = list(glob(f'{DATA_PATH}/*.csv'))

dats = [pd.read_csv(fname) for fname in fnames]

try:
    out_fname = sys.argv[1]
except:
    out_fname = 'pilco_model_30hz_experience.pkl' 

# dat['position_lower'] = dat['position_lower'] + math.pi
# dat['position_upper'] = dat['position_upper'] + math.pi

for dat in dats:
    for colname in ['position_lower', 'position_upper', 'velocity_lower', 'velocity_upper']:
        dat[colname + '_diff'] = np.append(dat[colname][1:].values - dat[colname][:-1].values, np.nan)

# print(dats)
# new function delta t mapping between prev and new state

target = np.array([3.14, 3.14, 0., 0.])
state_dim = 4
control_dim = 2
weights = np.diag([2.0, 1.0, 9.0, 8.0])

stride = dat.shape[0] // 270

X = pd.DataFrame(columns = ['position_lower', 'position_upper', 'velocity_lower', 'velocity_upper', 'effort_lower', 'effort_upper'])
Y = pd.DataFrame(columns=['position_lower_diff', 'position_upper_diff', 'velocity_lower_diff', 'velocity_upper_diff'])

points_per_run = 30

for dat in dats:
    startk = np.random.randint(dat.shape[0] - points_per_run)
    X = X.append(pd.DataFrame(dat[startk:startk+points_per_run], 
            columns=['position_lower', 'position_upper', 'velocity_lower', 
                        'velocity_upper', 'effort_lower', 'effort_upper']))
    Y = Y.append(pd.DataFrame(dat[startk:startk+points_per_run],
            columns=['position_lower_diff', 'position_upper_diff',
                        'velocity_lower_diff', 'velocity_upper_diff']))

#X = pd.DataFrame(dat[startk:startk+270], columns=['position_lower', 'position_upper', 'velocity_lower', 'velocity_upper', 'effort_lower', 'effort_upper']).values
#Y = pd.DataFrame(dat[startk:startk+270], columns=['position_lower_diff', 'position_upper_diff', 'velocity_lower_diff', 'velocity_upper_diff']).values

X = X.values
Y = Y.values

print('startk = ', startk)
print('X.shape = ', X.shape)
print('Y.shape =', Y.shape)

# X = X.T; Y = Y.T
T =  int(30 * 2)

m_init = np.reshape([0.0, 0.0, 0.0, 0.0], (1,4))
S_init = np.diag([0.01, 0.01, 0.01, 0.01])

controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
pilco_model = PILCO((X, Y), controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

pilco_model.optimize_models(maxiter=100)
pilco_model.optimize_policy(maxiter=20)


with open(out_fname, 'wb') as wf:
    frozen_model = gpflow.utilities.freeze(pilco_model)
    dill.dump(frozen_model, wf)
