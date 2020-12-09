import dill
import numpy as np

with open('./pilco_model2.pkl', 'rb') as f:
    pilco_model = dill.load(f)

print(pilco_model.compute_reward())

