import dill
import numpy as np

with open("./pilco_initial_model.pkl", "rb") as f:
    pilco_model = dill.load(f)

print(pilco_model)

print(pilco_model.compute_action(np.array([0., 1., 0., 1.])))
