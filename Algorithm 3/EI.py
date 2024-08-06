import numpy as np
import matplotlib.pyplot as plt
from Model_BO import model
import pickle

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from sklearn import gaussian_process as gp
from sklearn.preprocessing import StandardScaler

#n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
#init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

bo = BayesianOptimization(
    f=model,
    pbounds={"x1": (0.1, 1), "x2": (0.1, 1), "x3": (0.1, 2), "x4": (0.1, 2)},
    verbose=2,
    random_state=987234,
)
bo._gp.kernel = gp.kernels.RBF()
acquisition_function = UtilityFunction(kind="ei", xi=1e-1) #the points are more spread out within the range
bo.maximize(n_iter=6, acquisition_function=acquisition_function, init_points=10)

x_obs = np.array([[res["params"]["x1"], res["params"]["x2"], res["params"]["x3"],  res["params"]["x4"]] for res in bo.res])
y_obs = np.array([res["target"] for res in bo.res])

#save the observed points
with open('Data/X_EI_men.pickle', 'wb') as handle:
    pickle.dump(x_obs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Data/Y_EI_men.pickle', 'wb') as handle:
    pickle.dump(y_obs, handle, protocol=pickle.HIGHEST_PROTOCOL)