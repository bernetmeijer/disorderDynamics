import pickle
import numpy as np

# just simple kvecs
k_vecs = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])

with open('kvecs_N4_GX.pkl', 'wb') as handle:
    pickle.dump(k_vecs, handle)
