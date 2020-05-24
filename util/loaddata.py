import numpy as np

# Load Numpy zipped arrays (as an alternative to HDF5).
def load_npz(f):
    return np.float32(np.load(f)['arr_0'])

