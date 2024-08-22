import numpy as np

data = np.load('./hidden_states_20240714_053015.npz')
for key in data.files:
    print(data[key])