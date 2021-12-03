import numpy as np
import pickle

filename = './output_10_class//score_train_2021_12_2_10_27.pkl'
with open(filename, 'rb') as f:
    x = np.array(pickle.load(f))

print(x.max())
print(x.argmax())