from typing import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

data = pd.read_csv('resource/ionosphere_data.csv', header=None)
data = data.sample(frac=1).reset_index(drop=True)
targets = data.apply(lambda x: 1 if x.iloc[-1] == 'g' else 0, axis=1)
selected_data = data.iloc[:,:-1]

subset_size = len(selected_data) // 10
X_train = selected_data[:subset_size * 8].to_numpy().reshape(-1,34,1)
Y_train = targets[:subset_size * 8].to_numpy().reshape(-1,1,1)
X_valid = selected_data[subset_size * 8: subset_size * 9].to_numpy().reshape(-1,34,1)
Y_valid = targets[subset_size * 8: subset_size * 9].to_numpy().reshape(-1,1,1)
X_test = selected_data[subset_size * 9:].to_numpy().reshape(-1,34,1)
Y_test = targets[subset_size * 9:].to_numpy().reshape(-1,1,1)

network = [
    Dense(34, 16),
    Sigmoid(),
    Dense(16, 16),
    Sigmoid(),
    Dense(16, 16),
    Sigmoid(),
    Dense(16, 3),
    Sigmoid(),
    Dense(3, 1),
    Sigmoid(),
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    (X_train, Y_train),
    valid_data=(X_valid, Y_valid),
    epochs=300,
    learning_rate=10 ** -2,
    mini_batch_size=10,
    verbose=True
)

pred_res = [(1 if predict(network, x)[0,0] >= 0.5 else 0, y[0,0]) for x, y in zip(X_test, Y_test)]
print(f"test accuracy: {sum([int(x==y) for x,y in pred_res])/len(pred_res)}")
