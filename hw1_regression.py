import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

data = pd.read_csv('resource/energy_efficiency_data.csv')
data = data.sample(frac=1).reset_index(drop=True)
# print(data)

CatIndex = ['Orientation', 'Glazing Area Distribution']
selected_data = pd.get_dummies(data, columns=CatIndex)
# print(selected_data)

subset_size = len(selected_data) // 4
training_data = selected_data[:subset_size * 3]
test_data = selected_data[subset_size * 3:]
targets = ['Heating Load', 'Cooling Load']

X_train = training_data.drop(targets,axis=1).to_numpy().reshape(-1,16,1)
Y_train = training_data['Heating Load'].to_numpy().reshape(-1,1,1)
X_test = test_data.drop(targets,axis=1).to_numpy().reshape(-1,16,1)
Y_test = test_data['Heating Load'].to_numpy().reshape(-1,1,1)

network = [
    Dense(16, 16),
    Dense(16, 1),
]

learning_curve_points = train(
                            network,
                            mse,
                            mse_prime,
                            (X_train, Y_train),
                            valid_data=(X_test, Y_test),
                            epochs=300,
                            learning_rate=10 ** -9,
                            mini_batch_size=100
                        )

# plot
training_points = [(predict(network, x)[0,0], y[0,0]) for x, y in zip(X_train, Y_train)]
test_points = [(predict(network, x)[0,0], y[0,0]) for x, y in zip(X_test, Y_test)]
# for x, y in zip(X_test, Y_test):
#     output = predict(network, x)
#     # print(f"pred: {(output)}, true: {(y)}")
#     points.append((predict(network, x)[0,0], y[0,0]))

figure, axs = plt.subplots(3)
plots0 = axs[0].plot(list(range(len(learning_curve_points))), learning_curve_points, label='')
plots1 = axs[1].plot(list(range(len(training_points))), training_points, label='')
axs[1].legend(plots1, ('predict', 'actual'), loc='best', framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})
plots2 = axs[2].plot(list(range(len(test_points))), test_points, label='')
axs[2].legend(plots2, ('predict', 'actual'), loc='best', framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})

plt.show()