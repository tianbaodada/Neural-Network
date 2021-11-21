import time # 引入time
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from numpy.lib.npyio import load
from dense import Dense
from convolutional import Convolutional
from maxpool2d import Maxpool2d
from reshape import Reshape
from activations import Sigmoid, Tanh, Relu, Softmax
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train

def preprocess_data(x, y, limit):
    # zero_index = np.where(y == 0)[0][:limit]
    # one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack(list(np.where(y == x)[0][:limit] for x in range(10)))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_valid, y_valid = x_train[900:], y_train[900:]
x_train, y_train = x_train[:900], y_train[:900]
x_test, y_test = preprocess_data(x_test, y_test, 100)

# nowTime = int(time.time()) # 取得現在時間
# struct_time = time.localtime(nowTime) # 轉換成時間元組
# timeString = time.strftime("%m%d%I%M%S", struct_time) # 將時間元組轉換成想要的字串
# print(timeString)

l2 = 1e-1
learning_rate = 3e-2

# neural network
network = [
    Convolutional((1, 28, 28), 5, 20, l2=l2),
    Maxpool2d(0, 2, 2),
    Tanh(),
    Reshape((20, 12, 12), (20 * 12 * 12, 1)),
    Dense(20 * 12 * 12, 100, l2=l2),
    Tanh(),
    Dense(100, 10, l2=l2),
    Softmax()
]

net_id = '1121071336'

# train
training_accuracy, valid_accuracy, test_accuracy, training_loss = \
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    (x_train, y_train),
    valid_data=(x_valid, y_valid),
    test_data=(x_test, y_test),
    # net_id=net_id,
    epochs=70,
    learning_rate=learning_rate,
    verbose=True
)

myDpi = 192
plt.figure(figsize=(2560/myDpi, 1680/myDpi), dpi=myDpi)
figure, axs = plt.subplots(3, 2)

axs[0,0].set_title('accuracy rate')
axs[0,0].set_xlabel('Iteration')
axs[0,0].set_ylabel('Accuracy rate')
plots00 = axs[0,0].plot(list(range(len(training_accuracy))), list(zip(training_accuracy, valid_accuracy)), label='')
axs[0,0].legend(plots00, ('training', 'validation', 'test'), loc='best', framealpha=0.25, prop={'size': 'small', 'family': 'monospace'})

axs[0,1].set_title('Learning Curve')
axs[0,1].set_xlabel('Iteration')
axs[0,1].set_ylabel('Loss')
plots01 = axs[0,1].plot(list(range(len(training_loss))), training_loss, label='')

axs[1,0].set_title('histogram of conv1')
axs[1,0].set_xlabel('Value')
axs[1,0].set_ylabel('Number')
axs[1,0].hist(network[0].weights.reshape(-1),bins=100)

axs[1,1].set_title('histogram of dense1')
axs[1,1].set_xlabel('Value')
axs[1,1].set_ylabel('Number')
axs[1,1].hist(network[4].weights.reshape(-1),bins=100)

axs[2,0].set_title('histogram of output')
axs[2,0].set_xlabel('Value')
axs[2,0].set_ylabel('Number')
axs[2,0].hist(network[6].weights.reshape(-1),bins=100)

msg = f'net_id={net_id}\nl2={l2}\nlearning_rate={learning_rate}'
axs[2,1].text(0.1, 0.5, msg, ha='left', va='center', size=12, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
axs[2,1].axis('off')

plt.tight_layout()
plt.savefig('result.png' , dpi=myDpi)
# plt.savefig('drive/MyDrive/DocBase/nycuiim/深度學習/HW/HW2/test.png')
# plt.show()
