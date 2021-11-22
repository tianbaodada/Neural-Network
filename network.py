import numpy as np
import matplotlib.pyplot as plt
import datetime

def save_network(network, id):
    for i in range(len(network)):
        network[i].save(f'{id}-{i}')

def load_network(network, id):
    for i in range(len(network)):
        network[i].load(f'{id}-{i}')

def update_network(network, learning_rate):
    for layer in network:
        layer.update(learning_rate)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network,
          loss,
          loss_prime,
          training_data,
          valid_data=None,
          test_data=None,
          net_id='default',
          epochs=1000,
          learning_rate=0.01,
          mini_batch_size=10,
          verbose=False):
    
    x_train, y_train = training_data
    n = len(x_train)
    all_indices = range(n)

    training_loss = []
    training_accuracy = []
    valid_accuracy = []
    test_accuracy = []

    best_valid = 0
    print(f"{str(datetime.datetime.now())} - training start")
    for e in range(epochs):
        error = 0
        all_indices = np.random.permutation(all_indices)
        x_train, y_train = x_train[all_indices], y_train[all_indices]
        
        x_batches = [x_train[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        y_batches = [y_train[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        
        for x_batch, y_batch in zip(x_batches, y_batches):
            for x, y in zip(x_batch, y_batch):
                # forward
                output = predict(network, x)

                # error
                # error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad)
        
            update_network(network, learning_rate)
        
        for x,y in zip(x_train, y_train):
            output = predict(network, x)
            error += loss(y, output)
        
        error /= len(x_train)
        if verbose: print(f"{str(datetime.datetime.now())} - {e + 1}/{epochs}, error={error}")
        training_loss.append((error))

        pred_res = [int(np.argmax(predict(network, x)) == np.argmax(y)) for x,y in zip(x_train, y_train)]
        acc = sum(pred_res)/len(pred_res)
        training_accuracy.append(acc)
        if verbose: print(f"training accuracy: {acc}")

        if valid_data:
            x_valid, y_valid = valid_data
            pred_res = [int(np.argmax(predict(network, x)) == np.argmax(y)) for x,y in zip(x_valid, y_valid)]
            acc = sum(pred_res)/len(pred_res)
            valid_accuracy.append(acc)
            if acc > best_valid:
                best_valid = acc
                save_network(network, net_id)
            if verbose: print(f"validation accuracy: {acc}")

        if test_data:
            # load_network(network, net_id)
            x_test, y_test = test_data
            pred_res = [int(np.argmax(predict(network, x)) == np.argmax(y)) for x,y in zip(x_test, y_test)]
            acc = sum(pred_res)/len(pred_res)
            test_accuracy.append(acc)
            if verbose: print(f"test accuracy: {acc}")

    return (training_accuracy, valid_accuracy, test_accuracy, training_loss)

def plot_output(network, x):
    output = x
    for layer in network:
        output = layer.forward(output)
        if len(output.shape) == 3:
            output_plot = output
            
            output_plot = output_plot - output_plot.mean()
            output_plot /= output_plot.std()
            output_plot *= 64
            output_plot += 128
            output_plot = np.clip(output_plot, 0, 255).astype('uint8')
            
            num_row = 2
            num_col = 5
            fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
            for i in range(len(output_plot)):
                ax = axes[i // num_col, i % num_col]
                ax.imshow(output_plot[i], cmap='gray')
                ax.set_title(f'feature {i + 1}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()