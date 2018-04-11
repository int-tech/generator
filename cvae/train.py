from keras.datasets import mnist
import numpy as np
from cvae import model
import matplotlib.pyplot as plt


def data_preprocessing(data='mnist'):

    if data == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        n_classes = 10
        input_dim = 784

        return x_train, y_train, x_test, y_test, input_dim, n_classes
    else:
        raise NotImplementedError


def train(z_dim):
    x_train, y_train, x_test, y_test, input_dim, n_classes = data_preprocessing()

    epochs = 20
    batch_size = 100

    cvae, encoder, decoder = model.cvae(n_classes, input_dim, z_dim=z_dim)
    cvae.fit([x_train, y_train], x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([x_test, y_test], x_test))

    return cvae, encoder, decoder


if __name__ == '__main__':
    z_dim = 2
    cvae, encoder, generator = train(z_dim)

    epsilon_std = 0.2

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()