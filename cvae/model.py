from keras.models import Model
from keras.layers import Dense, Concatenate, Dropout, Input, Lambda, Layer
from keras import backend as K
from keras.metrics import binary_crossentropy


class cvae(object):

    def __init__(self, n_classes, input_dim, z_dim=100, intermediate_dim=300, dropout_keep_prob=1.0):
        """
        Conditional Variational AutoEncoder(CVAE) Class

        :param n_classes: int, the number of classes
        :param input_dim: int, dimension of input (1d array)
        :param z_dim: int, dimension of latent variable
        :param intermediate_dim: int, dimension of hidden layer in encoder and decoder
        :param dropout_keep_prob: float, probability of using a node (Default is 1.0, without Dropout)
        """
        self.n_classes = n_classes
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_keep_prob = dropout_keep_prob

    def _sampling(self, args):
        """
        Sampling function for latent variable
        :param args: tuple(mean, log sigma), parameters of Gaussian distribution for latent variable
        :return: sampling value generated from Gaussian distribution
        """
        # gaussian parameters
        mean, log_sigma = args

        # generating random variable epsilon ~ N(0, I)
        epsilon = K.random_normal(shape=(K.shape(mean)[0], self.z_dim), mean=0.,stddev=1.0)

        # return random variable z. calculating z ~ N(mean, exp(log_sigma))
        return mean + K.exp(log_sigma) * epsilon

    def build_cvae_mlp(self, kl_weight=1.0):
        """
        build function of CVAE using MLP model.
        :param kl_weight: float, weight for KL divergence(Default is 1.0)
        :return: model of CVAE
        """
        input_dim = self.input_dim

        # defining x and y using Input Layer
        x = Input(shape=(input_dim,))
        y = Input(shape=(self.n_classes,))

        # concatenating x and y using Concatenate Layer
        input_layer = Concatenate([x, y], name='input_layer')

        # building a encoder network
        enc_dense = Dense(self.intermediate_dim, activation='relu', name='enc_dense')(input_layer)  # dense layer
        # enc_drop = None #dropout layer
        z_mean = Dense(self.z_dim, activation='relu', name='z_mean')(enc_dense)   # mean layer (dense layer)
        z_log_sigma = Dense(self.z_dim, activation='relu', name='z_log_sigma')(enc_dense)   # variance layer (dense layer)

        z = Lambda(self._sampling)([z_mean, z_log_sigma])

        # concatenating z and y using Concatenate Layer
        dec_merged = Concatenate([z, y], name='dec_merged')

        # building a decoder network
        dec_dense = Dense(self.intermediate_dim, activation='relu', name='dec_dense')(dec_merged)  # dense layer
        dec_out_dim = input_dim
        dec_out = Dense(dec_out_dim, activation='sigmoid', name='dec_out')(dec_dense)  # output layer(dense layer) using sigmoid. you should use use_bias=False

        cvae_model = Model([x, y], dec_out)
        encoder_model = Model([x, y], z_mean)

        # TODO: building generator (Taku will implement this later)
        generator_model = None

        return cvae_model, encoder_model, generator_model
