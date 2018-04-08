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
        self.intermediate_dim=intermediate_dim
        self.dropout_keep_prob=dropout_keep_prob

    def _sampling(self, args):
        """
        Sampling function for latent variable
        :param args: tuple(mean, log sigma), parameters of Gaussian distribution for latent variable
        :return: sampling value generated from Gaussian distribution
        """
        mean, log_sigma = args

        #TODO: generating random variable epsilon ~ N(0, I)
        epsilon = 0

        #TODO: return random variable z. calculating z ~ N(mean, exp(log_sigma)) using z = mean + exp(log_sigma) * epsilon
        return None

    def build_cvae_mlp(self, kl_weight=1.0):
        """
        build function of CVAE using MLP model.
        :param kl_weight: float, weight for KL divergence(Default is 1.0)
        :return: model of CVAE
        """
        input_dim = self.input_dim

        #TODO: defining x and y using Input Layer
        x = None
        y = None

        # TODO: concatenating x and y using Concatenate Layer
        concat = None

        # TODO: building a encoder network
        enc_dense = None #dense layer
        enc_drop = None #dropout layer
        z_mean = None #mean layer (dense layer)
        z_log_sigma = None #variance layer(dense layer)

        z = Lambda(self._sampling)([z_mean, z_log_sigma])

        # TODO: concatenating z and y using Concatenate Layer
        dec_merged = None

        # TODO: building a decoder network
        dec_dense = None # dense layer
        dec_out = None # output layer(dense layer) using sigmoid. you should use use_bias=False

        cvae_model = Model([x, y], dec_out)
        encoder_model = Model([x, y], z_mean)

        # TODO: building generator (Taku will implement this later)
        generator_model = None

        return cvae_model, encoder_model, generator_model