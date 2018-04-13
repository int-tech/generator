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
        self.output_dim = input_dim
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

    def _build_cvae_mlp(self):
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
        input_layer = Concatenate(name='input_layer')([x, y])

        # building a encoder network
        enc_dense = Dense(self.intermediate_dim, activation='relu', name='enc_dense') # dense layer
        enc_drop = Dropout(rate=self.dropout_keep_prob, name='enc_drop') #dropout layer
        enc_mean = Dense(self.z_dim, activation='relu', name='z_mean')   # mean layer (dense layer)
        enc_log_sigma = Dense(self.z_dim, activation='relu', name='z_log_sigma')   # variance layer (dense layer)

        enc_hid = enc_drop(enc_dense(input_layer))
        self._z_mean = enc_mean(enc_hid)
        self._z_log_sigma = enc_log_sigma(enc_hid)

        sample_z = Lambda(self._sampling)([self._z_mean, self._z_log_sigma])

        # concatenating z and y using Concatenate Layer
        dec_merged = Concatenate(name='dec_merged')([sample_z, y])

        # building a decoder network
        dec_dense = Dense(self.intermediate_dim, activation='relu', name='dec_dense')  # dense layer
        dec_out = Dense(self.output_dim, activation='sigmoid', name='dec_out')  # output layer(dense layer) using sigmoid. you should use use_bias=False

        dec_hid = dec_dense(dec_merged)
        out = dec_out(dec_hid)

        cvae_model = Model([x, y], out)
        encoder_model = Model([x, y], self._z_mean)

        z = Input(shape=(self.z_dim,))
        gen_merged = Concatenate(name='gen_merged')([z, y])
        gen_hid = dec_dense(gen_merged)
        gen_out = dec_out(gen_hid)

        generator_model = Model([z, y], gen_out)

        return cvae_model, encoder_model, generator_model

    def _vae_loss(self, x, x_decoded_mean):
        '''
        loss function for VAE

        :param x: keras tensor object. target vector to be reconstructed.
        :param x_decoded_mean: keras tensor object. output of decoder.
        :return: loss
        '''
        ent_loss = self.input_dim * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self._z_log_sigma - K.square(self._z_mean) - K.exp(self._z_log_sigma), axis=-1)

        return ent_loss + kl_loss

    def get_simple_cvae(self):
        '''
        build a simple conditional VAE
        :return: tuple of models, (CVAE model, encoder model, decoder model)
        '''
        cvae_model, encoder_model, generator_model = self._build_cvae_mlp()
        cvae_model.compile(optimizer='rmsprop', loss=self._vae_loss)

        return cvae_model, encoder_model, generator_model
