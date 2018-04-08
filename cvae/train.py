
def vae_loss(self, x, x_decoded_mean):
    """
    defining the loss function for VAE
    :param x:
    :param x_decoded_mean:
    :return:
    """
    xent_loss = input_dim * binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return K.mean(xent_loss + kl_weight * kl_loss)

def call(self, inputs, **kwargs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    loss = self.vae_loss(x, x_decoded_mean)
    self.add_loss(loss, inputs=inputs)
    return x