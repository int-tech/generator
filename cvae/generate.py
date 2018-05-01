import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_model(dir_name='results/model', model_name='simpleCVAE'):
    '''
    loading model from weight file and model file

    :param dir_name: str, directory path of weight and model file
    :param model_name: str, base file name of model file
    :return: encoder and decoder
    '''
    json_string = open(os.path.join(dir_name, '{}_enc.json'.format(model_name))).read()
    encoder = model_from_json(json_string)
    encoder.load_weights(os.path.join('{}/{}_enc.h5'.format(dir_name, model_name)))

    json_string = open(os.path.join(dir_name, '{}_dec.json'.format(model_name))).read()
    generator = model_from_json(json_string)
    generator.load_weights(os.path.join('{}/{}_dec.h5'.format(dir_name, model_name)))

    return encoder, generator


def get_picture(filepath, target_shape):
    '''
    loading image file for generator

    :param filepath: str, file path of image file
    :param target_shape: tuple, shape for input image
    :return: ndarray, ndarray of input image
    '''
    temp_img = load_img(filepath, grayscale=True, target_size=target_shape)
    img_array = img_to_array(temp_img)
    img_array /= 255.
    img_array = img_array.reshape(np.prod(img_array.shape[:2]))

    return img_array

def generate_number(filepath, n_classes=10, shape = (28,28)):
    '''
    generating number from 0 to 9 using generator trained CVAE.

    :param filepath: str, file path of input file
    :param n_classes: int, the number of classes
    :return: list of img, generated images
    '''

    encoder, generator = load_model()
    x = get_picture(filepath, shape)
    x = np.where(x > 0.6, 0, 1.0-x)

    enc_list = []
    for i in range(n_classes):
        y = to_categorical(i, n_classes)
        z_encoded = encoder.predict([np.array([x]), np.array([y])])
        enc_list.append(z_encoded)

    z = np.average(enc_list, axis=0)

    gen_list = []
    for i in range(n_classes):
        y = to_categorical(i, n_classes)
        x_decoded = generator.predict([z, np.array([y])])
        x_decoded = x_decoded.reshape(28, 28, 1)
        gen_img = array_to_img(x_decoded, scale=True)
        gen_list.append(gen_img)

    return gen_list


if __name__ == '__main__':
    dir_path = 'test_data'
    label = 2

    gen_list = generate_number(filepath='{}/{}.png'.format(dir_path, label))

    for i, gen_img in enumerate(gen_list):
        plt.imsave('gen_{}.png'.format(i), gen_img)
        plt.show()
