import numpy as np
import util
import validation
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def crop_four_corners_image(img_input, corner_size_ratio=1.0):
    """
    crop four corners image for black_white_inverter()
    use four corners image to judge if input image should be inverted or not

        :param  img_input         : ndarray, 1ch image (binary image))
        :param  corner_size_ratio : 0 - 1.0, corner size ratio which is used to judge if image is inverted
        :return img_corners       : ndarray, four corner images
    """

    # limit corner_size_ratio between 0 and 1
    corner_size_ratio = util.limit_var_range(corner_size_ratio, 0.0, 1.0)

    # get size of input image
    h = img_input.shape[0]
    w = img_input.shape[1]

    # calcurate corner size
    h_corner_size = int(h * corner_size_ratio)
    w_corner_size = int(w * corner_size_ratio)

    # if corner area size is zero, replace the size to one
    h_corner_size = util.replace_num(h_corner_size, 0, 1)
    w_corner_size = util.replace_num(w_corner_size, 0, 1)

    # get 4 corner images
    img_corners = np.empty([4, h_corner_size, w_corner_size])         # initialize
    img_corners[0] = img_input[0:h_corner_size, 0:w_corner_size]      # upper left
    img_corners[1] = img_input[0:h_corner_size, w-w_corner_size:w]    # upper right
    img_corners[2] = img_input[h-h_corner_size:h, 0:w_corner_size]    # lower left
    img_corners[3] = img_input[h-h_corner_size:h, w-w_corner_size:w]  # lower right

    return img_corners


def black_white_inverter(img_input, corner_size_ratio=0.2, th_pixel_value=128):
    """
    invert black and white area so that number area is white area
    if 4 corner's areas have white area more than black one, invert black and white
    we want input image that number area is white to learn with Neural Network

        :param  img_input                 : ndarray, 1ch image (binary image))
        :param  corner_size_ratio         : 0 - 1.0, corner size ratio which is used to judge if image is inverted
        :return img_dst                   : ndarray, 1ch image
        :return is_inverted : boolean, True: inverted, False: not inverted
    """

    # exception handling : corner_size_ratio
    validation.validate_corner_size_ratio_range(corner_size_ratio)

    # get 4 corner images
    img_corners = crop_four_corners_image(img_input, corner_size_ratio)

    # if white area is more than black one, invert black and white
    mean_four_corner_pixel_value = np.mean(img_corners)
    if (mean_four_corner_pixel_value < th_pixel_value):
        # not inversion
        img_dst = img_input
        is_inverted = False
    else:
        # inversion
        img_dst = cv2.bitwise_not(img_input)
        is_inverted = True

    return img_dst, is_inverted


def resize_keeping_aspect_ratio(img_input, resized_size, OPT='LONG'):
    """
    resize image keeping aspect ratio

        :param  img_input    : ndarray, 1ch or 3ch image
        :param  resized_size : int, size that we want to resize
        :param  OPT          : str, 'LONG' or 'SHORT', fit to longer or shorter length of image
        :return img_dst      : ndarray, resized image
    """

    # exception handling : OPT
    validation.validate_option_resize_keeping_aspect_ratio(OPT)

    # get size of input image
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # exception handling : resized_size
    ratio_min = min(width_input, height_input) / max(width_input, height_input)
    validation.validate_resized_size(resized_size, ratio_min, OPT)

    # resize image
    if (height_input > width_input):
        if (OPT == 'LONG'):
            ratio = width_input / height_input
            img_dst = cv2.resize(img_input, (int(resized_size*ratio), resized_size))
        if (OPT == 'SHORT'):
            ratio = height_input / width_input
            img_dst = cv2.resize(img_input, (resized_size, int(resized_size*ratio)))
    else:
        if (OPT == 'LONG'):
            ratio = height_input / width_input
            img_dst = cv2.resize(img_input, (resized_size, int(resized_size*ratio)))
        if (OPT == 'SHORT'):
            ratio = width_input / height_input
            img_dst = cv2.resize(img_input, (int(resized_size*ratio), resized_size))

    return img_dst


def make_square_img(img_input):
    """
    make square image (1:1) from input image
    prepare background image which is black image
    and paste input image on background image

    HACK: 
    I use both OpenCV and PILLOW in this function to utilize paste function of PILLOW.
    I basically develop this function with OpenCV, but cannot find paste function of OpenCV.
    This is why I use both image processing library.

        :param  img_input  : ndarray, front image, 1ch or 3ch image
        :return img_square : ndarray, square image
    """

    # get image size
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # background image (paste input image on this backgraound image)
    back_size = max(width_input, height_input)
    img_back = np.zeros([back_size, back_size])

    # convert ndarray (opencv) to PIL (pillow) (to use paste function of PIL)
    img_back_pil = Image.fromarray(np.uint8(img_back))
    img_input_pil = Image.fromarray(np.uint8(img_input))

    # paste input image in approximately center position
    # pasting position depends on height and width length
    if (height_input > width_input):
        paste_position = int((height_input - width_input)/2)
        img_back_pil.paste(img_input_pil, (paste_position, 0))
    else:
        paste_position = int((width_input - height_input)/2)
        img_back_pil.paste(img_input_pil, (0, paste_position))

    # convert PIL to ndarray
    img_square = np.asarray(img_back_pil)

    return img_square


def limit_img_size(img_input, max_img_size):
    """
    adjust input image size if the size exceeds upper limit

        :param img_input    : ndarray, input image
        :param max_img_size : int, upper limit of an image size
        :return img_out     : ndarray, output image
    """

    # get input image size
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # check input image size and resize if the size exceed upper limit
    if (max(height_input, width_input) > max_img_size):
        img_out = resize_keeping_aspect_ratio(img_input, max_img_size, 'LONG')
    else:
        img_out = img_input
    return img_out


def denoise_with_opening(img_input, opening_ratio):
    """
    denoise after binarising with opening

        :param img_input     : ndarray, input image
        :param opening_ratio : 0 - 1.0, denoising size
        :return img_out      : ndarray, output image
    """

    # get input image size
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # opening processing
    if (opening_ratio > 1.0):
        opening_ratio = 1.0
    if (opening_ratio < 0.0):
        opening_ratio = 0.0
    opening_size = int(((width_input + height_input) / 2) * opening_ratio)
    if (opening_size > 0):
        kernel = np.ones((opening_size, opening_size), np.uint8)
        img_out = cv2.morphologyEx(img_input, cv2.MORPH_OPEN, kernel)
    return img_out


def process_img_for_input(img_input, output_size=0, opening_ratio=0.01, corner_size_ratio=0.2, max_img_size=1080):
    """
    make input image for neural network so that a number area is white and the other is black
    by processing image that user inputs

    XXX:
    This function might not work if an input image is not 8 bit.
    A Countermesure might be needed.

    TODO:
    OPT "GRAY" is implemented soon.
    Pixel shading is probably related to handwriting.

        :param img_input       : ndarray, "uint8" image (rgb or gray)
        :param output_size     : int, resized size (if 0 is set, output image size is equal to input)
        :param opening_ratio   : 0 - 1.0, opening size
        :param max_img_size    : int, limit an input image size with this variable
        :return img_out_square : ndarray, image which is input into neural network
    """
 
    # get input image size
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # if an input image size is too large, resize into 1080 size
    img_src = limit_img_size(img_input, max_img_size)

    # if an input is a rgb image, convert rgb image to gray image
    if (len(img_src.shape) == 3):
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # if an input is a gray image
    else:
        img_gray = img_src

    # convert a gray image to a binary image using Otsu'size method
    ret, img_bin = cv2.threshold(img_gray,
                                 0,
                                 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # invert black and white pixel only if a number area is not white pixel
    # and this image is used as mask image
    img_mask, _ = black_white_inverter(img_bin, corner_size_ratio, ret)

    # opening for eliminating small noise (erosion -> dilation)
    img_mask = denoise_with_opening(img_mask, opening_ratio)

    # resize an output image
    if (output_size >= 1):
        img_out_bin = resize_keeping_aspect_ratio(img_mask, output_size, 'LONG')
    else:
        img_out_bin = img_mask
    # translate into square image
    img_out_square = make_square_img(img_out_bin)
    return img_out_square


if __name__ == '__main__':
    # set parameters
    denoising_ratio = 0.01          # denoising ratio based on image size
    corner_size_ratio = 0.2         # confirmed corners size to judge if color is invered or not
    output_size = 0                 # equal to input image size in the case of "0"
    max_limitation_img_size = 1080  # limit an input image size with this size

    # define test images (Attension: use 8bit image)
    img1 = "../../test_data/numimages/5.png"
    img2 = "../../test_data/numimages/5_.png"
    img3 = "../../test_data/numimages/6_.png"
    img4 = "../../test_data/numimages/6_r.png"
    img_list = np.array([img1, img2, img3, img4])

    # test function "process_img_for_input()"
    plt.figure(figsize=(2, 4))
    for i in range(len(img_list)):
        # load image and translate
        filename = img_list[i]
        img_src = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        img_dst = process_img_for_input(img_src, output_size, denoising_ratio, corner_size_ratio, max_limitation_img_size)

        # plot input images
        plt.subplot(2, 4, i+1)
        plt.title("in:{}".format(str(i+1)))
        plt.imshow(img_src)
        plt.axis("off")
        
        # plot output images
        plt.subplot(2, 4, i+1+4)
        plt.gray()
        plt.title("out:{}".format(str(i+1)))
        plt.imshow(img_dst)
        plt.axis("off")
    
    plt.show()
