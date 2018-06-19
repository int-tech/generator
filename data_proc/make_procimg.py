import cv2
import numpy as np
import util
import validation
from PIL import Image


def crop_four_corners_image(img_input, corner_size_ratio=1.0):
    """
    crop four corners image for black_white_inverter()
    use four corners image to judge if input image should be inverted or not

        :param  img_input         : ndarray, 1ch image (binary image))
        :param  corner_size_ratio : 0 - 1.0, corner size ratio which is used to judge if image is inverted
        :return img_corners       : four corner images
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
        :return flag_inversion_activation : boolean, True: inverted, False: not inverted
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
        flag_inversion_activation = False
    else:
        # inversion
        img_dst = cv2.bitwise_not(img_input)
        flag_inversion_activation = True

    return img_dst, flag_inversion_activation


def resize_keeping_aspect_ratio(img_input, size, OPT='LONG'):
    """
    resize image keeping aspect ratio

        :param  img_input : ndarray, 1ch or 3ch image
        :param  size      : int    , size that we want to resize
        :param  OPT       : str    , 'LONG' or 'SHORT', fit to longer or shorter length of image
        :return img_dst   : ndarray, resized image
    """

    # exception handling : OPT
    validation.validate_option_resize_keeping_aspect_ratio(OPT)

    # get size of input image
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # FIXME: below part should be change to exception part
    #        in this case, finish this program
    # if size is not integer
    size = int(size)
    # if setting size is equal to or less than zero
    ratio_min = min(width_input, height_input) / max(width_input, height_input)
    if (size <= 0 or int(size*ratio_min) <= 0):
        print("Output image size is equal to or less than zero.")
        print("Please set more larger size")
        img_dst = img_input
        return img_dst

    # resize
    if (height_input > width_input):
        if (OPT == 'LONG'):
            ratio = width_input / height_input
            img_dst = cv2.resize(img_input, (int(size*ratio), size))
        if (OPT == 'SHORT'):
            ratio = height_input / width_input
            img_dst = cv2.resize(img_input, (size, int(size*ratio)))
    else:
        if (OPT == 'LONG'):
            ratio = height_input / width_input
            img_dst = cv2.resize(img_input, (size, int(size*ratio)))
        if (OPT == 'SHORT'):
            ratio = width_input / height_input
            img_dst = cv2.resize(img_input, (int(size*ratio), size))

    return img_dst


def make_square_img(img_input):
    """
    make square image (1:1) from input image

        :param img_input: ndarray, front image, 1ch or 3ch image
        :return img_square: ndarray, square image
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

    # paste (approximately center)
    if (height_input > width_input):
        paste_position = int((height_input - width_input)/2)
        img_back_pil.paste(img_input_pil, (paste_position, 0))
    else:
        paste_position = int((width_input - height_input)/2)
        img_back_pil.paste(img_input_pil, (0, paste_position))

    # convert PIL to ndarray
    img_square = np.asarray(img_back_pil)

    return img_square


def thresh_kmeans(img_input, itr=5, mu1_init=50, mu2_init=150):
    """
    binarize input image using k-means algorithm

        :param img_input: ndarray uint8, 1ch image
        :param itr: int, iterator of k-means algorithm
        :param mu1_init: int, initial centroid of cluster1
        :param mu2_init: int, initial centroid of cluster2
        :return img_bin: ndarray uint8, binarized image
    """

    # histogram
    hist, bins = np.histogram(img_input.ravel(), 256, [0, 256])

    # -- k-means algoritym -- #
    # initialize mean (centroid)
    mu1_current = mu1_init  # initial centroid of cluster1
    mu2_current = mu2_init  # initial centroid of cluster2

    # iterative calculation
    itr_cnt = 0  # iterate conter
    while(itr_cnt < itr):
        # initialize
        mu1_updated = 0
        mu2_updated = 0
        sum1 = 0
        sum2 = 0
        cnt1 = 0
        cnt2 = 0
        for i in range(len(hist)):
            # in the case of belonging to cluster1
            if (np.abs(i - mu1_current) < np.abs(i - mu2_current)):
                sum1 += hist[i]*i
                cnt1 += hist[i]
            # in the case of belonging to cluster2
            else:
                sum2 += hist[i]*i
                cnt2 += hist[i]
        # update each centroid
        mu1_updated = sum1 / cnt1
        mu2_updated = sum2 / cnt2
        mu1_current = mu1_updated
        mu2_current = mu2_updated
        itr_cnt += 1

    # calculate binary threshold
    thresh = (mu1_current + mu2_current) / 2
    # convert gray image to binary image
    # Note: THRESH_OTSU is also good
    max_pixel = 255
    ret, img_bin = cv2.threshold(img_input,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    return img_bin


def make_procimg(img_input, size=0, OPT="GRAY", opening_ratio=0.01, bw_inv_size=0.2):
    """
    make binary or gray image by processing image that user inputs

        :param img_input: ndarray, "uint8" image (rgb or gray)
        :param output_size: int, resized number, if 0 is set, input image size is output
        :param OPT: str, "GRAY" or "BIN", output image type
        :param opening_ratio: 0 - 1.0, opening size
        :return img_bin: ndarray, binarized image
    """

    # FIXME: revize assert part
    # -*- exception part -*-
    # OPT must be 'LONG' or 'SHORT'
    assert (OPT == 'GRAY' or OPT == 'BIN'), (
        "OPT must be 'GRAY' or 'BIN'.")

    # if input image size is too large, resize into 1080 size
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]
    if (max(height_input, width_input) > 1080):
        img_src = resize_keeping_aspect_ratio(img_input, 1080, 'LONG')
    else:
        img_src = img_input

    # if input is rgb image, convert rgb image to gray image
    if (len(img_src.shape) == 3):
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # if input is gray image
    else:
        img_gray = img_src

    # convert gray image to binary image using Otsu'size method
    # img_bin = thresh_kmeans(img_gray)
    ret, img_bin = cv2.threshold(img_gray,
                                 0,
                                 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # opening for eliminating small noise (erosion -> dilation)
    if (opening_ratio > 1.0):
        opening_ratio = 1.0
    if (opening_ratio < 0.0):
        opening_ratio = 0.0
    opening_size = int(((width_input + height_input) / 2) * opening_ratio)
    if (opening_size > 0):
        kernel = np.ones((opening_size, opening_size), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

    # invert black and white pixel so that number area is white pixel
    # And this image is used as mask image
    img_mask, _ = black_white_inverter(img_bin, bw_inv_size, ret)

    # return gray image or binary image
    if (OPT == "GRAY"):
        # resize
        if (size >= 1):
            img_gray = resize_keeping_aspect_ratio(img_gray, size, 'LONG')
            img_mask = resize_keeping_aspect_ratio(img_mask, size, 'LONG')
        # invert black and white of gray image if flag is True
        _, flag = black_white_inverter(img_gray, bw_inv_size, ret)
        if (flag is True):
            img_out_gray = cv2.bitwise_not(img_gray)
        else:
            img_out_gray = img_gray
        # change into black excluding number area
        img_out_gray_mask = cv2.bitwise_and(img_out_gray, img_out_gray, mask=img_mask)

        # adjust aspect ratio of image to 1:1
        img_out_square = make_square_img(img_out_gray_mask)
        return img_out_square
    elif (OPT == "BIN"):
        # resize
        if (size >= 1):
            img_out_bin = resize_keeping_aspect_ratio(img_mask, size, 'LONG')
        else:
            img_out_bin = img_mask
        # adjust aspect ratio of image to 1:1
        img_out_square = make_square_img(img_out_bin)
        return img_out_square


if __name__ == '__main__':
    # load image (Attension: use 8bit image)
    filename = "../../test_data/numimages/6_r.png"
    img_src = cv2.imread(filename, cv2.IMREAD_COLOR)

    # convert to bin image
    # img_bin = binarize_kmeans(img_src, 28, 5, 50, 150)
    img_square = make_procimg(img_src, 0, "BIN", 0.01, 0.2)
    print(img_square.shape)

    # show image
    cv2.imshow("", img_square)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
