import cv2
import numpy as np
from PIL import Image


def bwinverter(img_input, ratio=1.0, th=128):
    """
    invert black and white area so that number area is white
    if 4 corner'size areas have white area more than black one, invert

    :param img_input: ndarray, 1ch image (binary image))
    :param ratio: 0 - 1.0, coner size which is used to judge if inverted of not
    :return img_dst: ndarray, 1ch image
    :return flag: boolean, True: inverted, False: not inverted
    """

    # get size of input image
    h = img_input.shape[0]
    w = img_input.shape[1]

    # if ratio is not between 0 and 1
    if (ratio > 1.0):
        ratio = 1.0
    elif (ratio < 0):
        ratio = 0.0

    # corner size
    h_size = int(h * ratio)
    w_size = int(w * ratio)

    # if corner area size is zero
    if (h_size == 0):
        h_size = 1
    if (w_size == 0):
        w_size = 1

    # get 4 corner images
    area = np.empty([4, h_size, w_size])
    area[0] = img_input[0:h_size, 0:w_size]    # upper left
    area[1] = img_input[0:h_size, w-w_size:w]  # upper right
    area[2] = img_input[h-h_size:h, 0:w_size]  # lower left
    area[3] = img_input[h-h_size:h, w-w_size:w]  # lower right

    # if white area is more than black one, invert
    if (np.mean(area) < th):
        img_dst = img_input
        flag = False
    else:
        img_dst = cv2.bitwise_not(img_input)
        flag = True

    return img_dst, flag


def resize_keeping_aspect_ratio(img_input, size, OPT='LONG'):
    """
    resize image keeping aspect ratio

    :param img_input: ndarray, 1ch or 3ch image
    :param size: int, size that we want to resize
    :param OPT: str, 'LONG' or 'SHORT', fit to longer or shorter of image
    :return img_dst: ndarray, resized image
    """

    # get size of input image
    height_input = img_input.shape[0]
    width_input = img_input.shape[1]

    # -*- exception part -*-
    # OPT must be 'LONG' or 'SHORT'
    assert (OPT == 'LONG' or OPT == 'SHORT'), (
        "OPT must be 'LONG' or 'SHORT'.")

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


def make_procimg(img_input, size=0, OPT="GRAY", opening_ratio=0.01):
    """
    make binary or gray image by processing image that user inputs

    :param img_input: ndarray, "uint8" image (rgb or gray)
    :param size: int, resized number
    :param OPT: str, "GRAY" or "BIN", output image type
    :param opening_ratio: 0 - 1.0, opening size
    :return img_bin: ndarray, binarized image
    """

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

    # invertion of black and white
    img_bin, _ = bwinverter(img_bin, 0.1, ret)

    # return gray image or binary image
    if (OPT == "GRAY"):
        # resize
        if (size >= 1):
            img_gray = resize_keeping_aspect_ratio(img_gray, size, 'LONG')
        # invert black and white if flag is True
        _, flag = bwinverter(img_gray, 0.1, ret)
        if (flag is True):
            img_gray = cv2.bitwise_not(img_gray)

        # adjust aspect ratio of image to 1:1
        img_gray_square = make_square_img(img_gray)
        return img_gray_square
    elif (OPT == "BIN"):
        # resize
        if (size >= 1):
            img_bin = resize_keeping_aspect_ratio(img_bin, size, 'LONG')
        # adjust aspect ratio of image to 1:1
        img_bin_square = make_square_img(img_bin)
        return img_bin_square


if __name__ == '__main__':
    # load image (Attension: use 8bit image)
    filename = "/Users/khashimoto/Desktop/workspace/6_.png"
    img_src = cv2.imread(filename, cv2.IMREAD_COLOR)

    # convert to bin image
    # img_bin = binarize_kmeans(img_src, 28, 5, 50, 150)
    img_square = make_procimg(img_src, 28, "GRAY", 0.01)
    print(img_square.shape)

    # # show image
    cv2.imshow("", img_square)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
