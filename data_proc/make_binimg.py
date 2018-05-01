import cv2
import numpy as np


def binarize(img_input, th=100):
    """
    binarize input image using fixed threshold

    :param img_imput: ndarray, 8bit rgb image
    :param th: int, threshold of binary
    """

    # cnvert rgb image to gray image
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # convert gray image to binary image
    max_pixel = 255
    thresh = th
    ret, img_bin = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    return img_bin


def binarize_maxhist(img_input, th=50):
    """
    binarize input image using max value of histogram

    :param img_input: ndarray, 8bit rgb image
    :param th: int, threshold of binary
    """

    # cnvert rgb image to gray image
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # histogram
    hist, bins = np.histogram(img_gray.ravel(), 256, [0, 256])

    # convert gray image to binary image
    thresh = list(hist).index(max(hist)) - th
    max_pixel = 255
    ret, img_bin = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)

    return img_bin


if __name__ == '__main__':
    # load image (8bit)
    filename = "image path"
    img_src = cv2.imread(filename)

    # convert to bin image
    img_bin = binarize(img_src, 50)

    # show image
    cv2.imshow("", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
