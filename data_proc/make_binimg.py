import cv2
import numpy as np


def binarize(img_input, th=100):
    """
    binarize input image using fixed threshold

    :param img_imput: ndarray, 8bit rgb image
    :param th: int, threshold of binary
    :return img_bin: binarized image
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
    :return img_bin: binarized image
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


def binarize_kmeans(img_input, size=28, itr=5, mu1_init=50, mu2_init=150):
    """
    binarize input image using k-means algorithm

    :param img_input: ndarray, 8bit rgb image
    :param size: int, size of resized input image
    :param itr: int, iterator of k-means algorithm
    :param mu1_init: int, initial centroid of cluster1
    :param mu2_init: int, initial centroid of cluster2
    :return img_bin: binarized image
    """

    # cnvert rgb image to gray image
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # resize image
    img_gray_resized = cv2.resize(img_gray, (size, size))

    # histogram
    hist, bins = np.histogram(img_gray_resized.ravel(), 256, [0, 256])

    # -- apply k-means algoritym -- #
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
    max_pixel = 255
    ret, img_bin = cv2.threshold(img_gray,
                                 thresh,
                                 max_pixel,
                                 cv2.THRESH_BINARY)
    return img_bin


if __name__ == '__main__':
    # load image (Attension: use 8bit image)
    filename = "input image path"
    img_src = cv2.imread(filename)

    # convert to bin image
    img_bin = binarize_kmeans(img_src, 28, 5, 50, 150)

    # # show image
    cv2.imshow("", img_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
