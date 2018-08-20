import numpy as np
import cv2
import os
import math
import time

ALPHA = 0.055


def linear(one):
    """

    :param one: One Channel, R, G or B
    :return:
    """
    return float(one) / 12.92 if float(one) <= 0.04045 else math.pow((float(one) + ALPHA) / (1 + ALPHA), 2.4)


def transform_channel(channel):
    """

    :param channel: [R, G, B]
    :return: new channel: [R, G, B]
    """
    # Normalization
    channel = np.divide(channel, 255.0)
    #channel = [[[linear(s[0]), linear(s[1]), linear(s[2])] for s in x] for x in channel]
    c_linear = np.divide(channel, 12.92)
    c_power = np.power(np.divide(np.add(channel, ALPHA), 1 + ALPHA), 2.4)
    channel = np.where(channel <= 0.04045, c_linear, c_power)

    channel = np.asarray(channel)

    channel = np.multiply(channel, 255.0)
    channel = np.add(channel, 1.0)
    channel = np.log2(channel)
    channel = np.divide(channel, 8.0)
    # channel = np.multiply(channel, 255.0)

    return channel


def transformation(c):
    """

    :param img: [[[R, G, B], ... []], ...[]]
    :return:
    """

    c = transform_channel(c.astype(float))

    return c


if __name__ == '__main__':

    img_dir = "./one"
    output_dir = "./out_reverse"
    count = 0

    for im_name in os.listdir(img_dir):
        st = time.time()
        im_file = os.path.join(img_dir, im_name)
        # im_out_file = os.path.join(output_dir,im_name)
        # if os.path.isfile(im_out_file):
        #     print(im_name + " Image Exist")
        img = cv2.imread(im_file)
        c = img.copy()
        copy = transformation(c)
        channel = np.multiply(copy, 255)
        channel = channel.astype(int)

        count += 1
        print("Finish Image " + str(count))
        print("Time: " + str(time.time() - st))

        # numpy_vertical_concat = np.concatenate((img, channel), axis=0)
        cv2.imwrite(output_dir + "/" + im_name, channel)
