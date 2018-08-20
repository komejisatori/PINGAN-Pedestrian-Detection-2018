import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
import cv2
import os


def output_img_name(output_dir, origin_name):
    result_name = origin_name.split(".")[0] + "_grey_blur.jpg"
    return os.path.join(output_dir, result_name)


def get_corrdinate_from_bbox(bbox):
    return get_coordinate(bbox[0], bbox[1], bbox[2], bbox[3])


def get_coordinate(x, y, w, h):
    return x, y, x + w, y + h


def get_bbox(x1, y1, x2, y2):
    return x1, y1, x2 - x1, y2 - y1


def read_label(label_file):
    gt_lable = dict()
    with open(label_file) as fp:
        for i, line in enumerate(fp):
            key = line.split(" ")[0]
            gt_lable[key] = line

    fp.close()
    return gt_lable


def bbox_output(bbs_aug, img_name):
    single_string = ""
    single_string += img_name
    for i, bounding_box in enumerate(bbs_aug.bounding_boxes):

        single_string += " "
        single_string += str(bounding_box.label)

        x, y, w, h = get_bbox(bounding_box.x1, bounding_box.y1, bounding_box.x2, bounding_box.y2)
        single_string += " "
        single_string += str(x)
        single_string += " "
        single_string += str(y)
        single_string += " "
        single_string += str(w)
        single_string += " "
        single_string += str(h)

    return single_string


def image_show(image_before, image_after):
    both = np.hstack((image_before, image_after))
    misc.imshow(both)
    cv2.waitKey(0)


def process_grey_blur(image_file, image_name, gt_lables, output_dir):

    image = cv2.imread(image_file, 1)

    line = gt_lables[image_name]
    parts = line.split(" ")
    image_name = parts[0]
    image_box = list()
    for j in range(int((len(parts) - 1) / 5)):
        bbox = list()
        bbox.append(float(parts[2 + 5 * j]))
        bbox.append(float(parts[2 + 5 * j + 1]))
        bbox.append(float(parts[2 + 5 * j + 2]))
        bbox.append(float(parts[2 + 5 * j + 3]))
        x1, y1, x2, y2 = get_corrdinate_from_bbox(bbox)
        image_box.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(parts[2 + 5 * j - 1])))

    bbs = ia.BoundingBoxesOnImage(image_box, shape=image.shape)

    # bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=23, y1=56, x2=276, y2=287)], shape=image.shape)

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            # iaa.Grayscale(1),
            # iaa.GaussianBlur(2)
            iaa.Crop(px=(100, 200))
        ],
        random_order=True
    )

    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    image_before = bbs.draw_on_image(image, thickness=2)
    image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    # misc.imsave(output_img_name(output_dir, image_name), image_aug)

    image_show(image_before, image_after)

    return bbox_output(bbs_aug, image_name)


def bbox_write_file(all_lable, output_file):
    with open(output_file, "w+") as fw:
        for i, single_string in enumerate(all_lable):
            fw.write(single_string + '\n')
    fw.close()


if __name__ == '__main__':
    img_dir = "./images"
    label_file = "./label.txt"
    output_file = "./output_label.txt"
    output_dir = "./output"

    gt_lables = read_label(label_file)
    all_labels = list()

    for im_name in os.listdir(img_dir):
        im_file = os.path.join(img_dir, im_name)
        all_labels.append(process_grey_blur(im_file, im_name, gt_lables, output_dir))

    bbox_write_file(all_labels, output_file)
