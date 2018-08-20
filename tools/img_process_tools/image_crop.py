import os
import cv2


IMG_DIR = './imgs'
OUT_DIR = './out'
POSITION_FILE = 'Position.txt'
position = dict()
CROP_UNIT = 4


for im_name in os.listdir(IMG_DIR):
    im_file = os.path.join(IMG_DIR, im_name)
    img = cv2.imread(im_file, 1)

    h = img.shape[0]
    w = img.shape[1]

    u_h = float(h) / float(CROP_UNIT)
    u_w = float(w) / float(CROP_UNIT)

    count = 1

    for i in range(CROP_UNIT - 1):
        for j in range(CROP_UNIT - 1):
            y = int(i * u_h)
            x = int(j * u_w)
            img_name = im_name.split(".")[0] + "-" + str(count) + ".jpg"
            count += 1
            position[img_name] = (x, y, 2 * u_w, 2 * u_h)
            crop_img = img[y:y + 2 * int(u_h), x:x + 2 * int(u_w)]

            cv2.imwrite(os.path.join(OUT_DIR, img_name), crop_img)
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)

    # print(position)

    with open(POSITION_FILE, 'w') as fw:
        for key, value in position.items():
            result_string = key
            for item in value:
                result_string += " "
                result_string += str(item)
            # print(result_string)
            fw.write(result_string + '\n')
