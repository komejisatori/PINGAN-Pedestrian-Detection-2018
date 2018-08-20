import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_coordinate(x, y, w, h):
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    # get (x, y, x, y) from (c, x, y, w, h)
    boxA = get_coordinate(box1[1], box1[2], box1[3], box1[4])
    boxB = get_coordinate(box2[1], box2[2], box2[3], box2[4])

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


annotation = dict()
count = 0

with open('val_annotations.txt', 'r') as fw:
    for idx, line in enumerate(fw):
        parts = line.split(" ")
        box_count = int((len(parts) - 1) / 5)
        img_name = parts[0].rstrip("\r\n")
        img_boxes = list()
        annotation[img_name] = list()
        for i in range(box_count):
            box = list()
            for j in range(5):
                box.append(int(parts[i * 5 + j + 1].rstrip("\r\n")))
            annotation[img_name].append(box)

fw.close()

# 0, 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
labels = [0, 0, 0, 0, 0, 0]

dict_0 = dict()
dict_0_2 = dict()
dict_2_4 = dict()
dict_4_6 = dict()
dict_6_8 = dict()
dict_8_10 = dict()


for key, value in annotation.items():
    dict_0[key] = list()
    dict_0_2[key] = list()
    dict_2_4[key] = list()
    dict_4_6[key] = list()
    dict_6_8[key] = list()
    dict_8_10[key] = list()

    for pair in itertools.combinations(value, 2):
        iou = compute_iou(pair[0], pair[1])
        if iou == 0.0:
            labels[0] += 1
            if pair[0] not in dict_0[key]:
                dict_0[key].append(pair[0])
            if pair[1] not in dict_0[key]:
                dict_0[key].append(pair[1])
        elif 0 < iou <= 0.2:
            labels[1] += 1
            if pair[0] not in dict_0_2[key]:
                dict_0_2[key].append(pair[0])
            if pair[1] not in dict_0_2[key]:
                dict_0_2[key].append(pair[1])
        elif 0.2 < iou <= 0.4:
            labels[2] += 1
            if pair[0] not in dict_2_4[key]:
                dict_2_4[key].append(pair[0])
            if pair[1] not in dict_2_4[key]:
                dict_2_4[key].append(pair[1])
        elif 0.4 < iou <= 0.6:
            labels[3] += 1
            if pair[0] not in dict_4_6[key]:
                dict_4_6[key].append(pair[0])
            if pair[1] not in dict_4_6[key]:
                dict_4_6[key].append(pair[1])
        elif 0.6 < iou <= 0.8:
            labels[4] += 1
            if pair[0] not in dict_6_8[key]:
                dict_6_8[key].append(pair[0])
            if pair[1] not in dict_6_8[key]:
                dict_6_8[key].append(pair[1])
        elif 0.8 < iou <= 1:
            labels[5] += 1
            if pair[0] not in dict_8_10[key]:
                dict_8_10[key].append(pair[0])
            if pair[1] not in dict_8_10[key]:
                dict_8_10[key].append(pair[1])
        else:
            pass


with open("val_annotation_0.txt", "w") as fw:
    count = 0
    for key, value in dict_0.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("0: " + str(count))
fw.close()

with open("val_annotation_0_2.txt", "w") as fw:
    count = 0
    for key, value in dict_0_2.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("0_2: " + str(count))
fw.close()

with open("val_annotation_2_4.txt", "w") as fw:
    count = 0
    for key, value in dict_2_4.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("2_4: " + str(count))
fw.close()

with open("val_annotation_4_6.txt", "w") as fw:
    count = 0
    for key, value in dict_4_6.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("4_6: " + str(count))
fw.close()


with open("val_annotation_6_8.txt", "w") as fw:
    count = 0
    for key, value in dict_6_8.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("6_8: " + str(count))
fw.close()

with open("val_annotation_8_10.txt", "w") as fw:
    count = 0
    for key, value in dict_8_10.items():
        if value:
            result = key
            for item in value:
                count += 1
                for v in item:
                    result += " "
                    result += str(v)
            fw.write(result + "\n")
        else:
            fw.write(key + "\n")
    print("8_10: " + str(count))
fw.close()

print(labels)

l = np.asarray(labels)
fig, ax = plt.subplots()
plt.bar(range(6), l, log=True)
for a,b in zip(range(6), l):
    plt.text(a, b, str(b), horizontalalignment='center')
plt.xticks((0, 1, 2, 3, 4, 5), ('0', '0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1'))
plt.show()
