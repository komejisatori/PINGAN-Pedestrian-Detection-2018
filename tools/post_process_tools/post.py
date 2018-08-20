IOU_THRESH = 0.85
RESULT_FILE = "submission_ensemble_0.6.txt"
OUTPUT_FILE = "submission_ensemble_0.6_" + str(IOU_THRESH) + ".txt"
bbox = dict()
output_bbox = dict()


def combine_box(box1, box2):
    n_box = list()
    for i in range(len(box1)):
        n_box.append((box1[i] + box2[i]) / 2.0)
    return n_box


def get_coords(box):
    x1 = float(box[1])
    y1 = float(box[2])
    x2 = float(box[1] + box[3])
    y2 = float(box[2] + box[4])
    return x1, x2, y1, y2


def compute_iou(box1, box2):
    x11, x12, y11, y12 = get_coords(box1)
    x21, x22, y21, y22 = get_coords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou


def deal_one_image(boxes):
    n_boxes = list()
    for box1 in boxes:
        for box2 in boxes:
            if box1 == box2:
                continue
            else:
                iou = compute_iou(box1, box2)
                if iou > IOU_THRESH:
                    """
                    if box1[0] != box2[0]:
                        
                        if box1[0] == 2:
                            if box2 in boxes:
                                boxes.remove(box2)
                        else:
                            if box1 in boxes:
                                boxes.remove(box1)
                        

                        if box1[1] > box2[1]:
                            if box2 in boxes:
                                boxes.remove(box2)
                        else:
                            if box1 in boxes:
                                boxes.remove(box1)

                    """
                    # Remain High Confidence

                    if box1[0] >= box2[0]:
                        if box2 in boxes:
                            boxes.remove(box2)
                    else:
                        if box1 in boxes:
                            boxes.remove(box1)

                    """
                    # Remain Large or Small Area
                    if box1[3] * box1[4] <= box2[3] * box2[4]:
                        if box2 in boxes:
                            boxes.remove(box2)
                    else:
                        if box1 in boxes:
                            boxes.remove(box1)
                    
                    n_box = combine_box(box1, box2)
                    n_boxes.append(n_box)
                    if box1 in boxes:
                        boxes.remove(box1)
                    if box2 in boxes:
                        boxes.remove(box2)
                    """

    boxes.extend(n_boxes)

    return boxes


with open(RESULT_FILE, "r") as fw:
    for idx, line in enumerate(fw):
        parts = line.split(" ")
        box = list()
        """
        box.append(float(parts[1]))  # Class
        box.append(float(parts[2]))  # Confidence
        box.append(float(parts[3]))  # x
        box.append(float(parts[4]))  # y
        box.append(float(parts[5]))  # w
        box.append(float(parts[6]))  # h
        """
        box.append(float(parts[1]))  # Confidence
        box.append(float(parts[2]))  # x
        box.append(float(parts[3]))  # y
        box.append(float(parts[4]))  # w
        box.append(float(parts[5]))  # h

        if parts[0] not in bbox.keys():
            bbox[parts[0]] = list()
            output_bbox[parts[0]] = list()
            bbox[parts[0]].append(box)
        else:
            bbox[parts[0]].append(box)

fw.close()

for key, value in bbox.items():
    output = deal_one_image(value)
    output_bbox[key] = output

with open(OUTPUT_FILE, "w") as fr:
    count = 0
    for key, value in output_bbox.items():
        for box in value:
            single_str = str(key)
            single_str += " "
            single_str += str(box[0])
            single_str += " "
            single_str += str(box[1])
            single_str += " "
            single_str += str(box[2])
            single_str += " "
            single_str += str(box[3])
            single_str += " "
            single_str += str(box[4])
            # single_str += " "
            # single_str += str(box[5])
            single_str += "\n"

            count += 1
            fr.write(single_str)

fr.close()

print(count)
