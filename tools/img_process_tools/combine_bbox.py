
POSITION_FILE = "Position.txt"
BBOX_FILE = 'bbox.txt'
SUBMISSION_FILE = "submission_59.txt"

position = dict()
bboxs = dict()
final_bbox = dict()


with open(POSITION_FILE, 'r') as fr:
    for idx, line in enumerate(fr):
        parts = line.split(" ")
        position[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        bboxs[parts[0]] = list()
        ori_im_name = parts[0].split("-")[0] + ".jpg"
        final_bbox[ori_im_name] = list()
fr.close()

with open(BBOX_FILE, 'r') as fr:
    for idx, line in enumerate(fr):
        parts = line.split(" ")
        # Format img00001-1.jpg: (x, y, w, h, c)
        # c is the confidence
        bboxs[parts[0]].append((float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[1])))
fr.close()

for key, value in bboxs.items():
    for bbox in value:
        f_box = (bbox[0] + position[key][0], bbox[1] + position[key][1], bbox[2], bbox[3], bbox[4])
        im_name = key.split("-")[0] + ".jpg"
        final_bbox[im_name].append(f_box)

print(final_bbox)

with open(SUBMISSION_FILE, "w") as fr:
    for key, value in final_bbox.items():
        for bbox in value:
            res_string = str(key) + \
                         " " + str(bbox[4]) + \
                         " " + str(bbox[0]) + \
                         " " + str(bbox[1]) + \
                         " " + str(bbox[2]) + \
                         " " + str(bbox[3]) + "\n"
            fr.write(res_string)
fr.close()
