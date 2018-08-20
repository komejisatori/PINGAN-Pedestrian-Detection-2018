#!/usr/bin/env python3
"""
evaluate_training.py: draw bounding box on original image

This is a tool for drawing training loss of Detectron.
"""
__author__ = "Jiaqi Cai"
__email__ = "jiaqi.cai22@pactera.com"
import os
import re
import json
import pprint
import matplotlib.pyplot as plt

# Detectron log file
LOGFILE_PATH = './res/log'


lines = []
attr_dict = {}
with open(LOGFILE_PATH) as file:
    lines = file.readlines()
for line in lines:
    matchObj = re.match(r'json_stats: ([\S\s]*)', line)
    if matchObj:
        plain_json = matchObj.group(1)
        json_data = json.loads(plain_json)
        for json_key in json_data:
            if json_key in attr_dict:
                attr_dict[json_key].append(json_data[json_key])
            else:
                attr_dict[json_key] = [json_data[json_key]]
for attr in attr_dict:
    if attr == 'eta' or attr == 'time':
        continue
    plt.clf()
    plt.plot(attr_dict[attr])
    plt.savefig('%s.png' % attr)
