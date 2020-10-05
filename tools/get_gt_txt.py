#-------------------------------------#
#       mAP所需文件计算代码
#       具体教程请查看Bilibili
#       Bubbliiiing
#-------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

image_ids = open('C:/Users/Peter/PycharmProjects/datasets/tt100k/test/ids.txt').read().strip().split()
annos = json.loads(open('C:/Users/Peter/PycharmProjects/datasets/tt100k/annotations.json').read())
clses = open('../model_data/tt100k_classes.txt').read().splitlines()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        img = annos['imgs'][image_id]
        objs = img['objects']

        for obj in objs:
            obj_name = obj['category']
            if obj_name in clses:
                bbox = obj['bbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
                b = (xmin, ymin, xmax, ymax)
                new_f.write("%s %s %s %s %s\n" % (obj_name, xmin, ymin, xmax, ymax))
print("Conversion completed!")
