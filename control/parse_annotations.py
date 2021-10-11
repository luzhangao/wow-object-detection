# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/10/9
@description: parse the annotation json file
https://cocodataset.org/#format-data
https://zhuanlan.zhihu.com/p/29393415
"""

import os
import cv2
from pycocotools.coco import COCO
from utils.gerenal_tools import open_json, save_text
from control.check_bounding_box import visualize


def parse_annotation_file():
    """
    check some measures
    :return:
    """
    fpath = "../raw_data/coco2017/2017_TrainVal_annotations/annotations/instances_val2017.json"
    # fpath = "../raw_data/coco2017/2017_TrainVal_annotations/annotations/captions_val2017.json"
    # fpath = "../raw_data/coco2017/2017_TrainVal_annotations/annotations/person_keypoints_val2017.json"

    instances_val2017 = COCO(fpath)
    for label in instances_val2017.dataset["categories"]:
        print(label["id"], label["name"])
    print(len(instances_val2017.dataset["categories"]))  # 80
    print(len(instances_val2017.dataset["images"]))  # 5000
    print(len(instances_val2017.dataset["annotations"]))  # 36781

    f = open_json(fpath)
    category_id_group = set()
    image_id_group = list()
    for elem in f["annotations"]:
        category_id_group.add(elem["category_id"])
        image_id_group.append(elem["image_id"])

    print(sorted(category_id_group))
    # print(sorted(image_id_group))


def coco_to_yolo(bbox, width, height):
    """
    coco bbox -> yolo bbox
    [x_min, y_min, width, height] -> normalized [x_center, y_center, width, height]
    [98, 345, 322, 117] -> [0.4046875, 0.8613583, 0.503125, 0.24375]
    :param bbox: list, [x_min, y_min, width, height]
    :param width: int  the width px of the picture
    :param height: int  the height px of the picture
    :return: normalized [x_center, y_center, width, height]
    """
    x_min, y_min, coco_w, coco_h = bbox
    x_center = (x_min + coco_w / 2) / width
    y_center = (y_min + coco_h / 2) / height
    yolo_w = coco_w / width
    yolo_h = coco_h / height
    yolo_bbox = [x_center, y_center, yolo_w, yolo_h]
    return list(map(lambda x: round(x, 6), yolo_bbox))


def generate_labels():
    """
    generate labels txt files based on coco2017 dataset
    :return:
    """
    annotation_path = "../raw_data/coco2017/2017_TrainVal_annotations/annotations/"
    json_paths = {
        "val2017": "instances_val2017.json",
        "train2017": "instances_train2017.json"
    }
    image_paths = {
        "val2017": "../raw_data/coco2017/images/val2017/",
        "train2017": "../raw_data/coco2017/images/train2017/"
    }
    label_paths = {
        "val2017": "../raw_data/coco2017/labels/val2017/",
        "train2017": "../raw_data/coco2017/labels/train2017/"
    }

    for key in json_paths:
        fpath = annotation_path + json_paths[key]
        instances_dataset = COCO(fpath)
        # Generate a dictionary for categories
        # [{'supercategory': 'person', 'id': 1, 'name': 'person'}, ...] ==> {1: "person", ...}
        category_id_to_name = {}
        for elem in instances_dataset.dataset["categories"]:
            category_id_to_name[elem["id"]] = elem["name"]
        # Remove null labels and reindex the labels as 0 - {nc -1}
        yolo_categories = {}
        coco_to_yolo_map = {}
        for elem in zip(range(0, len(category_id_to_name)), sorted(category_id_to_name.items(), key=lambda kv: (kv[1], kv[0]))):
            coco_to_yolo_map[elem[1][0]] = elem[0]  # {coco index: yolo index} e.g. {1: 0} for "person"
            yolo_categories[elem[0]] = elem[1][1]  # {yolo index: name} e.g.  {0: "person"}

        for annot in instances_dataset.dataset["annotations"]:
            image_file_name = "000000" + str(annot["image_id"]) + ".jpg"
            label_file_name = "000000" + str(annot["image_id"]) + ".txt"
            if os.path.isfile(image_paths[key] + image_file_name):
                current_image = cv2.imread(image_paths[key] + image_file_name)  # shape (height, width, channels)
                # Display the picture with bounding box
                # visualize(current_image, [annot["bbox"]], [annot["category_id"]], category_id_to_name)

                yolo_bbox = coco_to_yolo(annot["bbox"], current_image.shape[1], current_image.shape[0])
                temp = str(coco_to_yolo_map[annot["category_id"]]) + " " + " ".join(map(lambda x: str(x), yolo_bbox)) + "\n"
                print(temp)
                save_text(temp, label_paths[key] + label_file_name, "a+")


if __name__ == '__main__':
    # parse_annotation_file()
    generate_labels()

