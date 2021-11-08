# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/10/9
@description: parse the annotation json file
https://cocodataset.org/#format-data
https://zhuanlan.zhihu.com/p/29393415
parent
├── datasets
    └── coco128
        └── images
            └── train
            └── val
        └── labels
            └── train
            └── val
└── raw_data
    └── coco2017
        └── 2017_TrainVal_annotations
            └── annotations
        └── images
            └── train
            └── val
        └── labels
            └── train
            └── val
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


def yolo_to_coco(bbox, width, height):
    """
    yolo bbox -> coco bbox
    normalized [x_center, y_center, width, height] -> [x_min, y_min, width, height]
    [0.4046875, 0.8613583, 0.503125, 0.24375] -> [98, 345, 322, 117]
    :param bbox: list, normalized [x_center, y_center, width, height]
    :param width: int  the width px of the picture
    :param height: int  the height px of the picture
    :return: [x_min, y_min, width, height]
    """
    x_center, y_center, yolo_w, yolo_h = bbox
    coco_w = yolo_w * width
    coco_h = yolo_h * height
    x_min = x_center * width - coco_w / 2
    y_min = y_center * height - coco_h / 2
    coco_bbox = [x_min, y_min, coco_w, coco_h]
    return list(map(lambda x: round(x), coco_bbox))


def generate_labels():
    """
    generate labels txt files based on coco2017 dataset
    :return:
    """
    annotation_paths = {
        "val2017": "../raw_data/coco2017/2017_TrainVal_annotations/annotations/",
        "train2017": "../raw_data/coco2017/2017_TrainVal_annotations/annotations/",
    }
    json_paths = {
        "val2017": "instances_val2017.json",
        "train2017": "instances_train2017.json",
    }
    image_paths = {
        "val2017": "../raw_data/coco2017/images/val2017/",
        "train2017": "../raw_data/coco2017/images/train2017/",
    }
    label_paths = {
        "val2017": "../raw_data/coco2017/labels/val2017/",
        "train2017": "../raw_data/coco2017/labels/train2017/",
    }

    for key in json_paths:
        fpath = annotation_paths[key] + json_paths[key]
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
        """
        coco_to_yolo_map {5: 0, 53: 1, 27: 2, 52: 3, 39: 4, 40: 5, 23: 6, 65: 7, 15: 8, 2: 9, 16: 10, 9: 11, 84: 12, 44: 13, 51: 14, 56: 15, 6: 16, 61: 17, 3: 18, 57: 19, 17: 20, 77: 21, 62: 22, 85: 23, 63: 24, 21: 25, 47: 26, 67: 27, 18: 28, 60: 29, 22: 30, 11: 31, 48: 32, 34: 33, 25: 34, 89: 35, 31: 36, 19: 37, 58: 38, 76: 39, 38: 40, 49: 41, 73: 42, 78: 43, 4: 44, 74: 45, 55: 46, 79: 47, 14: 48, 1: 49, 59: 50, 64: 51, 82: 52, 75: 53, 54: 54, 87: 55, 20: 56, 81: 57, 41: 58, 35: 59, 36: 60, 50: 61, 37: 62, 13: 63, 33: 64, 42: 65, 88: 66, 43: 67, 32: 68, 80: 69, 70: 70, 90: 71, 10: 72, 7: 73, 8: 74, 72: 75, 28: 76, 86: 77, 46: 78, 24: 79}
        yolo_categories {0: 'airplane', 1: 'apple', 2: 'backpack', 3: 'banana', 4: 'baseball bat', 5: 'baseball glove', 6: 'bear', 7: 'bed', 8: 'bench', 9: 'bicycle', 10: 'bird', 11: 'boat', 12: 'book', 13: 'bottle', 14: 'bowl', 15: 'broccoli', 16: 'bus', 17: 'cake', 18: 'car', 19: 'carrot', 20: 'cat', 21: 'cell phone', 22: 'chair', 23: 'clock', 24: 'couch', 25: 'cow', 26: 'cup', 27: 'dining table', 28: 'dog', 29: 'donut', 30: 'elephant', 31: 'fire hydrant', 32: 'fork', 33: 'frisbee', 34: 'giraffe', 35: 'hair drier', 36: 'handbag', 37: 'horse', 38: 'hot dog', 39: 'keyboard', 40: 'kite', 41: 'knife', 42: 'laptop', 43: 'microwave', 44: 'motorcycle', 45: 'mouse', 46: 'orange', 47: 'oven', 48: 'parking meter', 49: 'person', 50: 'pizza', 51: 'potted plant', 52: 'refrigerator', 53: 'remote', 54: 'sandwich', 55: 'scissors', 56: 'sheep', 57: 'sink', 58: 'skateboard', 59: 'skis', 60: 'snowboard', 61: 'spoon', 62: 'sports ball', 63: 'stop sign', 64: 'suitcase', 65: 'surfboard', 66: 'teddy bear', 67: 'tennis racket', 68: 'tie', 69: 'toaster', 70: 'toilet', 71: 'toothbrush', 72: 'traffic light', 73: 'train', 74: 'truck', 75: 'tv', 76: 'umbrella', 77: 'vase', 78: 'wine glass', 79: 'zebra'}
        category_id_to_name {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
        """
        # print("coco_to_yolo_map", coco_to_yolo_map)
        # print("yolo_categories", yolo_categories)
        # print("category_id_to_name", category_id_to_name)
        for annot in instances_dataset.dataset["annotations"]:
            image_file_name = "000000" + str(annot["image_id"]) + ".jpg"
            label_file_name = "000000" + str(annot["image_id"]) + ".txt"
            if os.path.isfile(image_paths[key] + image_file_name):
                """
                80 categories is too much for the training dataset with 4000 or less images. According the result from
                control.sample_dataset.analyze_distributions (table below), I plan to choose the top 5 categories. 
                The top 5 categories are ["person", "car", "chair", "book", "bottle"] and their indices are 
                [49, 18, 22, 12, 13]
                    clf  train2017  val2017   category_name
                49   49     217224     9000          person
                18   18      36114     1636             car
                22   22      31955     1422           chair
                12   12      20688      981            book
                13   13      20231      855          bottle
                """
                map_dict = {49: 0, 18: 1, 22: 2, 12: 3, 13: 4}
                if coco_to_yolo_map[annot["category_id"]] in [49, 18, 22, 12, 13]:
                    current_image = cv2.imread(image_paths[key] + image_file_name)  # shape (height, width, channels)
                    # Display the picture with bounding box
                    # visualize(current_image, [annot["bbox"]], [annot["category_id"]], category_id_to_name)
                    yolo_bbox = coco_to_yolo(annot["bbox"], current_image.shape[1], current_image.shape[0])

                    # e.g. 45 0.479492 0.688771 0.955609 0.5955 \n
                    # temp = str(coco_to_yolo_map[annot["category_id"]]) + " " + " ".join(map(lambda x: str(x), yolo_bbox)) + "\n"
                    # The category index should be changed from [49, 18, 22, 12, 13] to range(0, 5) because Yolo will
                    # validate the indices.
                    temp = str(map_dict[coco_to_yolo_map[annot["category_id"]]]) + " " + " ".join(map(lambda x: str(x), yolo_bbox)) + "\n"
                    print(temp)
                    save_text(temp, label_paths[key] + label_file_name, "a+")  # Save and Append the line


if __name__ == '__main__':
    # parse_annotation_file()
    generate_labels()

