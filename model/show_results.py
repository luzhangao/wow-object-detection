# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/11/18
@description:
"""

import os
import cv2
import torch
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from control.parse_annotations import yolo_to_coco
from control.check_bounding_box import visualize_bbox

image_sample_paths = {
    "val2017": "../raw_data/coco2017/images/sample_val2017/",
    "train2017": "../raw_data/coco2017/images/sample_train2017/",
    # "test2017": "../raw_data/coco2017/images/sample_test2017/"
}
label_paths = {
    "val2017": "../raw_data/coco2017/labels/val2017/",
    "train2017": "../raw_data/coco2017/labels/train2017/",
    # "test2017": "../raw_data/coco2017/labels/test2017/",
    "coco128": "../datasets/coco128/labels/train2017/"
}
label_sample_paths = {
    "val2017": "../raw_data/coco2017/labels/sample_val2017/",
    "train2017": "../raw_data/coco2017/labels/sample_train2017/",
    # "test2017": "../raw_data/coco2017/labels/sample_test2017/",
}


def visualize_images(image, bboxes, category_ids, category_id_to_name, result_image):
    """
    A modified function (control.check_bounding_box.visualize) to display two images at a time.
    :param image:
    :param bboxes:
    :param category_ids:
    :param category_id_to_name:
    :param result_image:
    :return:
    """
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure()
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    # plt.show()
    plt.pause(10)
    plt.close()


def show_images():
    result_image_path = "../raw_data/coco2017/images/result_sample_val2017/"
    # key = "train2017"
    key = "val2017"
    image_path = image_sample_paths[key]
    label_path = label_sample_paths[key]

    category_id_to_name = {0: 'person', 1: 'car', 2: 'chair', 3: 'book', 4: 'bottle'}

    for filename in os.listdir(label_path):
        """
        The content of the text file:
        2 0.670773 0.262381 0.055859 0.101052
        2 0.94168 0.212268 0.034734 0.090887
        0 0.673516 0.596629 0.553469 0.775278
        0 0.476828 0.292237 0.352094 0.574784
        """
        df = pd.read_csv(label_path + filename, header=None, sep=" ")
        df.columns = ["category_id", "x_center", "y_center", "width", "height"]
        # print(df)
        category_ids = df["category_id"].tolist()
        image_file_name = filename[: len(filename)-3] + "jpg"
        current_image = cv2.imread(image_path + image_file_name)
        df["bbox"] = df.apply(lambda x: yolo_to_coco([x["x_center"], x["y_center"], x["width"], x["height"]], current_image.shape[1], current_image.shape[0]), axis=1)
        bboxes = df["bbox"].tolist()
        result_image = cv2.imread(result_image_path + image_file_name)
        visualize_images(current_image, bboxes, category_ids, category_id_to_name, result_image)


if __name__ == '__main__':
    show_images()


