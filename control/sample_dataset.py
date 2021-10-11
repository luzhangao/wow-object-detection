# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/10/11
@description:
sample the dataset according to the current labels
"""

import os
import random
import shutil


def sample_and_save(**kwargs):
    """
    sample the labels and save corresponding images into another file folder
    :param: kwargs {key: number of samples}
    :return:
    """
    image_paths = {
        "val2017": "../raw_data/coco2017/images/val2017/",
        "train2017": "../raw_data/coco2017/images/train2017/"
    }
    image_copy_paths = {
        "val2017": "../raw_data/coco2017/images/sample_val2017/",
        "train2017": "../raw_data/coco2017/images/sample_train2017/"
    }
    label_paths = {
        "val2017": "../raw_data/coco2017/labels/val2017/",
        "train2017": "../raw_data/coco2017/labels/train2017/"
    }
    label_copy_paths = {
        "val2017": "../raw_data/coco2017/labels/sample_val2017/",
        "train2017": "../raw_data/coco2017/labels/sample_train2017/"
    }
    for key in label_paths:
        label_path = label_paths[key]
        all_labels = os.listdir(label_path)  # Find all label files
        sampling_labels = random.sample(all_labels, kwargs[key])  # Sampling
        for label in sampling_labels:
            image_name = label[:len(label)-4] + ".jpg"
            print(image_name)
            # Copy the image file
            image_source_path = image_paths[key] + image_name
            image_destination_path = image_copy_paths[key] + image_name
            shutil.copy(image_source_path, image_destination_path)
            # Copy the label file
            label_source_path = label_path + label
            label_destination_path = label_copy_paths[key] + label
            shutil.copy(label_source_path, label_destination_path)


if __name__ == '__main__':
    sample_and_save(val2017=100, train2017=500)

