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
import pandas as pd
from utils.gerenal_tools import open_text, open_yaml


image_paths = {
    "val2017": "../raw_data/coco2017/images/val2017/",
    "train2017": "../raw_data/coco2017/images/train2017/",
    # "test2017": "../raw_data/coco2017/images/test2017/",
    "coco128": "../datasets/coco128/images/train2017/"
}
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


def sample_and_save(**kwargs):
    """
    sample the labels and save corresponding images into another file folder
    :param: kwargs {key: number of samples}
    :return:
    """

    for key in label_paths:
        label_path = label_paths[key]
        all_labels = os.listdir(label_path)  # Find all label files
        if key != "coco128":
            sampling_labels = random.sample(all_labels, kwargs[key])  # Sampling
            for label in sampling_labels:
                image_name = label[:len(label)-4] + ".jpg"
                print(image_name)
                # Copy the image file
                image_source_path = image_paths[key] + image_name
                image_destination_path = image_sample_paths[key] + image_name
                shutil.copy(image_source_path, image_destination_path)
                # Copy the label file
                label_source_path = label_path + label
                label_destination_path = label_sample_paths[key] + label
                shutil.copy(label_source_path, label_destination_path)


def clear_train_and_val_set():
    """
    clear all images and label from sample_train2017 and val_train2017
    :return:
    """


def dataset_distributions(label_key, path_dict):
    """
    Compute the distributions of the classifications for the specific dataset.
    :param label_key: coco128
    :param path_dict: label_paths
    :return:
    """
    label_path = path_dict[label_key]
    all_labels = os.listdir(label_path)
    clfs = list()
    for label in all_labels:
        fpath = label_path + label
        f = open_text(fpath)
        if f:
            df = pd.read_csv(fpath, header=None, names=["classification", "x_center", "y_center", "width", "height"], sep=" ")
            clfs.extend(df["classification"])
    return clfs


def analyze_distributions():
    """

    :return:
    """
    # coco128 = dataset_distributions("coco128", label_paths)

    # train2017 = dataset_distributions("train2017", label_paths)
    # val2017 = dataset_distributions("val2017", label_paths)
    # temp = {"clf": list(), "train2017": list(), "val2017": list()}
    # for clf in set(train2017):
    #     print(clf, train2017.count(clf), val2017.count(clf))
    #     temp["clf"].append(clf)
    #     temp["train2017"].append(train2017.count(clf))
    #     temp["val2017"].append(val2017.count(clf))
    # df = pd.DataFrame(temp)
    # df.to_csv("coco2017_distribution.csv")

    # df = pd.read_csv("coco2017_distribution.csv", index_col=0)
    # coco_categories = open_yaml("../yolov5/data/coco128.yaml")["names"]
    # df["category_name"] = df["clf"].apply(lambda x: coco_categories[x])
    # # Display all rows
    # pd.set_option("display.max_rows", 100)
    # print(df.sort_values("train2017", ascending=False))

    train2017 = dataset_distributions("train2017", label_sample_paths)
    val2017 = dataset_distributions("val2017", label_sample_paths)
    df = pd.read_csv("coco2017_distribution.csv")
    print(df)
    for clf in set(train2017):
        print(clf, train2017.count(clf), val2017.count(clf),
              round(train2017.count(clf) / len(train2017), 4),
              round(val2017.count(clf) / len(val2017), 4),
              round(df.iloc[clf]["train2017"] / df["train2017"].sum(), 4),
              round(df.iloc[clf]["val2017"] / df["val2017"].sum(), 4))


if __name__ == '__main__':
    sample_and_save(val2017=800, train2017=4000)
    analyze_distributions()
