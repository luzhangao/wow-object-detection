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
from utils.gerenal_tools import open_text


image_paths = {
    "val2017": "../raw_data/coco2017/images/val2017/",
    "train2017": "../raw_data/coco2017/images/train2017/",
    "test2017": "../raw_data/coco2017/images/test2017/",
    "coco128": "../datasets/coco128/images/train2017/"
}
image_sample_paths = {
    "val2017": "../raw_data/coco2017/images/sample_val2017/",
    "train2017": "../raw_data/coco2017/images/sample_train2017/",
    "test2017": "../raw_data/coco2017/images/sample_test2017/"
}
label_paths = {
    "val2017": "../raw_data/coco2017/labels/val2017/",
    "train2017": "../raw_data/coco2017/labels/train2017/",
    "test2017": "../raw_data/coco2017/labels/test2017/",
    "coco128": "../datasets/coco128/labels/train2017/"
}
label_sample_paths = {
    "val2017": "../raw_data/coco2017/labels/sample_val2017/",
    "train2017": "../raw_data/coco2017/labels/sample_train2017/",
    "test2017": "../raw_data/coco2017/labels/sample_test2017/",
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
    coco128 train2017 val2017
    929 712378 30468
    0 254 4339 123
    1 6 4877 206
    2 46 7167 286
    3 5 7739 295
    4 6 2706 114
    5 7 3117 124
    6 3 1084 60
    7 12 3462 128
    8 6 8227 335
    9 14 5978 241
    11 2 8968 353
    13 9 20231 855
    14 16 11812 508
    15 4 5987 236
    16 9 5088 237
    17 2 5184 290
    20 17 3977 168
    21 1 5411 210
    22 4 31955 1422
    23 9 5207 227
    24 6 4828 218
    25 18 6781 340
    26 19 17197 729
    27 7 13026 557
    28 4 4532 182
    29 5 5958 289
    30 1 4615 221
    31 7 1530 82
    32 6 4494 180
    33 10 2200 98
    34 4 4152 203
    35 7 164 11
    36 5 10183 463
    38 7 2413 94
    39 18 2392 122
    40 16 7532 289
    41 36 6421 268
    42 6 4135 183
    43 16 1414 48
    44 22 7264 313
    45 28 1868 86
    46 1 5333 252
    48 2 1053 44
    49 4 217224 9000
    50 11 4765 248
    51 24 7163 286
    52 2 2204 104
    53 5 4673 235
    54 14 3657 146
    55 4 1226 33
    56 35 7891 337
    57 6 4690 191
    58 14 4602 151
    59 3 5555 200
    60 13 2302 52
    61 2 5130 209
    62 2 5215 221
    63 3 1658 65
    64 2 5026 265
    65 8 5124 217
    67 8 3984 176
    68 3 5349 180
    69 5 185 7
    71 6 1598 55
    72 5 10612 582
    73 29 3753 160
    74 9 8243 361
    75 2 4861 240
    76 1 9436 379
    77 21 5486 225
    79 5 4335 212
    :return:
    """
    coco128 = dataset_distributions("coco128", label_paths)
    train2017 = dataset_distributions("train2017", label_sample_paths)
    val2017 = dataset_distributions("val2017", label_sample_paths)
    print(len(coco128), len(train2017), len(val2017))
    for clf in set(coco128):
        print(clf, coco128.count(clf), train2017.count(clf), val2017.count(clf))


if __name__ == '__main__':
    # sample_and_save(val2017=1200, train2017=5000)
    analyze_distributions()
