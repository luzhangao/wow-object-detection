# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/10/10
@description:
Display the picture with bounding box
"""


import cv2
import pandas as pd
from matplotlib import pyplot as plt


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    """

    :param image:
    :param bboxes: list
           e.g. [[586.23, 324.18, 16.15, 38.93]]
    :param category_ids: list
           e.g. [44]
    :param category_id_to_name: dict
           e.g. {1: 'person', 2: 'bicycle', 3: 'car', ...}
    :return:
    """
    print(bboxes, category_ids, category_id_to_name)
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    import os
    from control.parse_annotations import yolo_to_coco
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

    key = "train2017"
    # key = "val2017"
    image_path = image_sample_paths[key]
    label_path = label_sample_paths[key]
    # names: ["person", "car", "chair", "book", "bottle"] # class names

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
        visualize(current_image, bboxes, category_ids, category_id_to_name)




