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

        for annot in instances_dataset.dataset["annotations"]:
            image_file_name = "000000" + str(annot["image_id"]) + ".jpg"
            label_file_name = "000000" + str(annot["image_id"]) + ".txt"
            if os.path.isfile(image_paths[key] + image_file_name):
                """
                80 categories is too much for the training dataset with 4000 or less images. According the result from
                control.sample_dataset.analyze_distributions (table below), I plan to choose the top 5 categories. 
                The top 5 categories are ["orange", "sheep", "zebra", "parking meter", "bench"] and their indices are 
                [49, 18, 22, 12, 13]
                    clf  train2017  val2017   category_name
                49   49     217224     9000          orange
                18   18      36114     1636           sheep
                22   22      31955     1422           zebra
                12   12      20688      981   parking meter
                13   13      20231      855           bench
                26   26      17197      729         handbag
                27   27      13026      557             tie
                14   14      11812      508            bird
                72   72      10612      582    refrigerator
                36   36      10183      463      skateboard
                76   76       9436      379        scissors
                10   10       9011      346    fire hydrant
                11   11       8968      353       stop sign
                74   74       8243      361           clock
                8     8       8227      335            boat
                56   56       7891      337           chair
                3     3       7739      295      motorcycle
                40   40       7532      289      wine glass
                44   44       7264      313           spoon
                2     2       7167      286             car
                51   51       7163      286          carrot
                25   25       6781      340        umbrella
                78   78       6565      285      hair drier
                19   19       6484      334             cow
                41   41       6421      268             cup
                15   15       5987      236             cat
                9     9       5978      241   traffic light
                29   29       5958      289         frisbee
                59   59       5555      200             bed
                77   77       5486      225      teddy bear
                21   21       5411      210            bear
                37   37       5400      232       surfboard
                68   68       5349      180       microwave
                46   46       5333      252          banana
                62   62       5215      221              tv
                23   23       5207      227         giraffe
                17   17       5184      290           horse
                61   61       5130      209          toilet
                65   65       5124      217          remote
                16   16       5088      237             dog
                64   64       5026      265           mouse
                1     1       4877      206         bicycle
                75   75       4861      240            vase
                24   24       4828      218        backpack
                50   50       4765      248        broccoli
                57   57       4690      191           couch
                53   53       4673      235           pizza
                30   30       4615      221            skis
                58   58       4602      151    potted plant
                28   28       4532      182        suitcase
                32   32       4494      180     sports ball
                0     0       4339      123          person
                79   79       4335      212      toothbrush
                34   34       4152      203    baseball bat
                42   42       4135      183            fork
                66   66       3997      156        keyboard
                67   67       3984      176      cell phone
                20   20       3977      168        elephant
                73   73       3753      160            book
                54   54       3657      146           donut
                7     7       3462      128           truck
                70   70       3403      152         toaster
                5     5       3117      124             bus
                47   47       2768      119           apple
                4     4       2706      114        airplane
                38   38       2413       94   tennis racket
                39   39       2392      122          bottle
                60   60       2302       52    dining table
                52   52       2204      104         hot dog
                33   33       2200       98            kite
                45   45       1868       86            bowl
                63   63       1658       65          laptop
                71   71       1598       55            sink
                31   31       1530       82       snowboard
                43   43       1414       48           knife
                55   55       1226       33            cake
                6     6       1084       60           train
                48   48       1053       44        sandwich
                69   69        185        7            oven
                5   35        164       11  baseball glove
                """
                if coco_to_yolo_map[annot["category_id"]] in [49, 18, 22, 12, 13]:
                    current_image = cv2.imread(image_paths[key] + image_file_name)  # shape (height, width, channels)
                    # Display the picture with bounding box
                    # visualize(current_image, [annot["bbox"]], [annot["category_id"]], category_id_to_name)
                    yolo_bbox = coco_to_yolo(annot["bbox"], current_image.shape[1], current_image.shape[0])

                    # e.g. 45 0.479492 0.688771 0.955609 0.5955 \n
                    temp = str(coco_to_yolo_map[annot["category_id"]]) + " " + " ".join(map(lambda x: str(x), yolo_bbox)) + "\n"
                    print(temp)
                    save_text(temp, label_paths[key] + label_file_name, "a+")  # Save and Append the line


if __name__ == '__main__':
    parse_annotation_file()
    # generate_labels()

