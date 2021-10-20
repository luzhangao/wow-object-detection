# wow-object-detection

## Overview
This repo is designed to detect objects in World of Warcraft with YOLOv5. 

## Test YOLOv5 on COCO2017 sample dataset
[YOLOv5](https://github.com/ultralytics/yolov5) is trained and visualized on a custom COCO2017 sample dataset. 

### Datasets
control.parse_annotations.py and control.sample_dataset.py will sample [COCO2017](https://cocodataset.org/#download) dataset and generate a new dataset for training.
The hierarchy of this new dataset is the same as the [COCO128](https://www.kaggle.com/ultralytics/coco128) dataset. Also, the details of the new dataset is saved in yolov5.data.coco2017.yaml.

### Training
`python train.py --img 640 --batch 16 --epochs 100 --data coco2017.yaml --weights yolov5s.pt --workers 0`
