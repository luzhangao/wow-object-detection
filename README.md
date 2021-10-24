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

### Evaluating
[wandb dashboard](https://wandb.ai/luzhangao/YOLOv5?workspace=user-luzhangao)

### Tips for Windows 10
Although the code can run perfectly in cloud servers, it has some problems when testing on Windows 10.
>- The default hyperparameter of workers is 8. However, it must be 0 or 1 on windows 10. 
`python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --workers 0`
>- The training loss is always be nan if the model is trained with gpu on window 10 and the version of cuDNN is lower than 8.2.2. 
Check the issue [here](https://issueexplorer.com/issue/ultralytics/yolov5/4839). To solve this, we can either train with cpu or upgrade cuDNN.
However, the latest version of cuDNN offered by Anaconda is 8.2.1 until Oct 19, 2021. So it can't be upgraded by `conda install cudnn`.
```python
# check the package versions with Python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
```

### TBD
>- Labelled WOW screenshots
