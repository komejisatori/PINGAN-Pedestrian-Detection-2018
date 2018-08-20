# Pedestrian Detection Project Model & Baselines

## Introduction

This file documents a large collection of baselines and models files path on workstation.

### Common Settings and Notes

* All the models were run on PingAnII `Work Station` with 4 NVIDIA 1080Ti GPU.
* PreTrained Models were trained on the union of `coco_2014_train` and `coco_2014_valminusminival`, 
which is exactly equivalent to the recently defined `coco_2017_train dataset`.
* Trained Models were trained on the WIDER Pedestrian Task `Train` Data.
* * All the Models should be at `Work Station 5` @ `/media/workspace/bgong/data/PedestrianDetection`

## ImageNet PreTrained Backbone Models

### Detectron

|Model Name|Notes|File Path|
|---|---|---|
|`R-101.pkl`|converted copy of MSRA's original ResNet-101 model|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/BackBone|
|`X-101-32x8d.pkl`|ResNeXt-101-32x8d model trained with Caffe2 at FaceBook|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/BackBone|
|`X-101-64x4d.pkl`|converted copy of FB's original ResNeXt-101-64x4d model trained with Torch7|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/BackBone|
|`X-152-32x8d-IN5k.pkl`|ResNeXt-152-32x8d model **trained on ImageNet-5k** with Caffe2 at FaceBook|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/BackBone|

## End-to-End Pre-Trained Faster R-CNN Baselines

### Detectron

|BackBone|Model Name|From|File Path|
|---|---|---|---|
|`X-101-32x8d-FPN`|model_faster_rcnn_X-101-32x8d-FPN_final.pkl|[Url](https://s3-us-west-2.amazonaws.com/detectron/36761786/12_2017_baselines/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml.06_33_22.VqFNuxk6/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/PreTrained|
|`X-101-64x4d-FPN`|model_faster_rcnn_X-101-64x4d-FPN_final.pkl|[Url](https://s3-us-west-2.amazonaws.com/detectron/35858015/12_2017_baselines/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml.01_40_54.1xc565DE/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/PreTrained|
|`X-152-32x8d-FPN-IN5k`|model_mask_rcnn_X-152-32x8d-FPN-IN5k_final.pkl|[Url](https://s3-us-west-2.amazonaws.com/detectron/37129812/12_2017_baselines/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x.yaml.09_35_36.8pzTQKYK/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/PreTrained|

### TensorPack

|Model Name|From|File Path|
|---|---|---|
|`COCO-R101C4-MaskRCNN-Standard.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`COCO-R101FPN-MaskRCNN-Standard.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`COCO-R50C4-MaskRCNN-Standard.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`COCO-R50FPN-MaskRCNN-Standard.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`COCO-R50FPN-MaskRCNN-StandardGN.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`ImageNet-R101-AlignPadding.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`ImageNet-R50-AlignPadding.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|
|`ImageNet-R50-GroupNorm32-AlignPadding.npz`|[Url](http://models.tensorpack.com/FasterRCNN)|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/PreTrained|


## End-to-End Faster R-CNN Baselines

### Detectron

|BackBone|Type|Train Dataset|Iter Epoch|File Path|
|---|---|---|---|---|
|`X-152-32x8d-FPN-IN5k`|Faster RCNN|WIDER Train + Val|260000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X152_Train_Val|
|`X-152-32x8d-FPN-IN5k`|Faster RCNN|WIDER Train|210000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X152_Train_FPN|
|`X-152-32x8d-FPN-IN5k`|Faster RCNN|WIDER Train|210000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X152_Train_No_FPN|
|`X-152-32x8d-FPN-IN5k`|Faster RCNN|WIDER Train|75000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X152_Train_Focal_Loss|
|`X-101-64x4d-FPN`|Faster RCNN|WIDER Train|300000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X101_Train_Anchor|
|`X-101-64x4d-FPN`|Faster RCNN|WIDER Train|210000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X101_64x4d_FPN_Dilation|
|`X-152-32x8d-FPN-IN5k`|Faster RCNN|WIDER Train|90000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Faster_RCNN_X152_32x8d_FPN_Train_Image_Aug|

### Notes

* In Folder `/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Test_Model`, there are a few models we trained before with `WIDER Train`. At that time, in order to save the disk space, we only save the model with the best performance.

### TensorPack

|BackBone|Type|Train Dataset|Iter Epoch|File Path|
|---|---|---|---|---|
|`ImageNet-R101-AlignPadding.npz`|Faster RCNN|WIDER Train|360000|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/Trained/TensorPack_Faster_R101_FPN|

## End-to-End Cascade R-CNN Baselines

### Detectron

|BackBone|Type|Train Dataset|Iter Epoch|File Path|
|---|---|---|---|---|
|`X-101-64x4d.pkl`|Cascade RCNN|WIDER Train|70000|/media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Cascade_RCNN_X101_64x4d_Train|

### TensorPack

|BackBone|Type|Train Dataset|Iter Epoch|File Path|
|---|---|---|---|---|
|`ImageNet-R101-AlignPadding.npz`|Cascade RCNN|WIDER Train|230000|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/Trained/TensorPack_Cascade_R101_FPN_train_ignore|
|`ImageNet-R101-AlignPadding.npz`|Cascade RCNN|WIDER Train|230000|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/Trained/TensorPack_Cascade_R101_FPN_no_train_ignore|
|`ImageNet-R101-AlignPadding.npz`|Cascade RCNN|WIDER Train|260000|/media/workspace/bgong/data/PedestrianDetection/TensorPack_Model/Trained/TensorPack_Cascade_R101_FPN_train_ignore_same_thresh_.5|