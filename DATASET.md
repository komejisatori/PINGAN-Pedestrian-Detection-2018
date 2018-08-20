# Pedestrian Detection Project DataSet

## Indroduction

This file documents a large collection of dataset file path on workstation.
Currently, we have 2 kinds of data.

1. WIDER Pedestrian Detection Task Image & Annotation.
2. COCO 2014 Train and Val Image & Annotation.


### Common Settings and Notes

* Dataset File Format Transformation Tools are [here](tools/annotation_tools).
* Image Augmentation Tools are [here](tools/img_process_tools)
* All the Images and Annotations should be at `Work Station 5` @ `/media/workspace/bgong/data/PedestrianDetection`
* All the result backup files are located at `Work Station 5` @ `/media/workspace/bgong/data/PedestrianDetection/Result_BAK`

## WIDER Pedestrian Detection Data

The datasets listed below are Competition Format

|DataSet|Number|Note|File Path|Annotation File|
|---|---|---|---|---|
|WIDER Train|11500|Train Set|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/train|train_annotations.txt|
|WIDER Val|5000|Val Set|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/val|val_annotations.txt|
|WIDER Test|3500|Test Set|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/test_new|-|
|WIDER Dev|1396|Test Set with Image Augmentation|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/dev|dev_annotations.txt|
|WIDER Train Aug|11500|Train Set with Image Augmentation|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/train_aug|train_annotations.txt|
|WIDER Val Aug|5000|Val Set with Image Augmentation|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/val_aug|val_annotations.txt|
|WIDER Dev Aug|1396|Dev Set with Image Augmentation|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/dev_aug|dev_annotations.txt|

The datasets listed below are COCO Format

|DataSet|Number|Note|File Path|Annotation File|
|---|---|---|---|---|
|WIDER Train|11500|Train Set|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/coco/train2014|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/coco/annotations/instances_train2014.json|
|WIDER Val|5000|Val Set|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/coco/val2014|/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/coco/annotations/instances_val2014.json|

### Notes

* All the ignore files are located at `/media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data`. Read more about [Ignore File](https://competitions.codalab.org/competitions/19118)

## COCO Object Detection Data

COCO Data Format:

```
COCO/DIR/
  annotations/
    instances_train2014.json
    instances_val2014.json
    instances_minival2014.json
    instances_valminusminival2014.json
  train2014/
    COCO_train2014_*.jpg
  val2014/
    COCO_val2014_*.jpg
```

|DataSet|Number|Note|File Path|Annotation File|
|---|---|---|---|---|
|COCO Train|11500|Train Set|/media/workspace/bgong/data/PedestrianDetection/COCO/DIR/train2014|/media/workspace/bgong/data/PedestrianDetection/COCO/DIR/annotations|
|COCO Val|5000|Val Set|/media/workspace/bgong/data/PedestrianDetection/COCO/DIR/val2014|/media/workspace/bgong/data/PedestrianDetection/COCO/DIR/annotations|

### Notes

* If you want to use this COCO data as train or evaluation, you could use this path directly: `/media/workspace/bgong/data/PedestrianDetection/COCO/DIR`
