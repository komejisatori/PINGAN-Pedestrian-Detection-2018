# Pedestrian Detection

Pedestrain Detection Project for PAII.

## Introduction

For now, we implemented Pedestrian Detection with `Detectron` and `Tensorpack`. They currently use different dependency.
We recommend to use `docker` to create a container for `Detectron`, and you could run `Tensorpack` directly with `Work Station 205`.

For Data, we prepare 2 kinds of data. First one is the [WIDER Challenge](http://wider-challenge.org/) Pedestrian Detection Task.
The Second one is the [COCO](http://cocodataset.org/#download) data with 2014 Train, Val data. Also, we provide coco data format version
of WIDER data in order to run `tensorpack` easily. Please read [Data Page](DATASET.md)

For Model, we provide `Detectron` and `Tensorpack` pre-trained backbone model and trained model. You could check it on 
the `workstation 205`. Please read [Model Page](MODEL.md).

For Running the code, please follow the instructions from [Get Start Part](https://github.com/PingAnIntelligence/pedestrian-detection#get-start) and [GETTING STARTED Page](GETTING_STARTED.md).

For more details about two framework, please refer [Detectron README](https://github.com/PacteraKun/Detectron) 
and [Tensorpack README](https://github.com/PacteraKun/tensorpack/). 

## Get Start

### First Step

```bash
git clone https://github.com/PingAnIntelligence/pedestrian-detection.git
cd pedestrain-detection
git submodule update --init --recursive --remote
cd model/Detectron
git checkout master
cd ../tensorpack
git checkout master
cd ../..
```

### Install Package

**Notes**: 
* Except `Detectron`, all the commands listed below are at `python3` environment.
* In model `Detectron`, we will use `Docker` to create container for `python2` environment.

```bash
sudo pip3 install -r requirements.txt
```

### Install Tensorpack

```bash
cd model/tensorpack # Make sure that you are at master branch.
pip install tensorpack --no-index --find-links ./ # Install this as a python package.
```

### Start Detectron Contrainer

```bash
cd model/Detectron/docker
nvidia-docker build -t pingan/detectron-baseline:v1.0 . # You will get an image id
nvidia-docker run -itd $(image-id) # You will get a container id
nvidia-docker exec -it $(container-id) bash # You will enter into container

# Then in the container
git clone https://github.com/PingAnIntelligence/detectron.git /detectron
pip install -r /detectron/requirements.txt
git clone https://github.com/cocodataset/cocoapi.git /cocoapi
cd /cocoapi/PythonAPI/
make installcd /detectron
make
make ops

cd /detectron
mkdir data
mkdir weights

apt update
apt install wget
apt install vim

# Till now you have successful install Detectron and cocoapi
# Then you go to outside of container
# Copy the data & weights
nvidia-docker cp /media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challengecoco/ $(contrainerid):/detectron/detectron/datasets/data/
nvidia-docker cp -r /media/workspace/bgong/data/PedestrianDetection/WIDER_Pedestrian_Challenge/data/val $(contrainerid):/detectron/data/
nvidia-docker cp /media/workspace/bgong/data/PedestrianDetection/Detectron_Model/Trained/Detectron_Cascade_RCNN_X101_64x4d_Train/model_final.pkl $(contrainerid):/detectron/weights/model_cascade_final.pkl
```

Furthermore, about how to run the two models as training and testing, 
please refer [Detectron README](https://github.com/PingAnIntelligence/detectron/blob/master/README.md) 
and [Tensorpack README`](https://github.com/PingAnIntelligence/tensorpack/blob/master/README.md).

## Result

All the results listed below are `mAP` of `val` dataset in `WIDER Pedestrian Detection Task`

|Framework|Method|Backbone|mAP|
|---|---|---|---|
|Yolo| Pre-trained Model|DarkNet| 0.2008|
|Detectron|Faster RCNN + Pre-trained Model|ResNet 101|0.3273|
|Detectron|Faster RCNN + Trained Model|ResNet 101|0.5370|
|Detectron|Faster RCNN + Trained Model|ResNeXt 101|0.5554|
|Detectron|Faster RCNN + Trained Model|ResNeXt 152|0.5728|
|Detectron|Faster RCNN + Trained Model|ResNeXt 152 + Parameter Tuning (Multiscale Test)|0.5957|
|Tensorpack|Faster RCNN + Trained Model|ResNet 101|0.5147|
|Tensorpack|Cascade RCNN + Trained Model|ResNet 101 + Stage 1st|0.5289|
|Tensorpack|Cascade RCNN + Trained Model|ResNet 101 + Stage 2nd|0.5413|
|Tensorpack|Cascade RCNN + Trained Model|ResNet 101 + Stage 3rd|0.5480|
|Tensorpack|Cascade RCNN + Trained Model|ResNet 101 + Stage 4th|0.5369|


## Author

[Jiaqi Cai](https://github.com/caiPactera), 
[Kun Li](https://github.com/PacteraKun), 
[Zhuoran Wu](https://github.com/PacteraOliver)

{jiaqi.cai22, kun.li24,  zhuoran.wu}@pactera.com

[Gong Bo](https://github.com/PATPAL), 
[Rueisung Lin](https://github.com/rueisung0), 
[Qi Chen](https://github.com/qichen20), 
[Xi Yao](https://github.com/yaooxii)

{bo.gong, rueisung.lin, qi.chen20, xi.yao}@pactera.com

## Citing PingAn Pedestrian Detection

```
@misc{PedestrianDetection2018,
  author =       {Jiaqi Cai, Kun Li, Zhuoran Wu, Gong Bo, Rueisung Lin, Qi Chen, Xi Yao},
  title =        {Pedestrian Detection},
  howpublished = {\url{https://github.com/PingAnIntelligence/pedestrian-detection}},
  year =         {2018}
}
```
