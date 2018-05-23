我们需要用deeplab 在cityscapes dataset上 train好的model来predict我们自己的data


cityscapes dataset已经在/media/workspace/bgong/data/cityscapes
其中，leftImg8bit是街景照片，gtFine是segmentation label  
可以去cityscapes官网详细了解一下data


关于我们自己的data:
下面说的data都是指pedestrian competition的validation data without label
1. data需要保证统一size, 请确认一下size。请记录下data size以及data的数量
2. 需要将 data convert成tfrecord。convert的代码我会发给你，代码请只修改data directory, 别的地方都不要改
3. 将convert好的tfrecord放在deeplab/datasets/cityscapes/tfrecord 


deeplab代码需要修改的部分：
datasets/cityscapes/cityscapesscripts/preparation/createTrainIdLabelImgs.py comment掉coarse相关代码，因为我们没有用gtCoarse数据
deeplab/datasets/segmentation_dataset.py : 在_CITYSCAPES_INFORMATION里面添加了“test”: data数量
cityscapeScripts/helpers/annotation.py : Comment print type(obj).__name__
deeplab/train.py 和 deeplab/vis.py 都加入限定gpu_id命令



请参考：https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md




