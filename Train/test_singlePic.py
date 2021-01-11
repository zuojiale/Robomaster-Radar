import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from utils.vis_tool import visdom_bbox

def test(**kwargs):
    opt._parse(kwargs)
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('C:/Users/86188/Desktop/simple-faster-rcnn-pytorch-master/checkpoints/fasterrcnn_08042317_0.9090909090909093')
    print('load successs!')
    img = read_image('test_img/test.jpg')
    img = t.from_numpy(img)[None]
    opt.caffe_pretrain=False # this model was trained from caffe-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    test_img = visdom_bbox(at.tonumpy(img[0]),
                      at.tonumpy(_bboxes[0]),
                      at.tonumpy(_labels[0]).reshape(-1),
                      at.tonumpy(_scores[0]).reshape(-1))
    trainer.vis.img('test_img', test_img)

if __name__ == '__main__':
    import fire

    fire.Fire()