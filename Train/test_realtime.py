
import cv2
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils import array_tool as at
from PIL import Image
from utils.vis_tool import visdom_bbox
import numpy as np

def test(img):
    img = t.from_numpy(img)[None]
    opt.caffe_pretrain=False # this model was trained from caffe-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    #output the 坐标
    bboxes = at.tonumpy(_bboxes[0])
    print(bboxes)  #输出框的坐标，array格式

    test_img = visdom_bbox(at.tonumpy(img[0]),
                      at.tonumpy(_bboxes[0]),
                      at.tonumpy(_labels[0]).reshape(-1),
                      at.tonumpy(_scores[0]).reshape(-1))
    trainer.vis.img('test_img', test_img)

def process(f,dtype=np.float32, color=True):
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


if __name__ == '__main__':
    #1.加载网络 load network
    opt.env = 'test'
    opt.caffe_pretrain = True
    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load(
        'C:/Users/86188/Desktop/simple-faster-rcnn-pytorch-master/checkpoints/fasterrcnn_04250634_0.6951113529274409')
    print('network loads successs!')

    # 2.Init the camera
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('C:/Users/86188/Desktop/simple-faster-rcnn-pytorch-master/test.mp4')
    cap.set(3, 900)
    cap.set(4, 900)

    while 1:
        ret, frame = cap.read()  # 读取每一帧
        cv2.imshow('摄像头', frame)  # 显示每一帧
        k = cv2.waitKey(1)
        #img = read_image('misc/catdog.jpg')
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  #cv->pil format tranform
        # image.show()
        image = process(image)  #reprocess the img
        test(image)
        if k == 'q':
            break
    cap.release()
    cv2.destroyAllWindows()

