"""
code snippet from http://mxnet.io/tutorials/python/image_io.html
"""

import mxnet as mx

'''
RecordIO

There are mainly three ways of loading image data in MXNet:

[NEW] mx.img.ImageIter: implemented in python, easily customizable, can load from both .rec files and raw image files.
[OLD] mx.io.ImageRecordIter: implemented in backend (C++), less customizable but can be used in all language bindings, load from .rec files
Custom iterator by inheriting mx.io.DataIter
'''
# Download datasets
import os
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

os.system('wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz -P data/')
os.chdir('data')
os.system('tar -xf 101_ObjectCategories.tar.gz')
os.chdir('../')


MXNET_HOME="/home/itrocks/Git/MXNet/mxnet" # change this to your mxnet location
os.system('python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --train-ratio=0.8 --test-ratio=0.2 data/caltech data/101_ObjectCategories'%MXNET_HOME)

os.system("python %s/tools/im2rec.py --num-thread=4 --pass-through=1 data/caltech data/101_ObjectCategories"%MXNET_HOME)

'''
ImageRecordIter
'''

train_data_iter = mx.io.ImageRecordIter(
    path_imgrec="./data/caltech_train.rec", # the target record file
    data_shape=(3, 227, 227), # output data shape. An 227x227 region will be cropped from the original image.
    batch_size=4, # number of samples per batch
    resize=256 # resize the shorter edge to 256 before cropping
    # ... you can add more augumentation options here. use help(mx.io.ImageRecordIter) to see all possible choices
    )
train_data_iter.reset()
batch = train_data_iter.next()
data = batch.data[0]
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
plt.show()