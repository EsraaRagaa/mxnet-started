'''
https://github.com/dmlc/mxnet-notebooks/blob/master/python/how_to/predict.ipynb
'''

'''
Download pre-trained models
'''
import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url, filename)
def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)


'''
Initlaization
'''
import mxnet as mx
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-50', 0)

vis = mx.viz.plot_network(sym)
vis.render('resnet')

mod = mx.mod.Module(symbol=sym, context=mx.gpu())
mod.bind(for_training = False,
         data_shapes=[('data', (1,3,224,224))])
mod.set_params(arg_params, aux_params)


'''
Prepare Data
'''
download('http://data.mxnet.io/models/imagenet/resnet/synset.txt')
with open('synset.txt') as f:
    synsets = [l.rstrip() for l in f]

import tarfile
download('http://data.mxnet.io/data/val_1000.tar')
tfile = tarfile.open('val_1000.tar')
tfile.extractall()
with open('val_1000/label') as f:
    val_label = [int(l.split('\t')[0]) for l in f]

import matplotlib
matplotlib.rc("savefig", dpi=100)
import matplotlib.pyplot as plt
import cv2
for i in range(0,8):
    img = cv2.cvtColor(cv2.imread('val_1000/%d.jpg' % (i,)), cv2.COLOR_BGR2RGB)
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.axis('off')
    label = synsets[val_label[i]]
    label = ' '.join(label.split(',')[0].split(' ')[1:])
    plt.title(label)

import numpy as np
import cv2
def get_image(filename):
    img = cv2.imread(filename)  # read image in b,g,r order
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # change to r,g,b order
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (channel, height, width)
    img = img[np.newaxis, :]  # extend to (example, channel, heigth, width)
    return img

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


'''
Predict
'''
img = get_image('val_1000/0.jpg')
mod.forward(Batch([mx.nd.array(img)]))
prob = mod.get_outputs()[0].asnumpy()
y = np.argsort(np.squeeze(prob))[::-1]
print('truth label %d; top-1 predict label %d' % (val_label[0], y[0]))