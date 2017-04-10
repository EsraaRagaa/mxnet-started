import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from vision import mnist_data as mnist_data

'''
Load Data
'''

# Download and read data
path='http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = mnist_data.read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = mnist_data.read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

# Plot the first 10 images and print their labels
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))

# Create data iterators
def to4d(img):
  return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32) / 255


batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

'''
Convolutional Neural Network
'''
data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

shape = {"data" : (batch_size, 1, 28, 28)}
vis = mx.viz.plot_network(symbol=lenet, shape=shape)
vis.render('mnist-cnn')

# Start training
import logging
logging.getLogger().setLevel(logging.DEBUG)

mod = mx.mod.Module(symbol=lenet,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# Train, Predict and Evaluate
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=5,
        batch_end_callback = mx.callback.Speedometer(batch_size, 200)) # output progress for each 200 data batches

# Evaluate the accucracu give an data iterator
valid_acc = mod.score(val_iter, ['acc'])
print 'Validation accuracy: %f%%' % (valid_acc[0][1] *100,)
assert valid_acc[0][1]  > 0.95, "Low validation accuracy."
