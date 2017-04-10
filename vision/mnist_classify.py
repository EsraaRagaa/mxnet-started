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
Multilayer Perceptron
'''
# Create a place holder variable for the input data
data = mx.sym.Variable('data')
# Flatten the data from 4-D shape (batch_size, num_channel, width, height)
# into 2-D (batch_size, num_channel*width*height)
data = mx.sym.Flatten(data=data)

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
shape = {"data" : (batch_size, 1, 28, 28)}
vis = mx.viz.plot_network(symbol=mlp, shape=shape)
vis.render('mnist-mlp')

# Start training
import logging
logging.getLogger().setLevel(logging.DEBUG)

mod = mx.mod.Module(symbol=mlp,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

# Train, Predict and Evaluate
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=6,
        batch_end_callback = mx.callback.Speedometer(batch_size, 200)) # output progress for each 200 data batches

# Evaluate the accucracu give an data iterator
valid_acc = mod.score(val_iter, ['acc'])
print 'Validation accuracy: %f%%' % (valid_acc[0][1] *100,)
assert valid_acc[0][1]  > 0.95, "Low validation accuracy."
