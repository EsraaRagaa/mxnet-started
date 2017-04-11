"""
code snippet from http://mxnet.io/tutorials/python/module.html
"""

import mxnet as mx
from util.data_iter import SyntheticData
import logging
# set logging level to INFO
logging.basicConfig(level=logging.INFO)

# synthetic 10 classes dataset with 128 dimension
data = SyntheticData(10, 128)

# simple multi-layer perceptron for 10 classes
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(net, name='softmax')


vis = mx.viz.plot_network(net)
vis.render('mlp')

'''
Mudle as a computation machine

A module has several states:
- Initial state. Memory is not allocated yet, not ready for computation yet.
- Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated, ready for computation.
- Parameter initialized. For modules with parameters, doing computation before initializing the parameters might result in undefined outputs.
- Optimizer installed. An optimizer can be installed to a module. After this, the parameters of the module can be updated according to the optimizer after gradients are computed (forward-backward).
'''
# initial state
mod = mx.mod.Module(symbol=net)

# bind, tell the module the data and label shapes, so
# that memory could be allocated on the devices for computation
batch_size=32
train_iter = data.get_iter(batch_size)
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

# init parameters
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

# init optimizer
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))

# use accuracy as the metric
metric = mx.metric.create('acc')

# train one epoch, i.e. going over the data iter one pass
for batch in train_iter:
  mod.forward(batch, is_train=True)  # compute predictions
  mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
  mod.backward()  # compute gradients
  mod.update()  # update parameters using SGD

# training accuracy
print(metric.get())