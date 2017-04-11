"""
code snippet from http://mxnet.io/tutorials/python/data.html
"""

import mxnet as mx

# Symbol and Data Variables
# In MXNet, an operator (mx.sym.*) has one or more input variables and output variables;
# some operators may have additional auxiliary variables for internal states.
# For an input variable of an operator,
# if do not assign it with an output of another operator during creating this operator,
# then this input variable is free.
# We need to assign it with external data before running
import mxnet as mx
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=64)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='softmax')
print(net.list_arguments())
print(net.list_outputs())

'''
Basic Data Iterator
'''
# Data Batch
class SimpleBatch(object):
  def __init__(self, data, label, pad=0):
    self.data = data
    self.label = label
    self.pad = pad


'''
Data Iterators

1. return a data batch or raise a StopIteration exception if reaching the end when call next() in python 2 or __next()__ in python 3
2. has reset() method to restart reading from the beginning
3. has provide_data and provide_label attributes,
   the former returns a list of (str, tuple) pairs, each pair stores an input data variable name and its shape.
   It is similar for provide_label,
'''

import numpy as np
class SimpleIter:
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0
    def __iter__(self):
        return self
    def reset(self):
        self.cur_batch = 0
    def __next__(self):
        return self.next()
    @property
    def provide_data(self):
        return self._provide_data
    @property
    def provide_label(self):
        return self._provide_label
    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            assert len(data) > 0, "Empty batch data."
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration



import logging
logging.basicConfig(level=logging.INFO)

n = 32
data = SimpleIter(['data'], [(n, 100)],
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['softmax_label'], [(n,)],
                  [lambda s: np.random.randint(0, num_classes, s)])

mod = mx.mod.Module(symbol=net)
mod.fit(data, num_epoch=5)