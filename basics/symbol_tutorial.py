"""
code snippet from http://mxnet.io/tutorials/python/symbol.html
"""

import mxnet as mx

'''
Symobol Composition
'''
# Configure computation graph
a = mx.sym.Variable('a')
b = a * 2 + 1

# Plot computation graph
vis = mx.viz.plot_network(symbol=b)
vis.render('graph')

# Bind data
ex = b.bind(ctx=mx.cpu(), args={'a': mx.nd.ones((2, 3))})
# Execute computation graph
ex.forward()
print(ex.outputs)
print(len(ex.outputs))
print(ex.outputs[0].asnumpy())

# Basic Neural Networks
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(data=net, name='out')
vis = mx.viz.plot_network(net, shape={'data': (100, 200)})
vis.render('basic')

# Modulelized Construction for Deep Networks
def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
  conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                               name='conv_%s%s' % (name, suffix))
  bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' % (name, suffix))
  act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' % (name, suffix))
  return act


prev = mx.symbol.Variable(name="Previos Output")
conv_comp1 = ConvFactory(data=prev, num_filter=64, kernel=(7, 7), stride=(2, 2), name='Conv1')
conv_comp2 = ConvFactory(data=conv_comp1, num_filter=32, kernel=(7, 7), stride=(2, 2), name='Conv2')
shape = {"Previos Output": (128, 3, 28, 28)}
vis = mx.viz.plot_network(symbol=conv_comp2, shape=shape)
vis.render('conv-factory')


# Load and save computation graph
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b

c.save('symbol-b.json')
c2 = mx.symbol.load('symbol-b.json')
print(c.tojson() == c2.tojson())