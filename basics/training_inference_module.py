"""
code snippet from http://mxnet.io/tutorials/python/module.html
"""

import mxnet as mx
from util.data_iter import SyntheticData
import logging
# set logging level to INFO
logging.basicConfig(level=logging.INFO)

'''
Basic Usage
'''
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

# Create Module
# symbol : the network Symbol
# context : the device (or a list of devices) for execution
# data_names : the list of data variable names
# label_names : the list of label variable names
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])



# Train
batch_size=32
mod.fit(data.get_iter(batch_size),
        eval_data=data.get_iter(batch_size),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=5)

# Predcit
y = mod.predict(data.get_iter(batch_size))
print(y.shape)
print(y[0].asnumpy())

# Forward
data.get_iter(batch_size).reset()

mod.forward(data.get_iter(batch_size).next())
z = mod.get_outputs()[0]
print(z.shape)
print(z[0].asnumpy())


# Predcit iteratively
for preds, i_batch, batch in mod.iter_predict(data.get_iter(batch_size)):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label==label))/len(label)))

print(mod.score(data.get_iter(batch_size), ['mse', 'acc']))


# Save checkpoints of model
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(data.get_iter(batch_size), num_epoch=5, epoch_end_callback=checkpoint)

# Load checkpoit at epoch 3rd
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
print(sym.tojson() == net.tojson())

# Resume training from a saved checkpoint by assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)

mod = mx.mod.Module(symbol=sym)
mod.fit(data.get_iter(batch_size),
        num_epoch=5,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)
