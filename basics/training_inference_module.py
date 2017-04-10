import mxnet as mx
from util.data_iter import SyntheticData
import logging
# set logging level to INFO
logging.basicConfig(level=logging.INFO)

'''
Basic Usage
'''
# simple multi-layer perceptron for 10 classes
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=10)
net = mx.sym.SoftmaxOutput(net, name='softmax')
# synthetic 10 classes dataset with 128 dimension
data = SyntheticData(10, 128)
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



# Train, Predict and Evaluate
batch_size=32
mod.fit(data.get_iter(batch_size),
        eval_data=data.get_iter(batch_size),
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=5)

y = mod.predict(data.get_iter(batch_size))
print('shape of predict: %s' % (y.shape,))

for preds, i_batch, batch in mod.iter_predict(data.get_iter(batch_size)):
    pred_label = preds[0].asnumpy().argmax(axis=1)
    label = batch.label[0].asnumpy().astype('int32')
    print('batch %d, accuracy %f' % (i_batch, float(sum(pred_label==label))/len(label)))

print(mod.score(data.get_iter(batch_size), ['mse', 'acc']))


# Save and Load
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)

mod = mx.mod.Module(symbol=net)
mod.fit(data.get_iter(batch_size), num_epoch=5, epoch_end_callback=checkpoint)

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
print(sym.tojson() == net.tojson())

# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)

# resume training from a saved checkpoint
mod = mx.mod.Module(symbol=sym)
mod.fit(data.get_iter(batch_size),
        num_epoch=5,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=3)


'''
Mudel as a computation machine

A module has several states:
- Initial state. Memory is not allocated yet, not ready for computation yet.
- Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated, ready for computation.
- Parameter initialized. For modules with parameters, doing computation before initializing the parameters might result in undefined outputs.
- Optimizer installed. An optimizer can be installed to a module. After this, the parameters of the module can be updated according to the optimizer after gradients are computed (forward-backward).
'''

# simpliedfied fit() implementation
mod = mx.mod.Module(symbol=net)

# bind, tell the module the data and label shapes, so
# that memory could be allocated on the devices for computation
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