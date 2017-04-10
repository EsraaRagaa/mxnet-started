import mxnet as mx

'''
Training a Multi-layer Pereceptron

Use the imperative NDArray and symbolic Symbol together to implement a complete training algorithm
'''

# Example MLP
num_classes = 10
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=num_classes)
net = mx.sym.SoftmaxOutput(data=net, name='out')
vis = mx.viz.plot_network(net)
vis.render('mlp')
print(net.list_arguments())

# bind symbols to executor, allocate all the ndarrays need
num_features = 100
batch_size = 100
ex = net.simple_bind(ctx=mx.cpu(), data=(batch_size, num_features))
print(ex.arg_arrays)
print(zip(net.list_arguments(), ex.arg_arrays))
print(dict(zip(net.list_arguments(), ex.arg_arrays)))
args = dict(zip(net.list_arguments(), ex.arg_arrays))
for name in args:
    print(name, args[name].shape)

# intialize allocated ndarrays
for name in args:
    data = args[name]
    if 'weight' in name:
        data[:] = mx.random.uniform(-0.1, 0.1, data.shape)
    if 'bias' in name:
        data[:] = 0


# prepare dataests
import numpy as np
import matplotlib.pyplot as plt


class ToyData:
    def __init__(self, num_classes, num_features):
        self.num_classes = num_classes
        self.num_features = num_features
        self.mu = np.random.rand(num_classes, num_features)
        self.sigma = np.ones((num_classes, num_features)) * 0.1

    def get(self, num_samples):
        num_cls_samples = num_samples / self.num_classes
        x = np.zeros((num_samples, self.num_features))
        y = np.zeros((num_samples,))
        for i in range(self.num_classes):
            cls_samples = np.random.normal(self.mu[i, :], self.sigma[i, :], (num_cls_samples, self.num_features))
            x[i * num_cls_samples:(i + 1) * num_cls_samples] = cls_samples
            y[i * num_cls_samples:(i + 1) * num_cls_samples] = i
        return x, y

    def plot(self, x, y):
        colors = ['r', 'b', 'g', 'c', 'y']
        for i in range(self.num_classes):
            cls_x = x[y == i]
            plt.scatter(cls_x[:, 0], cls_x[:, 1], color=colors[i % 5], s=1)
        plt.show()


toy_data = ToyData(num_classes, num_features)
x, y = toy_data.get(1000)
print(x.shape)
print(y.shape)
toy_data.plot(x, y)


# training with mini-batch sgd
learning_rate = 0.1
final_acc = 0
for i in range(100):
    x, y = toy_data.get(batch_size)
    args['data'][:] = x
    args['out_label'][:] = y
    ex.forward(is_train=True)
    ex.backward()
    for weight, grad in zip(ex.arg_arrays, ex.grad_arrays):
        weight[:] -= learning_rate * (grad / batch_size)
    if i % 10 == 0:
        acc = (mx.nd.argmax_channel(ex.outputs[0]).asnumpy() == y).sum()
        final_acc = acc
        print('iteration %d, accuracy %f' % (i, float(acc)/y.shape[0]))
assert final_acc > 0.95, "Low training accuracy."