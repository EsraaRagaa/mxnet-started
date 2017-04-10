import mxnet as mx
import numpy as np

'''
Array Creation
'''

# create a 1-dimensional array with a python list
a = mx.nd.array([1, 2, 3])
print(a)
print(a.shape)
print(a.dtype)
print(a.asnumpy())

# create a 2-dimensional array with a nested python list
b = mx.nd.array([[1,2,3], [2,3,4]])
print(b)
print(b.shape)
print(a.dtype)
print(b.asnumpy())

# specify the element type with dtype
a = mx.nd.array([1, 2, 3], dtype=np.int32)
print(a.dtype)

# special initializers
a = mx.nd.zeros((2,3))
print(a.shape)
print(a.asnumpy())

a = mx.nd.ones((2,3))
print(a.shape)
print(a.asnumpy())

a = mx.nd.full((2,3), 7)
print(a.shape)
print(a.asnumpy())


'''
Basic Operations
'''
a = mx.nd.ones((2,3))
b = mx.nd.full((2,3), 5)
# element-wise operation
c = a + b
print(c.asnumpy())

c = a - b
print(c.asnumpy())

'''
Indexing and Slicing
'''
a = mx.nd.array(np.arange(6).reshape(3, 2))
print(a.asnumpy())
print(a.shape)
print(a[1:2].asnumpy())

# slice with particular axis
d = mx.nd.slice_axis(a, axis=0, begin=1, end=2)
print(d.asnumpy())
# using another axis
d = mx.nd.slice_axis(a, axis=1, begin=1, end=2)
print(d.asnumpy())


'''
# Shape Manipulation
'''
# reshape
a = mx.nd.array(np.arange(24))
b = a.reshape((2, 3, 4))
print(b.asnumpy())

# concatenate
a = mx.nd.ones((2, 3))
b = mx.nd.ones((2, 3)) * 2
print(a.asnumpy())
print(b.asnumpy())
c = mx.nd.concatenate([a, b])
print(c.asnumpy())

# concatenate with a particular axis
c = mx.nd.concatenate([a, b], axis=1)
print(c.asnumpy())


'''
Reduce
'''
a = mx.nd.ones((2, 3))
b = mx.nd.sum(a)
print(b.asnumpy())

# reduce along a particular axis
c = mx.nd.sum_axis(a, axis=0)
print(c.asnumpy())
c = mx.nd.sum_axis(a, axis=1)
print(c.asnumpy())


'''
GPU support
'''
def f():
  a = mx.nd.ones((100, 100))
  b = mx.nd.ones((100, 100))
  c = a + b
  print(c)

# in default mx.cpu() is used
f()

# change the default context to the first GPU
with mx.Context(mx.gpu()):
  f()