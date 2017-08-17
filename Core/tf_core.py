import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Tensors

3 # A Rank 0 Tensor; this is a scalar with shape []
[1.,2.,3.] # A Rank 1 Tensor; this is a vector with shape [3]
[[1.,2.,3.],[4.,5.,6.]] # A Rank 2 Tensor; a matrix with shape [2,3]
[[[1.,2.,3.]],[[4.,5.,6.]]] # A Rank 3 Tensor with shape[2,1,3]

# ____________________________________________________________________

# The Computational Graph

# A computational graph is a series of TensorFlow operations arranged
# into a graph of nodes.

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

# print(node1,node2)

# This only prints the nodes themselves, not OUTPUT their values.
# To evaluate them, a computational graph must run within a Session.

# The following creates a Session Obj and then invokes its run method.

sess = tf.Session()
# print(sess.run([node1,node2]))

# Build more complicated computations by combining Tensor nodes with
# operations (Operations are also nodes).

node3 = tf.add(node1,node2)
# print("node3:",node3)
# print("sess.run(node3):",sess.run(node3))

# As it stands, this greaph is not interesting because it always produces
# a constant result.

# ____________________________________________________________________

# Placeholders

# A graph can be parameterized to accept external inputs, known as
# placeholders. A placeholder is a promised value to be returned later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)

# We can evaluate this graph with multiple input by using the
# feed_dict argument to the run method to feed concrete values to the
# placeholders:

# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1,3], b: [2,4]}))

# We can make the computational graph more complex by adding another
# operation. For example:

add_and_triple = adder_node * 3.
# print(sess.run(add_and_triple, {a: 3,b: 4.5}))

# ____________________________________________________________________

# Variables

# In machine learning, we will typically want a model that can take
# arbitrary inputs.

# Variables allow us to add trainable paramters to a graph. They are
# constructed with a type and initial value:

m = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m * x + b

# To initialize all the variables in a TensorFlow program, you must
# explicitly call a special operation as follows:

init = tf.global_variables_initializer()
sess.run(init)

# It is important to realize init is a handle to the TensorFlow
# sub-graph that initializes all the global variables. Until we call
# sess.run, the variables are unitialized.

# Since x is a placeholder, we can evaluate linear_model for several
# values of x simultaneously as follows:

# print(sess.run(linear_model, {x: [1,2,3,4]}))

# To evaluate the model on a training data, we need a y placeholder to
# provide the desired values, and we need to write a loss function.

# ____________________________________________________________________

# Evaluations and Loss Functions

# A loss function measures how far apart the current model is from the
# provided data.

# This example uses linear regression, which sums the squares of the deltas
# between the current model and the provided data.

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
# print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

# (linear_model - y) creates a vector where each element is the corresponding
# example's error delta. We call (tf.square) to square that error. Then we sum
# all squared errors to create a single scalar that abstracts the error of all
# examples using tf.reduce_sum.

# ____________________________________________________________________

# We could improve this manually by reassigning the valus of m & b to the perfect
# values of -1 and 1. We initialize variables with tf.variable, but can change
# them using operations like (tf.assign).

fixm = tf.assign(m, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixm,fixb])
# print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))

# ____________________________________________________________________

# tf.train API

# TensorFlow provides optimizers that slowly change each variable in order to minimize
# the loss function. The simplest optimizer is Gradient Descent. It modifies
# each variable according tot eh magnitude of the derivative of loss with
# respect to that variable.

# TensorFlow can automatically produce derivatives given only a description
# of the model using the function tf.gradients

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})

print(sess.run([m,b]))