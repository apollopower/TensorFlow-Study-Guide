from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# We want to be able to input any number of MNIST images, each
# flattened into a 784-dimensional vector. We represent this as\
# a 2-D tensor of floating-point numbers, with a shape [None, 784].
# (Here None means that a dimension can be of any length.)
x = tf.placeholder(tf.float32,[None,784])

# Setting up our variables. Since we are going to 'learn' W and b,
# their initial values don't matter, so we create them as Tensors
# full of zeros
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# Implementing our model here...
# We multiply x and W first, then add b. Once finished, we apply
# the softmax model
y = tf.nn.softmax(tf.matmul(x,W) + b)


# Now for the loss function:
# We will use Cross-Entropy

# First add a new placeholder to input the correct answers...
y_ = tf.placeholder(tf.float32, [None, 10])

# ...Now implement cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# Implementing optimization algorithm to minimize cross entropy:
# Using Gradient Descent in this case
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch our model in a interactive session
sess = tf.InteractiveSession()

# Initialize our variables we created:
tf.global_variables_initializer().run()

# Now to train our program. We'll do this 1000 times
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

# Each step of the loop, we get a "batch" of one hundred random data
# points from our training set. We run train_step feeding in the
# batches data to replace the placeholders.

# Using small batches of random data is called "stochastic training":
# Its alot cheaper than analyzing all data points, and has much of
# the same benefit.


# Evaluating our Model

# argmax returns the index of the highest entry in a tensor along some
# axis. tf.argmax(y,1) is the label our model thinks is most likely for
# each input. tf.argmax(y_,1) is the correct label. tf.equal checks if
# the prediction is true
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# What is returned is a list of booleans [True,False,True,True]. To
# determine what fraction is correct, we cast them as floats and then
# take the mean. Ex: [True,False,True,True] become [1.,0.,1.,1.]. The
# mean for that is 0.75

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Finally, we ask for the accuracy on our test data:
print(sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels}))