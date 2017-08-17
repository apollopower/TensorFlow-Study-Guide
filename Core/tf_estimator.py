# tf.estimator is a high-level TensorFlow library that simplifies the
# mechanics of machine learning, including:
# - running training loops
# - running evaluation loops
# - managing data sets

# Compared to tf_core_demo.py, notice how much simpler the linear
# regression program becomes with tf.estimator:

# ______________________________________________________________________

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np # Often used to load, manipulate, and preprocess data.

# Declare list of features. We only have one numeric fature. There are many
# types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x",shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we have two data sets: one for training and one for evaluation
# We have to tell the function how many batches of data (num_epochs)
# we want and how big each batch should be.

x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("Train Metrics: %r"% train_metrics)
print("Eval Metrics: %r"% eval_metrics)