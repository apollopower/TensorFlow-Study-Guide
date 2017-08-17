# tf.estimator is a high-level TensorFlow library that simplifies the
# mechanics of machine learning, including:
# - running training loops
# - running evaluation loops
# - managing data sets

# Compared to tf_core_demo.py, notice how much simpler the linear
# regression program becomes with tf.estimator:

# ______________________________________________________________________

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

