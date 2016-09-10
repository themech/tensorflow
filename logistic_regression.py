"""Simple logistic regression example using Tensorflow.

The same data is used as in binary_logistic_regression example, but the
classification is represented as a vector, so it could be used for 
logistic regression with multiple classes. For N classes just use N-elements
vector, where each positions corresponds to a separate class, for example
Y = [[0, 0, 1], 
     [1, 0, 0]]
means the first example belongs to class #3 while the second - to class #1.
"""

import numpy as np
import pandas
import tensorflow as tf

import show_data	# local module for plotting the data and hypotesis

# Number or training epochs.
numEpochs = 1500

# Load the data from CSV file. It has two feature columns and one classificaion
# column. 
data = pandas.read_csv('data.txt', names=['feature1', 'feature2', 'answer'])

# Split the data into features and class (labels) subsets.
features = data[['feature1', 'feature2']]
# Map the binary class into a vector.
labels = np.asarray([[1., 0.] if v > 0.5 else [0., 1.] \
                     for v in data['answer']])
numFeatures = features.shape[1]
numLabels = labels.shape[1]

# Tensorflow placeholders for the features and labels data.
x = tf.placeholder(tf.float32, shape=[None, numFeatures])
y_ = tf.placeholder(tf.float32, shape=[None, numLabels])

# Weight and bias variables - those are the values that we will try to optimize
# by training the model.
W = tf.Variable(tf.zeros([numFeatures, numLabels]))
b = tf.Variable(tf.zeros([numLabels]))

# Use softmax activation for the hypotesis.
y = tf.nn.softmax(tf.matmul(x , W) + b)

# Cross entropy is used as a cost function.
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Regularized, again, not really needed with only two features.
regularizer = tf.reduce_sum(tf.square(y))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy+regularizer)

# Accuracy. The argmax finds the position of the output with the highest value
# which indicates the predicted class.
correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


s = tf.Session()
s.run(tf.initialize_all_variables())

for i in range(numEpochs):
  s.run(train_step, feed_dict = {x: features, y_: labels})
  c, acc = s.run([cross_entropy, accuracy], 
                  feed_dict = {x: features, y_: labels})
  # Display some console debug data.
  if not i % 10:
    print "Epoch %d, accuracy %.2f%%, current cost: %f." % (i, acc * 100, c)

show_data.plot(features.values, 
               ['b' if v[0] == 1 else 'r' for v in labels], 
               s,
               x,
               tf.argmax(y, 1))

