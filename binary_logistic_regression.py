"""Simple binary logistic regression using Tensorflow.

The y vector predicted by the model contains only a single number for each
example row - it is either '0' or '1'.
Additionally this example shows how to write summary data that can be analyzed
later.
"""

import numpy as np
import pandas
import tensorflow as tf

import show_data	# local module for plotting the data and hypotesis

# Number or training epochs.
numEpochs = 800

# Load the data from CSV file. It has two feature columns and one classificaion
# column. 
data = pandas.read_csv('data.txt', names=['feature1', 'feature2', 'answer'])

# Split the data into features and class (label) subsets.
features = data[['feature1', 'feature2']]
labels = data[['answer']]
numFeatures = features.shape[1]

# Tensorflow placeholders for the features and labels data.
x = tf.placeholder(tf.float32, shape=[None, numFeatures])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Weight and bias variables - those are the values that we will try to optimize
# by training the model.
W = tf.Variable(tf.zeros([numFeatures, 1]))
b = tf.Variable(tf.zeros([1, 1]))

# Sigmoid is used for the hypotesis - h(x) = x * W + b
y = tf.nn.sigmoid(tf.matmul(x , W) + b)

# COST FUNCTION
# In general when using gradient descent for binary classification, it is
# recommended to use the following cost function:
# cost(h, (x), y) = -ylog( h(x) ) - (1-y)log( 1- h(x) )
# This is what is used by default in this example, although the simple
# squared error function commented out below works as well. 
cost = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-9, 1)) + \
                      (1 - y_)*tf.log(tf.clip_by_value(1 - y, 1e-9, 1)))
#cost = tf.nn.l2_loss(y - y_, name="squared_error_cost")

# Optional regularizer, not really needed in this example, as we try to fit a
# linear 2D funation to the 2-D data, so we don't risk overfitting. It may be
# handy when using more features.
regularizer = tf.reduce_sum(tf.square(W))

# ACCURACY.
# We map the sigmoid output to either 0 or 1 and compare it with the golden
# set data.
correct_predict = tf.equal(tf.cast(tf.greater(y, 0.5), tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

# As an alternative, we could use try to use the difference between sigmoid
# output and the correct answer as our accuracy metric.
#accuracy = 1.0 - tf.reduce_mean(tf.square(y_ - y))

# In tensorflow AdamOptimizer works much better than GradientDescent.
train_step = tf.train.AdamOptimizer(0.01).minimize(cost + regularizer)

# INITIALIZE our weights and biases
init_OP = tf.initialize_all_variables()

# Create a tensorflow session
s = tf.Session()
s.run(tf.initialize_all_variables())

# SUMMARIES.
# Those will be stored in a subdirectory and can be displayed and analyzed
# later using tensorboard.
cost_summary = tf.scalar_summary("cost", cost)
regularizer_summary = tf.scalar_summary("regularizer", regularizer)

# Summary ops to check how variables (W, b) are updating after each iteration.
weightSummary = tf.histogram_summary("weights", W)
biasSummary = tf.histogram_summary("biases", b)

accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# Merge all summaries.
all_summary = tf.merge_all_summaries()

writer = tf.train.SummaryWriter("summary_logs", s.graph)

# Training loop.
for i in range(numEpochs):
  s.run(train_step, feed_dict = {x: features, y_: labels})
  summary_results, c, acc= s.run([all_summary, cost,accuracy], \
                                 feed_dict = {x: features, y_: labels})

  writer.add_summary(summary_results, i)

  # Display some console debug data.
  if not i % 10:
    print "Epoch %d, accuracy %.2f%%, current cost: %f." % (i, acc * 100, c)
    print "\tCurrent weights", str(W.eval(s)).replace('\n', ' ')
    print "\tCurrent bias", b.eval(s)

# Plot the final prediction model.
show_data.plot(features.values, 
               ['r' if v == 1 else 'b' for v in labels.values],
               s,
               x,
               tf.greater(y, 0.5))

