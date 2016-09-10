"""Helper module for visualizing the trained model.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot(features, labels, tf_session, x, classifier):
  """
  Plot the 2d graph showing the input as scatter data and the model hypotesis
  as the background.

  Args:
    features: Inputa data for features, 2 features per example.
    labels: example colors.
    tf_session: tensorflow session object.
    x: model placeholder for the input data.
    classifier: model tensor for classifing the data, will be used for
        calculating the plot backgrodund color (representing the hypotesis).
  """
  x_pos = [v[0] for v in features]
  y_pos = [v[1] for v in features]
  x_min, x_max = min(x_pos) - 0.5, max(x_pos) + 0.5 
  y_min, y_max = min(y_pos) - 0.5, max(y_pos) + 0.5
  mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, 0.02), 
                               np.arange(y_min, y_max, 0.02))
  pts = np.c_[mesh_x.ravel(), mesh_y.ravel()].tolist()

  mesh_color = tf_session.run(classifier, feed_dict = {x: pts})
  mesh_color = np.array(mesh_color).reshape(mesh_x.shape)
  plt.pcolormesh(mesh_x, mesh_y, mesh_color)
  plt.scatter(x_pos, y_pos, c=labels, edgecolor='k', s=50)
  plt.show()

