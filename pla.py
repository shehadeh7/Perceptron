# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:29:07 2021

@author: Mahmoud
"""

import numpy as np
import matplotlib.pyplot as plt

def get_data_in_file(filename):
  """ 
  Read the labels and data points 
  """
  with open(filename, 'r') as f:
      data = []
      # read the data line by line
      for line in f: 
          data.append([int(x) for x in line.split()]) 
          
  # store the data points in x and the labels in y        
  data_array = np.array(data)     
  y = data_array[:,0]   # labels
  x = data_array[:,1:3] # data points
  
  return (x, y)

if __name__ == "__main__":

  # PLA learning algorithm
  # assume 2d graph with points
  (data_x, data_y) = get_data_in_file("points.txt")
  bias = np.ones((np.shape(data_x)[0], 1))
  X = np.append(bias, data_x, axis=1)
  w = np.zeros(np.shape(X)[1])
  y = data_y

  num_epochs = 5000
  l_rate = 0.01
  num_samples = np.shape(y)[0]

  # Optimal weight and associated error
  w_opt = np.zeros(np.shape(X)[1])
  opt_error = num_samples

  # iterate over each sample when calculating error and updating weight
  # Run for a fixed number of iterations and keep track of best weight
  for epoch in range(0, num_epochs):
    w_error = 0
    for i in range(0, num_samples):
      vec = X[i, :]
      y_hat = np.dot(w, vec)
      if np.sign(y_hat) != y[i]:
        w = w + l_rate * y[i] * vec
        w_error += 1  
    if (opt_error > w_error):
      opt_error = w_error
      w_opt = w
      
      
  # plot the data points with labels
  c = [(("black", 'o') if (row==1) else ("red", 'x')) for row in y]
  c_list = [x[0] for x in c]
  m_list = [x[1] for x in c]  
  for i in range(0, num_samples):
   plt.scatter(data_x[i, 0], data_x[i, 1], c=c_list[i], marker=m_list[i])
   
   
  # plot the line of separation found
  # 0 = w0 + w1x1 + w2x2
  # edge case check for vline or hline
  print(w_opt)
  if (w_opt[2] == 0):
      plt.axvline(x=-w_opt[0]/w_opt[1])
  elif (w_opt[1] == 0):
      plt.axhline(y=-w_opt[0]/w_opt[2])
  else:
      x1_max = np.amax(data_x[:, 0])
      x1_min = np.amin(data_x[:, 0])
      x1 = np.linspace(x1_min - 0.5, x1_max + 0.5, 100)
      x2 = -(w_opt[0] + w_opt[1] * x1) / w_opt[2]
      plt.plot(x1, x2, label="Decision Boundary")      

  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.title("Perceptron Learning Algorithm in 2D")
  plt.grid()
  plt.legend()
  plt.show()