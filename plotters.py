#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:37:27 2019

@author: shojaee
"""
def plot_scatter_colored(X, location, title=None, ax=None):
  import matplotlib.pyplot as plt
  import matplotlib
  import numpy as np
  if ax is None:
    fig, ax = plt.subplots()
  cmap = matplotlib.cm.get_cmap('jet')
  if len(X.shape) > 1:
    X = X.reshape(X.shape[0],)
  idx = np.isnan(X)
  X = X[~idx]
  location = location[~idx,:]  
  normalize = matplotlib.colors.Normalize(vmin=np.min(X), vmax=np.max(X))
  colors = [cmap(normalize(value)) for value in X]
  x, y = location.T
  cf = ax.scatter(x, y, color=colors)
  cax, _ = matplotlib.colorbar.make_axes(ax)
  cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
  if title is not None:
      ax.set_title(title)
  
def plot_cmp(X1, loc1, X2, loc2, title=None, scale = False):
  import matplotlib.pyplot as plt
  from sklearn import preprocessing


  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
  if scale:
    X1 = preprocessing.scale(X1)
    X2 = preprocessing.scale(X2)
  plot_scatter_colored(X1.reshape(loc1.shape[0],), loc1, ax=ax1)
  plot_scatter_colored(X2.reshape(loc2.shape[0],), loc2, ax=ax2)
  if title is not None:
      fig.suptitle(title)

      
def plot_cdf(data, num_bins, title=None, xlabel=None):
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  counts, bin_edges = np.histogram (data, bins=num_bins)
  cdf = np.cumsum(counts)
  
  sns.set_style("darkgrid")
  plt.figure(figsize=(5,5))
  plt.plot (bin_edges[1:], cdf/cdf[-1])
  plt.ylabel('CDF')
  
  if title:
      plt.title(title)
  if xlabel:
      plt.xlabel(xlabel)

def error_dist(X_s, train_y_s, X_t, train_y_t, error_test, test_y_t, weights= None, title=None):
  from scipy.interpolate import griddata
  import matplotlib.pyplot as plt
  import numpy as np
  rssi_interp_s = griddata(train_y_s, X_s, train_y_t, method='cubic')
  rssi_diff = rssi_interp_s - X_t
  rssi_diff = np.sum(rssi_diff, axis = 1)
  if weights is None:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    plot_scatter_colored(rssi_diff, train_y_t, title = 'measure difference', ax= ax1)
    plot_scatter_colored(error_test, test_y_t, title = 'error distribution', ax= ax2)
  else:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14, 5))
    plot_scatter_colored(rssi_diff, train_y_t, title = 'measure difference', ax= ax1)
    plot_scatter_colored(error_test, test_y_t, title = 'error distribution', ax= ax2)
    plot_scatter_colored(weights, train_y_s, title = 'weights based on kernel', ax= ax3)
  if title is not None:
    fig.suptitle(title)
