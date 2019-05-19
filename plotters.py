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


def plot_embeding(x_s, x_t, coef_s, train_y_s, train_y_t, fig_name = None):
 
  import numpy as np
  coef_ind = coef_s > np.mean(coef_s)
  coef_ind = coef_ind.reshape(coef_ind.shape[0],)
  from embedders import embedding, variable_embedder
  n_components = 2
  embedding_type = "autoencoder"
  embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
  if embedding_type == "autoencoder":
    split = 0.3
    cut = np.floor(x_t.shape[0] * (1- split) ).astype(int)
    test_X = x_t[:cut,:]
    val_X  = x_t[cut:, :]
    cut = np.floor(x_s.shape[0] * (1 - split) ).astype(int)
    test_C = x_s[:cut,:]
    val_C  = x_s[cut:, :]
    emb_c, emb_val_c, emb_x, emb_val_x = variable_embedder(embedder,\
                        [test_C, test_X],[val_C, val_X])
    emb_train_x_s = np.concatenate((emb_c, emb_val_c), axis = 0)
    emb_train_x_t = np.concatenate((emb_x, emb_val_x), axis = 0)
  else:
    emb_train_x_s, emb_train_x_t, = variable_embedder(embedder,[x_s, x_t])
  ##############################################################################
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set_style("whitegrid")
  bg_color = 'white'
  fg_color = 'black'
  f = plt.figure(figsize=(10,10),facecolor=bg_color, edgecolor=fg_color)

  marker_size = 5
  
  cmap_s = cmap2d(train_y_s)
  cmap_t = cmap2d(train_y_t)
  # print(cmap_t.shape)
  # print(emb_train_x_t.shape)
  l00 = plt.scatter(emb_train_x_s[:,0],emb_train_x_s[:,1], c = cmap_s,
              label  = 'source', marker =  "o", s = 3)
  # plt.setp(l00, markersize=marker_size)
  l01 = plt.scatter(emb_train_x_t[:,0],emb_train_x_t[:,1], c = cmap_t,
              label  = 'target', marker =  "^", s = 3)
  # # plt.setp(l01, markersize=marker_size-3)
  
  l22 = plt.scatter(emb_train_x_s[coef_ind,0],emb_train_x_s[coef_ind,1], c ='r',
              label  = 'ssbc', s = 1)
  # plt.setp(l22, markersize=1)
  if fig_name:
    plt.savefig( fig_name + ".svg", dpi=1200)
  else:
    plt.show()
  
  
  
def cmap2d(data):
  import colormap2d as cmap2d
  rgb = cmap2d.data2d_to_rgb(data.T.reshape(data.shape[1],data.shape[0],1), cmap2d = 'wheel')
  cmap2d = rgb.reshape(rgb.shape[0], rgb.shape[2])
  return cmap2d