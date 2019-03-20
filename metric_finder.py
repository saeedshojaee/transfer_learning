from NN_models import my_models 
from load_data import load_data
from print_stat import print_stat
import numpy as np
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing

#dataset parameter
NN_scaling = False
source_BuildingID = 0
source_FloorID = 0 #data to pretrain

target_BuildingID = 0
target_FloorID = 3 #data to be fine tuned

source_percentage = 1 #percentage of the srource training data 
target_percentage = 0.05 #percentage of the target training data
split_size = 0.3
dropout_pr = 0.5
file_name =  "trainingData.csv"
train_x_s, train_y_s, val_x_s, val_y_s, origin_s = \
  load_data(file_name, source_BuildingID, source_FloorID,
            source_percentage, split_size, [], scaling =  not NN_scaling)
train_x_t, train_y_t, val_x_t, val_y_t, origin_t = \
  load_data(file_name, target_BuildingID, target_FloorID,
            target_percentage, split_size, [], scaling = not NN_scaling)


test_file_name = "validationData.csv"
test_x_t, test_y_t, _, _, _ = load_data(test_file_name, target_BuildingID,
                                    target_FloorID, 1, 0, origin_s,
                                    scaling = not NN_scaling)

# =============================================================================
#normalize the vectors of Xs and Ys
def normalizer(X):
  mean_X = np.mean(X)
  std_X = np.sqrt(np.var(X))
  X = (X-mean_X) / std_X
  return X
# =============================================================================
from embedders import embedding
n_components = 50
embedding_type = "no_embedding"
embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)

# =============================================================================
if embedding_type != "autoencoder":
  emb_x_s = embedder.fit_transform(train_x_s)
  emb_x_t = embedder.fit_transform(train_x_t)
  emb_val_x_s = embedder.fit_transform(val_x_s)
  emb_val_x_t = embedder.fit_transform(val_x_t)
  emb_test_x = embedder.fit_transform(test_x_t)
else:
  from autoencoder import autoencoder
  emb_x_s, emb_val_x_s, _,_ = autoencoder(train_x_s, emb_val_x_s,
                                          n_components = n_components)
  emb_x_t, emb_val_x_t, ae_enc, ae_dec = autoencoder(train_x_t, emb_val_x_t,
                                                     n_components = n_components)
  emb_test_x = ae_enc(test_x_t)
    
  
if embedding_type == 'tsne' or embedding_type == "autoencoder":
  emb_x_s = emb_x_s.astype(np.double)
  emb_x_t = emb_x_t.astype(np.double)
  emb_val_x_s = emb_val_x_s.astype(np.double)
  emb_val_x_t = emb_val_x_t.astype(np.double)
  emb_test_x = emb_test_x.astype(np.double)
# =============================================================================
from metric_learning import hisc_meme

X_s = np.concatenate((emb_x_s, emb_val_x_s), axis = 0)
X_t = np.concatenate((emb_x_t, emb_val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)
print('Finding the metric')

#Y_s = np.ones(Y_s.shape)
Y_t = np.zeros(Y_t.shape)

M , L = hisc_meme(X_s, Y_s, X_t, Y_t, p = 2, B = 1000,
                  threshhold = 1)

emb_x_s = emb_x_s @ L
emb_val_x_s = emb_val_x_s @ L

emb_x_t = emb_x_t @ L
emb_val_x_t = emb_val_x_t @ L

emb_test_x = emb_test_x @ L
print('Done')

# =============================================================================
kmm_kernel = 'lin'
B = 10
# =============================================================================
#coef_s, coef_t =  ekmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_val_s, coef_val_t =  ekmm(emb_x_t, emb_val_x_s, kern = kmm_kernel, B = B)

#coef_s, coef_t =  ekmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_val_s, coef_val_t =  ekmm(emb_x_t, emb_val_x_s, kern = kmm_kernel, B = B)
# =============================================================================
# This way we preserve the statistics of the signal
x_s = emb_x_s
x_t = emb_x_t
emb_x_s = np.concatenate((emb_x_s, emb_val_x_s), axis = 0)
emb_x_t = np.concatenate((emb_x_t, emb_val_x_t), axis = 0)

print('Kernel mean matching')

from kernel_mean_matching import eprimical_kmm_emb as ekmm_emb
#coef_s =  kmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_s, coef_t =  ekmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_s, coef_t =  ekmmd(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
coef_s, coef_t =  ekmm_emb(emb_x_t, emb_x_s, kern = kmm_kernel, B = B,
                           embedder_type = 'no_embedding', n_components = 20)

emb_x_s = x_s
emb_x_t = x_t
print('Done')

# =============================================================================
#
#emb_y_s = np.concatenate((train_y_s, val_y_s), axis = 0)
#emb_y_t = np.concatenate((train_y_t, val_y_t), axis = 0)
#coef_s, coef_t =  ekmm_emb(emb_y_t, emb_y_s, kern = kmm_kernel, B = B,
#                           embedder_type = 'no_embedding', n_components = 100)
# =============================================================================
coef_val_s = coef_s[train_y_s.shape[0]:]
coef_s = coef_s[:train_y_s.shape[0]]

# =============================================================================
import numpy.matlib as npm
training_weights = npm.repmat(coef_s, 1 , train_y_s.shape[1])
training_val_weights = npm.repmat(coef_val_s, 1 , val_y_s.shape[1])
num_weights = training_weights.shape[1]

num_inputs = emb_x_s.shape[1]# input layer size
# =============================================================================
w_model_obj = my_models(num_inputs, num_weights = num_weights,
                        model_type='weighted', dropout = dropout_pr)
model = w_model_obj.build_model()
model = w_model_obj.fit(emb_x_s, train_y_s, emb_val_x_s, val_y_s, 
                        val_w = training_val_weights ,
                        scale = NN_scaling, training_w = training_weights)

error_sample_bias, train_loss, val_loss, test_loss =\
  w_model_obj.evaluate(emb_x_t, train_y_t, emb_val_x_t, val_y_t,
                       emb_test_x, test_y_t, scale = NN_scaling)
title = 'metric learning plus sample selection bias'
print_stat(title, error_sample_bias, train_loss, val_loss, test_loss)
# =============================================================================

def error_dist(X_s, X_t, error_test, weights= None, title=None):
  from scipy.interpolate import griddata
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
  
error_dist(emb_x_s, emb_x_t, error_sample_bias, weights=coef_s,  title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_sample_bias, 100)    
plt.show()