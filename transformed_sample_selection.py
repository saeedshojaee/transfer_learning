from NN_models import my_models
from print_stat import print_stat
import numpy as np
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing

dropout_pr = 0.5
NN_scaling = False
fine_tuning = False
load_data_flag = True

if load_data_flag == True:
  from load_data import load
  train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
    val_x_t, val_y_t, test_x_t, test_y_t = load()
# =============================================================================
# from embedders import embedding
# n_components = 50
# embedding_type = "no_embedding"
# embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
# 
# # =============================================================================
# if embedding_type != "autoencoder":
#   emb_x_s = embedder.fit_transform(train_x_s)
#   # emb_x_t = embedder.fit_transform(train_x_t)
#   emb_val_x_s = embedder.fit_transform(val_x_s)
#   emb_val_x_t = embedder.fit_transform(val_x_t)
#   
#   emb_u_x_t = embedder.fit_transform(train_x_t)
#   emb_val_x_t = embedder.fit_transform(val_x_t)
#   
#   emb_test_x = embedder.fit_transform(test_x_t)
# else:
#   from autoencoder import autoencoder
#   emb_x_s, emb_val_x_s, _,_ = autoencoder(train_x_s, emb_val_x_s,
#                                           n_components = n_components)
#   emb_x_t, emb_val_x_t, ae_enc, ae_dec = autoencoder(train_x_t, emb_val_x_t,
#                                                      n_components = n_components)
#   
#   emb_test_x = ae_enc(test_x_t)
#     
#   
# if embedding_type == 'tsne' or embedding_type == "autoencoder":
#   emb_x_s = emb_x_s.astype(np.double)
#   # emb_x_t = emb_x_t.astype(np.double)
#   emb_val_x_s = emb_val_x_s.astype(np.double)
#   # emb_val_x_t = emb_val_x_t.astype(np.double)
#   emb_test_x = emb_test_x.astype(np.double)
#   
#   emb_x_t = emb_x_t.astype(np.double)
#   emb_val_x_t = emb_val_x_t.astype(np.double)
# # =============================================================================
# from metric_learning import hisc_meme_no_labeled_data
# 
# X_s = np.concatenate((emb_x_s, emb_val_x_s), axis = 0)
# Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
# X_t = np.concatenate((emb_x_t, emb_val_x_t), axis = 0)
# 
# print('Finding the metric')
# M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000,
#                   threshhold = 1)
# 
# emb_x_s = emb_x_s @ L
# emb_val_x_s = emb_val_x_s @ L
# 
# emb_x_t = emb_x_t @ L
# emb_val_x_t = emb_val_x_t @ L
# 
# emb_test_x = emb_test_x @ L
# 
# emb_x_t = emb_x_t @ L
# val_x_t = emb_val_x_t @ L
# 
# =============================================================================

X_s = np.concatenate((train_x_s, val_x_s), axis = 0)
X_t = np.concatenate((train_x_t, val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)

print('Finding the transformation')

from metric_learning import hisc_meme_no_labeled_data

M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000,
                  threshhold = 1)

t_ssbc_x_s = train_x_s @ L
t_ssbc_val_x_s = val_x_s @ L

t_ssbc_x_t = train_x_t @ L
t_ssbc_val_x_t = val_x_t @ L
t_ssbc_test_x = test_x_t @ L


print('Done')

# =============================================================================
print('Kernel mean matching')
from kernel_mean_matching import eprimical_kmm_emb as ekmm_emb
t_ssbc_x_s_tmp = np.concatenate((t_ssbc_x_s, t_ssbc_val_x_s), axis = 0)
t_ssbc_x_t_tmp = np.concatenate((t_ssbc_x_t, t_ssbc_val_x_t), axis = 0)

kmm_kernel = 'lin'
B = 10
coef_s, coef_t =  ekmm_emb(t_ssbc_x_t_tmp, t_ssbc_x_s_tmp, kern = kmm_kernel, B = B,
                           embedder_type = 'no_embedding', n_components = 100)
print('Done')
# =============================================================================
coef_val_s = coef_s[t_ssbc_x_s.shape[0]:]
coef_s = coef_s[:t_ssbc_x_s.shape[0]]

# =============================================================================
import numpy.matlib as npm
training_weights = npm.repmat(coef_s, 1 , train_y_s.shape[1])
training_val_weights = npm.repmat(coef_val_s, 1 , val_y_s.shape[1])
num_weights = training_weights.shape[1]
# =============================================================================
num_inputs = t_ssbc_x_s.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'w_model_obj' in locals():
  del w_model_obj  
  
w_model_obj = my_models(num_inputs, num_weights = num_weights,
                        model_type='weighted', dropout = dropout_pr)
model = w_model_obj.build_model()
model = w_model_obj.fit(t_ssbc_x_s, train_y_s, t_ssbc_val_x_s, val_y_s, 
                        val_w = training_val_weights ,
                        scale = NN_scaling, training_w = training_weights)

error_metric_plus_sample, t_ssbc_train_loss, t_ssbc_val_loss, t_ssbc_test_loss =\
  w_model_obj.evaluate(t_ssbc_x_t, train_y_t, t_ssbc_val_x_t, val_y_t,
                       t_ssbc_test_x, test_y_t, scale = NN_scaling)

if fine_tuning:
  training_weights = np.ones(train_y_t.shape)
  training_val_weights = np.ones(val_y_t.shape)
  model = w_model_obj.fit(t_ssbc_x_t, train_y_t, t_ssbc_val_x_t, val_y_t, 
                          val_w = training_val_weights ,
                          scale = NN_scaling, training_w = training_weights)

  error_metric_plus_sample_f, t_ssbc_train_loss_f, t_ssbc_val_loss_f, t_ssbc_test_loss_f =\
    w_model_obj.evaluate(t_ssbc_x_t, train_y_t, t_ssbc_val_x_t, val_y_t,
                         t_ssbc_test_x, test_y_t, scale = NN_scaling)
  
title = 'no labeled data - transformed sample selection bias'
print_stat(title, error_metric_plus_sample, t_ssbc_train_loss, t_ssbc_val_loss, t_ssbc_test_loss)

if fine_tuning:
  title = 'no labeled data -  transformed sample selection bias'
  print_stat(title, error_metric_plus_sample_f, t_ssbc_train_loss_f, t_ssbc_val_loss_f,
             t_ssbc_test_loss_f)
# =============================================================================
  
from plotters import error_dist
error_dist(t_ssbc_x_s, train_y_s, t_ssbc_x_t, train_y_t, error_metric_plus_sample,
           test_y_t, weights=coef_s,  title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_metric_plus_sample, 100)    
plt.show()

del model
del w_model_obj