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

if load_data_flag:
  from load_data import load
  train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
    val_x_t, val_y_t, test_x_t, test_y_t = load()

# =============================================================================
kmm_kernel = 'lin'
B = 10
# =============================================================================
# This way we preserve the statistics of the signal

x_s = np.concatenate((train_x_s, val_x_s), axis = 0)
x_t = np.concatenate((train_x_s, val_x_t), axis = 0)


print('Kernel mean matching')
#coef_s =  kmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_s, coef_t =  ekmm(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
#coef_s, coef_t =  ekmmd(emb_x_t, emb_x_s, kern = kmm_kernel, B = B)
coef_s, coef_t =  ekmm_emb(x_t, x_s, kern = kmm_kernel, B = B,
                           embedder_type = 'no_embedding', n_components = 20)


print('Done')

# =============================================================================
coef_val_s = coef_s[train_y_s.shape[0]:]
coef_s = coef_s[:train_y_s.shape[0]]

# =============================================================================
training_weights = npm.repmat(coef_s, 1 , train_y_s.shape[1])
training_val_weights = npm.repmat(coef_val_s, 1 , val_y_s.shape[1])
num_weights = training_weights.shape[1]

num_inputs = train_x_s.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'w_model_obj' in locals():
  del w_model_obj
  
w_model_obj = my_models(num_inputs, num_weights = num_weights,
                        model_type='weighted', dropout = dropout_pr)
model = w_model_obj.build_model()
model = w_model_obj.fit(train_x_s, train_y_s, val_x_s, val_y_s, 
                        val_w = training_val_weights ,
                        scale = NN_scaling, training_w = training_weights)

error_sample_bias, ssbc_train_loss, ssbc_val_loss, ssbc_test_loss =\
  w_model_obj.evaluate(train_x_t, train_y_t, val_x_t, val_y_t,
                       test_x_t, test_y_t, scale = NN_scaling)
  
if fine_tuning:
  training_weights = np.ones(train_y_t.shape)
  training_val_weights = np.ones(val_y_t.shape)
  model = w_model_obj.fit(train_x_t, train_y_t, val_x_t, val_y_t, 
                          val_w = training_val_weights ,
                          scale = NN_scaling, training_w = training_weights)

  error_sample_bias_f, ssbc_train_loss_f, ssbc_val_loss_f, ssbc_test_loss_f =\
    w_model_obj.evaluate(train_x_t, train_y_t, val_x_t, val_y_t,
                         test_x_t, test_y_t, scale = NN_scaling)
title = 'sample selection bias'
print_stat(title, error_sample_bias, ssbc_train_loss, ssbc_val_loss, ssbc_test_loss)

if fine_tuning:
  title = 'sample selection bias with fine_tuning'
  print_stat(title, error_sample_bias_f, ssbc_train_loss_f, ssbc_val_loss_f, ssbc_test_loss_f)
# =============================================================================
from plotters import error_dist 
error_dist(train_x_s, train_y_s, train_x_t, train_y_t, error_sample_bias,
           test_y_t, weights=coef_s,  title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_sample_bias, 100)    
plt.show()


del model
del w_model_obj