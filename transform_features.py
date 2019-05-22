from NN_models import my_models
from print_stat import print_stat
import numpy as np
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing

from load_data import load

dropout_pr = 0.5
NN_scaling = False
fine_tuning = False

try:
  load_data_flag 
except :
  load_data_flag = True 
  
if load_data_flag == True:
  from load_data import load
  train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
    val_x_t, val_y_t, test_x_t, test_y_t = load()


X_s = np.concatenate((train_x_s, val_x_s), axis = 0)
X_t = np.concatenate((train_x_t, val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)
print('Finding the metric')

from metric_learning import hisc_meme_no_labeled_data, HSIC2, hisc_meme
M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000)
# M , L = hisc_meme(X_s, X_s, X_t, X_t, p = 2, B = 1000)
# M , L = HSIC2(X_s, Y_s, p = 2, B = 1000)
# from metric_learning import hisc_meme_no_loc
# M , L = hisc_meme_no_loc(X_s, X_t, p = 2, B = 1000,
#                   threshhold = 1)


ml_x_s = train_x_s @ L
ml_val_x_s = val_x_s @ L

ml_x_t = train_x_t @ L
ml_val_x_t = val_x_t @ L

ml_test_x = test_x_t @ L
print('Done')

# =============================================================================

num_inputs = ml_x_s.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'model_obj' in locals():
  del model_obj
model_obj = my_models(num_inputs, dropout = dropout_pr)
model = model_obj.build_model()
model = model_obj.fit(ml_x_s, train_y_s, ml_val_x_s, val_y_s, 
                      scale = NN_scaling)


error_metric, ml_train_loss, ml_val_loss, ml_test_loss = \
  model_obj.evaluate(ml_x_t,train_y_t, ml_val_x_t, val_y_t, ml_test_x,
                     test_y_t, scale = NN_scaling)
  
if fine_tuning:
  model = model_obj.fit(ml_x_t, train_y_t, ml_val_x_t, val_y_t, 
                      scale = NN_scaling)

  error_metric_f, ml_train_loss_f, ml_val_loss_f, ml_test_loss_f = \
    model_obj.evaluate(ml_x_t,train_y_t, ml_val_x_t, val_y_t, ml_test_x,
                       test_y_t, scale = NN_scaling)
    
title = 'metric learning'
print_stat(title, error_metric, ml_train_loss, ml_val_loss, ml_test_loss)
if fine_tuning:
  title = 'metric learning with fine tuning'
  print_stat(title, error_metric_f, ml_train_loss_f, ml_val_loss_f, ml_test_loss_f)

# =============================================================================

from plotters import error_dist
  
error_dist(ml_x_s, train_y_s, ml_x_t, train_y_t, error_metric,
           test_y_t,  title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_metric, 100)    
plt.show()

del model
del model_obj
