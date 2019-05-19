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

try:
  fine_tuning
except :
  fine_tuning = False

try:
  load_data_flag 
except :
  load_data_flag = True 
  

if load_data_flag == True:
  from load_data import load
  train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
    val_x_t, val_y_t, test_x_t, test_y_t = load()

from embedders import variable_embedder, embedding 
# embedding_type = "pca"
# embedder = embedding(embedding_type, n_cmp = 100)
embedding_type = "no_embedding"
embedder = embedding(embedding_type)
# =============================================================================
emb_x_s, emb_val_x_s, emb_x_t, emb_val_x_t, emb_test_x = \
variable_embedder(embedder,\
[train_x_s, val_x_s, train_x_t, val_x_t, test_x_t]) 
# =============================================================================
# emb_x_s = embedder.fit_transform(train_x_s)
# emb_x_t = embedder.fit_transform(train_x_t)
# emb_val_x_s = embedder.fit_transform(val_x_s)
# emb_val_x_t = embedder.fit_transform(val_x_t)
# emb_test_x = embedder.fit_transform(test_x_t)

# =============================================================================
num_inputs = emb_x_s.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'model_obj' in locals():
  del model_obj
  
model_obj = my_models(num_inputs, dropout = dropout_pr)
model = model_obj.build_model()
model = model_obj.fit(emb_x_s, train_y_s, emb_val_x_s, val_y_s, 
                      scale = NN_scaling)

error_naive, nl_train_loss, nl_val_loss, nl_test_loss = \
  model_obj.evaluate(emb_x_t,train_y_t, emb_val_x_t, val_y_t, emb_test_x,
                     test_y_t, scale = NN_scaling)
  
if fine_tuning:
  model = model_obj.fit(emb_x_t, train_y_t, emb_val_x_t, val_y_t, 
                      scale = NN_scaling)

  error_naive_f, nl_train_loss_f, nl_val_loss_f, nl_test_loss_f = \
    model_obj.evaluate(emb_x_t,train_y_t, emb_val_x_t, val_y_t, emb_test_x,
                       test_y_t, scale = NN_scaling)
  
title = 'Naive learning'
print_stat(title, error_naive, nl_train_loss, nl_val_loss, nl_test_loss)

if fine_tuning:
  title = 'Naive learning with fine-tuning'
  print_stat(title, error_naive_f, nl_train_loss_f, nl_val_loss_f, nl_test_loss_f)
# =============================================================================
from plotters import error_dist
error_dist(emb_x_s, train_y_s, emb_x_t, train_y_t, error_naive,
           test_y_t, title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_naive, 100)    
plt.show()
  
del model
del model_obj