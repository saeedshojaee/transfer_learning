from NN_models import my_models 
from load_data import load_data
from print_stat import print_stat
import numpy as np
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing

NN_scaling = False
fine_tuning = True
load_data_flag = True
#dataset parameter
source_BuildingID = 0
source_FloorID = 0 #data to pretrain

target_BuildingID = 0
target_FloorID = 3 #data to be fine tuned

source_percentage = 1 #percentage of the srource training data 
labele_target_percentage = 100.0 / 1356  #percentage of the target training data
unlabele_target_percentage = 300.0 / 1356 #percentage of the target training data

split_size = 0.3
dropout_pr = 0.5

if load_data_flag:
  file_name =  "trainingData.csv"
  train_x_s, train_y_s, val_x_s, val_y_s, origin_s = \
    load_data(file_name, source_BuildingID, source_FloorID,
              source_percentage, split_size, [], scaling = not NN_scaling, 
              add_power = 80)
      
#  print("Increazing the power level of the source floor with 50dB")
#  train_x_s[train_x_s != -110] = train_x_s[train_x_s != -110] + 50 
#  val_x_s[val_x_s != -110] = val_x_s[val_x_s != -110] + 50 
  
  
  
  train_x_t, train_y_t, val_x_t, val_y_t, origin_t = \
    load_data(file_name, target_BuildingID, target_FloorID,
              labele_target_percentage, split_size, [], scaling = not NN_scaling)
  
  u_train_x_t, u_train_y_t, u_val_x_t, u_val_y_t, _ = \
    load_data(file_name, target_BuildingID, target_FloorID,
              unlabele_target_percentage, split_size, [], scaling = not NN_scaling)
    
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
  
  emb_u_x_t = embedder.fit_transform(u_train_x_t)
  emb_u_val_x_t = embedder.fit_transform(u_val_x_t)
  
  emb_test_x = embedder.fit_transform(test_x_t)
else:
  from autoencoder import autoencoder
  emb_x_s, emb_val_x_s, _,_ = autoencoder(train_x_s, emb_val_x_s,
                                          n_components = n_components)
  emb_x_t, emb_val_x_t, ae_enc, ae_dec = autoencoder(train_x_t, emb_val_x_t,
                                                     n_components = n_components)
  emb_u_x_t, emb_u_val_x_t, ae_enc, ae_dec = autoencoder(train_x_t, emb_val_x_t,
                                                     n_components = n_components)
  
  emb_test_x = ae_enc(test_x_t)
    
  
if embedding_type == 'tsne' or embedding_type == "autoencoder":
  emb_x_s = emb_x_s.astype(np.double)
  emb_x_t = emb_x_t.astype(np.double)
  emb_val_x_s = emb_val_x_s.astype(np.double)
  emb_val_x_t = emb_val_x_t.astype(np.double)
  emb_test_x = emb_test_x.astype(np.double)
  
  emb_u_x_t = emb_u_x_t.astype(np.double)
  emb_u_val_x_t = emb_u_val_x_t.astype(np.double)
# =============================================================================
from metric_learning import hisc_meme

X_s = np.concatenate((emb_x_s, emb_val_x_s), axis = 0)
X_t = np.concatenate((emb_x_t, emb_val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)

print('Finding the metric')
M , L = hisc_meme(X_s, Y_s, X_t, Y_t, p = 2, B = 1000,
                  threshhold = 1)

M , L = hisc_meme_modified(X_s, Y_s, X_t, Y_t, p = 2, B = 1000,
                  threshhold = 1)

emb_x_s = emb_x_s @ L
emb_val_x_s = emb_val_x_s @ L

emb_x_t = emb_x_t @ L
emb_val_x_t = emb_val_x_t @ L

emb_test_x = emb_test_x @ L

emb_u_x_t = emb_u_x_t @ L
u_val_x_t = emb_u_val_x_t @ L
print('Done')

# =============================================================================
print('Kernel mean matching')
from kernel_mean_matching import eprimical_kmm_emb as ekmm_emb
x_s = emb_x_s
x_t = emb_u_x_t
emb_x_s_tmp = np.concatenate((emb_x_s, emb_val_x_s), axis = 0)
emb_x_t_tmp = np.concatenate((emb_u_x_t, emb_u_val_x_t), axis = 0)

kmm_kernel = 'lin'
B = 10
coef_s, coef_t =  ekmm_emb(emb_x_t_tmp, emb_x_s_tmp, kern = kmm_kernel, B = B,
                           embedder_type = 'no_embedding', n_components = 100)
print('Done')
# =============================================================================
coef_val_s = coef_s[emb_x_s.shape[0]:]
coef_s = coef_s[:emb_x_s.shape[0]]

# =============================================================================
import numpy.matlib as npm
training_weights = npm.repmat(coef_s, 1 , train_y_s.shape[1])
training_val_weights = npm.repmat(coef_val_s, 1 , val_y_s.shape[1])
num_weights = training_weights.shape[1]
# =============================================================================
num_inputs = emb_x_s.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'w_model_obj' in locals():
  del w_model_obj  
  
w_model_obj = my_models(num_inputs, num_weights = num_weights,
                        model_type='weighted', dropout = dropout_pr)
model = w_model_obj.build_model()
model = w_model_obj.fit(emb_x_s, train_y_s, emb_val_x_s, val_y_s, 
                        val_w = training_val_weights ,
                        scale = NN_scaling, training_w = training_weights)

error_metric_plus_sample, train_loss, val_loss, test_loss =\
  w_model_obj.evaluate(emb_u_x_t, u_train_y_t, emb_u_val_x_t, u_val_y_t,
                       emb_test_x, test_y_t, scale = NN_scaling)

if fine_tuning:
  training_weights = np.ones(train_y_t.shape)
  training_val_weights = np.ones(val_y_t.shape)
  model = w_model_obj.fit(emb_x_t, train_y_t, emb_val_x_t, val_y_t, 
                          val_w = training_val_weights ,
                          scale = NN_scaling, training_w = training_weights)

  error_metric_plus_sample_f, train_loss_f, val_loss_f, test_loss_f =\
    w_model_obj.evaluate(emb_u_x_t, u_train_y_t, emb_u_val_x_t, u_val_y_t,
                         emb_test_x, test_y_t, scale = NN_scaling)
  
title = 'metric training plus sample selection bias'
print_stat(title, error_metric_plus_sample, train_loss, val_loss, test_loss)

if fine_tuning:
  title = 'metric training plus sample selection bias'
  print_stat(title, error_metric_plus_sample_f, train_loss_f, val_loss_f,
             test_loss_f)
# =============================================================================
  
from plotters import error_dist
error_dist(emb_x_s, train_y_s, emb_u_x_t, u_train_y_t, error_metric_plus_sample,
           test_y_t, weights=coef_s,  title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_metric_plus_sample, 100)    
plt.show()
#
#del model
#del w_model_obj