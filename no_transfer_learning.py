from NN_models import my_models 
from load_data import load_data
from print_stat import print_stat
import numpy as np
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing
NN_scaling = True
fine_tuning = False
load_data_flag = False
#dataset parameter
source_BuildingID = 0
source_FloorID = 2 #data to pretrain

target_BuildingID = 0
target_FloorID = 0 #data to be fine tuned

target_percentage = 1 #100.0 / 1356 #percentage of the target training data
split_size = 0.3
dropout_pr = 0.5

if load_data_flag:
  file_name =  "trainingData.csv"
  train_x_t, train_y_t, val_x_t, val_y_t, origin_t = \
    load_data(file_name, target_BuildingID, target_FloorID,
              target_percentage, split_size, [], scaling = not NN_scaling)
  
  
  test_file_name = "validationData.csv"
  test_x_t, test_y_t, _, _, _ = load_data(test_file_name, target_BuildingID,
                                      target_FloorID, 1, 0, origin_t,
                                      scaling = not NN_scaling)


#normalize the vectors of Xs and Ys
def normalizer(X):
  mean_X = np.mean(X)
  std_X = np.sqrt(np.var(X))
  X = (X-mean_X) / std_X
  return X

from embedders import embedding

embedding_type = "no_embedding"
embedder = embedding(embedding_type)
# =============================================================================
emb_x_t = embedder.fit_transform(train_x_t)
emb_val_x_t = embedder.fit_transform(val_x_t)
emb_test_x = embedder.fit_transform(test_x_t)

# =============================================================================
num_inputs = emb_x_t.shape[1]# input layer size
# =============================================================================
if 'model' in locals():
  del model
if 'model_obj' in locals():
  del model_obj
  
model_obj = my_models(num_inputs, dropout = dropout_pr)
model = model_obj.build_model()
model = model_obj.fit(emb_x_t, train_y_t, emb_val_x_t, val_y_t, 
                      scale = NN_scaling)

error_normal, train_loss, val_loss, test_loss = \
  model_obj.evaluate(emb_x_t,train_y_t, emb_val_x_t, val_y_t, emb_test_x,
                     test_y_t, scale = NN_scaling)
  
title = 'normal learning'
print_stat(title, error_normal, train_loss, val_loss, test_loss)

# =============================================================================
from plotters import error_dist
#error_dist(emb_x_s, train_y_s, emb_x_t, train_y_t, error_naive,
#           test_y_t, title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_normal, 100)    
plt.show()

#
#del model
#del model_obj