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

#dataset parameter
source_BuildingID = 0
source_FloorID = 0 #data to pretrain

target_BuildingID = 0
target_FloorID = 1 #data to be fine tuned

source_percentage = 1 #percentage of the srource training data 
target_percentage = 0.3 #percentage of the target training data
split_size = 0.3
dropout_pr = 0.5

file_name =  "trainingData.csv"
train_x_s, train_y_s, val_x_s, val_y_s, origin_s = \
  load_data(file_name, source_BuildingID, source_FloorID,
            source_percentage, split_size, [], scaling =  not NN_scaling)
train_x_t, train_y_t, val_x_t, val_y_t, origin_t = \
  load_data(file_name, target_BuildingID, target_FloorID,
            target_percentage, split_size, [], scaling =  not NN_scaling)


test_file_name = "validationData.csv"
test_x_t, test_y_t, _, _, _ = load_data(test_file_name, target_BuildingID,
                                    target_FloorID, 1, 0, origin_s,
                                    scaling =  not NN_scaling)


print("Increazing the power level of the source floor with 50dB")
train_x_s[train_x_s != 110] = train_x_s[train_x_s != 110] + 80 
val_x_s[val_x_s != 110] = val_x_s[val_x_s != 110] + 80 

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
emb_x_s = embedder.fit_transform(train_x_s)
emb_x_t = embedder.fit_transform(train_x_t)
emb_val_x_s = embedder.fit_transform(val_x_s)
emb_val_x_t = embedder.fit_transform(val_x_t)
emb_test_x = embedder.fit_transform(test_x_t)

# =============================================================================
num_inputs = emb_x_s.shape[1]# input layer size
# =============================================================================

model_obj = my_models(num_inputs, dropout = dropout_pr)
model = model_obj.build_model()
model = model_obj.fit(emb_x_s, train_y_s, emb_val_x_s, val_y_s, 
                      scale = NN_scaling)
model = model_obj.fit(emb_x_t, train_y_t, emb_val_x_t, val_y_t, 
                      scale = NN_scaling)

error_fine_tuning, train_loss, val_loss, test_loss = \
  model_obj.evaluate(emb_x_t,train_y_t, emb_val_x_t, val_y_t, emb_test_x,
                     test_y_t, scale = NN_scaling)
  
title = 'Naive learning + fine tuning'
print_stat(title, error_fine_tuning, train_loss, val_loss, test_loss)

# =============================================================================
from plotters import error_dist
error_dist(emb_x_s, train_y_s, emb_x_t, train_y_t, error_fine_tuning,
           test_y_t, title = title)
plt.show()

from plotters import plot_cdf
plot_cdf(error_fine_tuning, 100)    
plt.show()