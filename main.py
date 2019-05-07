from NN_models import my_models 
from print_stat import print_stat
import matplotlib.pyplot as plt
from utility_func import sum_power
from plotters import plot_scatter_colored 
from plotters import plot_cmp
from sklearn import preprocessing
from keras.utils import plot_model

from load_data import load

train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
  val_x_t, val_y_t, test_x_t, test_y_t = load()


fine_tuning = True
%run -i naive_learning.py

fine_tuning = False
%run -i metric_learning_training.py
%run -i sample_selection_bias_by_unlabeled_tranining.py
%run -i transformed_sample_selection.py



print("cat 1 no data from target location")
title = 'Naive learning'
print_stat(title, error_naive, nl_train_loss, nl_val_loss, nl_test_loss)

print("cat 2 additional labeled data set from target location")
title = 'Naive learning with fine-tuning'
print_stat(title, error_naive_f, nl_train_loss_f, nl_val_loss_f, nl_test_loss_f)

title = 'metric learning'
print_stat(title, error_metric, ml_train_loss, ml_val_loss, ml_test_loss)

print("cat 3 unlabeled data set from target location")
title = 'sample selection bias'
print_stat(title, error_sample_bias, ssbc_train_loss, ssbc_val_loss, ssbc_test_loss)

title = 'no labeled data - transformed sample selection bias'
print_stat(title, error_metric_plus_sample, t_ssbc_train_loss, t_ssbc_val_loss, t_ssbc_test_loss)