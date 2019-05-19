import numpy as np
def print_stat(name, error_stat, training_loss, val_loss, test_loss):
  print('\n',name)
  print("\nLoss for training:",training_loss)
  print("Loss for validation:",val_loss)
  print("Loss for test data: ",test_loss)
  print_error_dist(error_stat)
  
  
def print_error_dist(error_stat, name = None):
  if name:
    print('\n',name)
  print('\naverage error:' ,np.mean(error_stat),
        '\nminimum error:', np.amin(error_stat), 
        '\nmaximum error:', np.amax(error_stat),
        '\nvariance:     ', np.var(error_stat))


