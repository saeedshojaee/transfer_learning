def load_data(file_name, building_id, floor_id, percentage, split_size, origin,
              scaling = True, add_power = 0):
  import pandas as pd
  import numpy as np
  from sklearn import preprocessing
  from sklearn.model_selection import train_test_split
  #read training source
  dataset = pd.read_csv(file_name,header = 0)
  #select the floor and building  & (dataset['BUILDINGID'] == BuildingID)
  dataset =  dataset[(dataset['FLOOR'] == floor_id)  & (dataset['BUILDINGID'] == building_id) ] 
  dataset = dataset.sample(frac=1)
  
  #extract the features(delete the location, floor information etc.)
  rss_values = np.asarray(dataset.iloc[:,0:-9]) 
  
  #make undefined measurements -110dbm
  rss_values[rss_values == 100] = -110 
  locations = np.asarray(dataset.iloc[:,-9:-7])
  if len(origin)==0:
    origin = np.amin(locations,axis=0) #calculate the origin
    
#  room_size = np.amax(locations, axis=0) - origin #size of the room
  Y = locations - origin #position respect to origin
  X = np.asarray(rss_values, dtype=np.float64) #convert to numpy array
  num_data = len(X) #number of training points
  # take a percentage of available data
  Y = Y[:int(percentage*num_data)]#take part of the data from source floor
  X = X[:int(percentage*num_data)]
  
  
# =============================================================================
  # scale X to have higher power
  X[X != -110] = X[X != -110] + add_power
#  X = db2pow(X)
  if scaling :
    X = preprocessing.scale(X) #zero mean
    
# =============================================================================

  X, val_X, Y, val_Y = train_test_split(X, Y, test_size=split_size)
  return X, Y, val_X, val_Y, origin


def load(source_BuildingID = 0,
         source_FloorID = 0,
         target_BuildingID = 0,
         target_FloorID = 3,
         source_percentage = 1,
         unlabele_target_percentage = 300.0 / 1356,
         split_size = 0.3,
         scaling = True):
  
  import numpy as np
  import random
  random.seed(0)
  np.random.seed(50)
  #dataset parameter
  
 
  file_name =  "trainingData.csv"
  train_x_s, train_y_s, val_x_s, val_y_s, origin_s = \
    load_data(file_name, source_BuildingID, source_FloorID,
              source_percentage, split_size, [], scaling = scaling, 
              add_power = 80)
      
#  print("Increazing the power level of the source floor with 50dB")
#  train_x_s[train_x_s != -110] = train_x_s[train_x_s != -110] + 50 
#  val_x_s[val_x_s != -110] = val_x_s[val_x_s != -110] + 50 
  
 
  train_x_t, train_y_t, val_x_t, val_y_t, _ = \
    load_data(file_name, target_BuildingID, target_FloorID,
              unlabele_target_percentage, split_size, [], scaling = scaling)
    
  test_file_name = "validationData.csv"
  test_x_t, test_y_t, _, _, _ = load_data(test_file_name, target_BuildingID,
                                      target_FloorID, 1, 0, origin_s,
                                      scaling = scaling)
  
  return train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
val_x_t, val_y_t, test_x_t, test_y_t