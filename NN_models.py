import keras.backend as K
from keras.layers import Dense, Input, Dropout
from keras import Model, Sequential
from functools import partial
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import regularizers
import keras

class my_models:
  def __init__(self, num_inputs, model_type = 'sequential', num_weights=None,
               n_components=None, dropout = 0.5):
    self.type = model_type
    self.num_inputs = num_inputs
    self.num_weights = num_weights
    self.n_components = n_components
    self.encoder = None
    self.dropout = dropout
        
  def build_model(self):
    if self.type == 'weighted':
      self.model = build_model_trans_localtions (self.num_inputs,
                                                 self.num_weights,
                                                 dropout = self.dropout)
      return self.model
    elif self.type == 'sequential':
      self.model = build_model_normal_transfer (self.num_inputs,
                                                dropout = self.dropout)
      return self.model
    elif  self.type == 'autoencoder':
      self.model, self.encoder = build_model_autoencoder (self.num_inputs, 
                                                          self.n_components,
                                                          dropout = self.dropout)
      return self.model , self.encoder
    else:
      print("model type is not defined")
      return sys.exit() 
    
  def evaluate(self, train_x, train_y, val_x, val_y, test_x , test_y,
               scale = False):
    if scale:
      from sklearn import preprocessing
      import numpy as np
      # notic ehtat it is not equivalent to the preprocessing before spliting
      n = train_x.shape[0]
      train_x = preprocessing.scale(np.concatenate((train_x, val_x), axis = 0))
      val_x = train_x[n:,:]
      train_x = train_x[:n,:]
      test_x = preprocessing.scale(test_x)
      
    import numpy as np
    if self.type == 'weighted':
      ones_weights = np.ones(train_y.shape)
      train_x = [train_x, ones_weights]
      ones_weights = np.ones(val_y.shape)
      val_x = [val_x, ones_weights]
      ones_weights = np.ones(test_y.shape)
      test_x = [test_x, ones_weights]
    
    train_loss = self.model.evaluate(train_x, train_y,
                                        batch_size=len(train_y), sample_weight=None) #loss for training data2(not trained by neural network)
    val_loss = self.model.evaluate(val_x, val_y,
                                      batch_size=len(val_y), sample_weight=None)
    test_loss = self.model.evaluate(test_x, test_y,
                                       batch_size=len(test_y), sample_weight=None)
    predict_y = self.model.predict(test_x)
    
    error_NN = np.linalg.norm(predict_y - test_y, axis=1)
    return error_NN, train_loss, val_loss, test_loss
  
  def fit(self, X, Y, val_X, val_Y, 
                      scale = False, training_w = None, val_w = None):
    if scale:
      from sklearn import preprocessing
      import numpy as np
      # notic ehtat it is not equivalent to the preprocessing before spliting
      n = X.shape[0]
      X = preprocessing.scale(np.concatenate((X, val_X), axis = 0))
      val_X = X[n:,:]
      X = X[:n,:]
  
    if self.type == 'weighted':
      X = [X, training_w]
      val_X = [val_X, val_w]

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60,
                                                verbose=0, mode='auto')
    self.model.fit(x=X, y=Y,
                   epochs=10000,
                   batch_size=64,
                   callbacks=[earlyStopping],
                   validation_data=(val_X,  val_Y))
    return self.model
  
             

def custom_wighted_loss(y_true, y_pred, weights):
  return K.mean(K.square(y_true - y_pred) * weights, axis=-1)
   
act_fun = 'relu'
regularzation_penalty = 0.08
initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
#Optimizer
adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Regression', histogram_freq=0, write_graph=True, write_images=True)#tensorboard tensorboard --logdir Regression
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')  
  
def build_model_trans_localtions (num_input, num_weight, dropout = 0.25):

  input_layer = Input(shape=(num_input,))
  weights_tensor = Input(shape=(num_weight,))
  cl4 = partial(custom_wighted_loss, weights=weights_tensor)
  inputs = [input_layer, weights_tensor]
  l1 = Dense(500, activation=act_fun, input_dim= num_input , kernel_initializer=initilization_method, 
             kernel_regularizer=regularizers.l2(regularzation_penalty))(input_layer)
  l2 = Dropout(dropout)(l1)
  l3 = Dense(500, activation=act_fun, kernel_initializer=initilization_method,
              kernel_regularizer=regularizers.l2(regularzation_penalty))(l2)
  l4 = Dropout(dropout)(l3)
  l5 = Dense(500, activation=act_fun, kernel_initializer=initilization_method,
             kernel_regularizer=regularizers.l2(regularzation_penalty))(l4)
  l6 = Dropout(dropout)(l5)
  output = Dense(2, activation='linear', kernel_initializer=initilization_method, 
                 kernel_regularizer=regularizers.l2(regularzation_penalty))(l6)
  model = Model(inputs, output)
  #Model compile
  model.compile(loss = cl4 ,
                optimizer=adam)
  return model

#neuron network for extrapolation
#define model
def build_model_normal_transfer (num_input, dropout = 0.25):
  model = Sequential()
  model.add(Dense(500, activation=act_fun, input_dim=num_input, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
  model.add(Dropout(dropout))
  model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
  model.add(Dropout(dropout))
  model.add(Dense(500, activation=act_fun, kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
  model.add(Dropout(dropout))
  model.add(Dense(2, activation='linear', kernel_initializer=initilization_method ,kernel_regularizer=regularizers.l2(regularzation_penalty)))
  #Model compile
  model.compile(loss='mean_squared_error',
                optimizer=adam)
  return model


def build_model_autoencoder (num_input, n_components, dropout = 0.25):
  input_layer = Input(shape=(num_input,))
  
  #encoder
  enc1 = Dense(500, activation=act_fun, input_dim= num_input , kernel_initializer=initilization_method, 
             kernel_regularizer=regularizers.l2(regularzation_penalty))(input_layer)
  enc2 = Dropout(dropout)(enc1)
  enc3 = Dense(500, activation=act_fun, kernel_initializer=initilization_method,
              kernel_regularizer=regularizers.l2(regularzation_penalty))(enc2)
  enc4 = Dropout(dropout)(enc3)
  enc5 = Dense(200, activation=act_fun, kernel_initializer=initilization_method,
             kernel_regularizer=regularizers.l2(regularzation_penalty))(enc4)
  enc6 = Dropout(dropout)(enc5)
  enc7 = Dense(100, activation=act_fun, kernel_initializer=initilization_method,
              kernel_regularizer=regularizers.l2(regularzation_penalty))(enc6)
  enc8 = Dropout(dropout)(enc7)
  enc_out = Dense(n_components, activation=act_fun, kernel_initializer=initilization_method,
             kernel_regularizer=regularizers.l2(regularzation_penalty))(enc8)
  
  #decoder 
  dec1 = Dense(100, activation=act_fun, input_dim= num_input , kernel_initializer=initilization_method, 
             kernel_regularizer=regularizers.l2(regularzation_penalty))(enc_out)
  dec2 = Dropout(dropout)(dec1)
  dec3 = Dense(200, activation=act_fun, kernel_initializer=initilization_method,
              kernel_regularizer=regularizers.l2(regularzation_penalty))(dec2)
  dec4 = Dropout(dropout)(dec3)
  dec5 = Dense(500, activation=act_fun, kernel_initializer=initilization_method,
             kernel_regularizer=regularizers.l2(regularzation_penalty))(dec4)
  dec6 = Dropout(dropout)(dec5)
  dec_out = Dense(num_input, activation='linear', kernel_initializer=initilization_method,
              kernel_regularizer=regularizers.l2(regularzation_penalty))(dec6)
  
  #   autoencoder model
  ae_model = Model(input_layer, dec_out)
  #Model compile
  ae_model.compile(loss='mean_squared_error',
                   optimizer=adam)
  
  #   encoder model
  enc_model = Model(input_layer, enc_out)
  #Model compile
  enc_model.compile(loss='mean_squared_error',
                    optimizer=adam)
  
  
  return ae_model, enc_model
 