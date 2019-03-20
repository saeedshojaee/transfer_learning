def autoencoder(train_x , val_x, n_components = 1):

  import numpy as np
  from sklearn import preprocessing

  num_inputs = train_x.shape[1]# input layer size
  # =============================================================================
   
  from keras.layers import Dense, Input, Dropout
  from keras import Model
  from keras.optimizers import SGD, RMSprop, Adadelta, Adam
  from keras import regularizers
  import keras
  
  act_fun = 'relu'
  regularzation_penalty = 0.5
  initilization_method = 'random_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
  #Optimizer
  adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
  rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
  sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
  tbCallBack = keras.callbacks.TensorBoard(log_dir='./Regression', histogram_freq=0, write_graph=True, write_images=True)#tensorboard tensorboard --logdir Regression
  earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')  
    
  input_layer = Input(shape=(num_inputs,))
  #encoder
  enc1 = Dense(500, activation=act_fun, input_dim= num_inputs )(input_layer)
  enc2 = Dense(100, activation=act_fun)(enc1)
  enc_out = Dense(n_components, activation=act_fun)(enc2)
  #decoder
  dec1 = Dense(100, activation=act_fun)
  dec2 = Dense(500, activation=act_fun)
  dec_out = Dense(num_inputs, activation = 'linear')
  
  dec1_e = dec1(enc_out)
  dec2_e = dec2(dec1_e)
  dec_out_e = dec_out(dec2_e)
  
  
  
  #   autoencoder model
  ae_model = Model(inputs = input_layer, outputs = dec_out_e)
  #Model compile
  ae_model.compile(loss='mean_squared_error',
                     optimizer=adam)
  
  
  earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60,
                                              verbose=0, mode='auto')
  
  
  ae_model.fit(x=train_x, y=train_x,
                 epochs=100,
                 batch_size=64,
                 callbacks=[earlyStopping],
                 shuffle = False,
                 validation_data=(val_x,  val_x))
  
  
  #   encoder model
  enc_model = Model(inputs = input_layer, outputs = enc_out)
   
  enc_x = enc_model.predict(train_x)
  enc_val_x = enc_model.predict(val_x)
  #decoder 
  decoder_input = Input(shape=(n_components,))
  dec1_ = dec1(decoder_input)
  dec2_ = dec2(dec1_)
  dec_out_ = dec_out(dec2_)
  
  dec_model = Model(inputs = decoder_input, outputs = dec_out_)
  
  return enc_x, enc_val_x, enc_model, dec_model



