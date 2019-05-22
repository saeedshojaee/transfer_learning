from __future__ import print_function, division

# from keras.datasets import mnist
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import RepeatVector, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

import keras.backend as K
from keras.layers import Dense, Input, Dropout
from keras import Model, Sequential
from functools import partial
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras import regularizers
import keras


import sys

import numpy as np

# def initializaer():
#   act_fun = 'relu'
#   regularzation_penalty = 0.08
#   initilization_method = 'he_normal' #'random_uniform' ,'random_normal','TruncatedNormal' ,'glorot_uniform', 'glorot_nomral', 'he_normal', 'he_uniform'
#   #Optimizer
#   adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#   # adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
#   rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#   sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
#   tbCallBack = keras.callbacks.TensorBoard(log_dir='./Regression', histogram_freq=0, write_graph=True, write_images=True)#tensorboard tensorboard --logdir Regression
#   earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=0, mode='auto')  
  
class DANN():
    def __init__(self, num_input):
        self.optimizer_1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.optimizer_2 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.act_fun = 'relu'
        self.num_input = num_input
        self.encode_num = 200
        self.discriminator_num = num_input

        self.dropout = 0.5
        
        

        # Build and compile the discriminator
        self.discriminator = self.build_floor_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=self.optimizer_1,
            metrics=['accuracy'])

        # Build and compile the localization
        self.localization = self.build_localization()
        self.localization.compile(loss='mean_squared_error',
            optimizer=self.optimizer_2,
            metrics=['accuracy'])
       
        
        # Build and compile the encoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss='mean_squared_error', optimizer=self.optimizer_2)

        # Build and compile the decoder
        self.decoder = self.build_decoder()
        self.decoder.compile(loss='mean_squared_error', optimizer=self.optimizer_2)


        # generator takes x as input
        x = Input(shape=(self.num_input ,))
        y = self.encoder(x)
		z = self.decoder(y)
 
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(z)
        
        # The location takes generated images as input and localize
        location = self.localization(y)

        # The combined model  
        self.encoder.trainable = True
        self.decoder.trainable = False
        self.autoencoder = Model(x, z)
        self.autoencoder.compile(loss='mean_squared_error', optimizer=self.optimizer_2)
        
        self.decoder.trainable = True
        self.floor_combined = Model(y, valid)
        self.floor_combined.compile(loss='binary_crossentropy', optimizer=self.optimizer_1)
        
 
        self.encoder.trainable = True
        self.localization.trainable = True
        self.localization_combined = Model(x, location)
        self.localization_combined.compile(loss='mean_squared_error', optimizer=self.optimizer_2)
            
    
    def build_encoder(self):
        initilization_method = 'he_normal'
        regularzation_penalty = 0.08
        input_layer = Input(shape=(self.num_input,))
        l1 = Dense(500, activation=self.act_fun, input_dim = self.num_input, kernel_initializer=initilization_method, 
                   kernel_regularizer=regularizers.l2(regularzation_penalty))(input_layer)
        l1 = Dropout(self.dropout)(l1)
        l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l1)
        l3 = Dropout(self.dropout)(l3)
        l4 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        y = Dense(self.encode_num, activation=self.act_fun, kernel_initializer=initilization_method)(l4)
        return Model(input_layer, y)
    
    
    def build_decoder(self):
        initilization_method = 'he_normal'
        regularzation_penalty = 0.08
        input_layer = Input(shape=(self.encode_num,))
        l1 = Dense(500, activation=self.act_fun, input_dim = self.num_input, kernel_initializer=initilization_method, 
                   kernel_regularizer=regularizers.l2(regularzation_penalty))(input_layer)
        l1 = Dropout(self.dropout)(l1)
        l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l1)
        l3 = Dropout(self.dropout)(l3)
        l4 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        y = Dense(self.discriminator_num, activation=self.act_fun, kernel_initializer=initilization_method)(l4)
        return Model(input_layer, y)
    
    def build_floor_discriminator(self):
        initilization_method = 'he_normal'
        regularzation_penalty = 0.08
        y = Input(shape=(self.discriminator_num,))
        import flipGradientTF
        Flip = flipGradientTF.GradientReversal(1)
        dann_in = Flip(y)
        dann_out = Dense(2)(dann_in)     
        # from flip_gradient import flip_gradient
        # dann_out = flip_gradient(y)
        # dann_out = y
        
        l1 = Dense(800, activation=self.act_fun, input_dim= self.encode_num , kernel_initializer=initilization_method, 
                   kernel_regularizer=regularizers.l2(regularzation_penalty))(dann_out)
        # l1 = Dropout(self.dropout)(l1)
        l3 = Dense(800, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l1)
        # l3 = Dropout(self.dropout)(l3)
        l3 = Dense(800, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        l3 = Dense(800, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        l4 = Dense(20, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
                
        output = Dense(1, activation='sigmoid', kernel_initializer=initilization_method)(l4)
        return Model(y, output)
      
    def build_localization(self):
        initilization_method = 'he_normal'
        regularzation_penalty = 0.08
        y = Input(shape=(self.encode_num,))
        l1 = Dense(500, activation=self.act_fun, input_dim= self.encode_num , kernel_initializer=initilization_method, 
                   kernel_regularizer=regularizers.l2(regularzation_penalty))(y)
        l1 = Dropout(self.dropout)(l1)
        l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l1)
        l3 = Dropout(self.dropout)(l3)
        l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        # l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
        #             kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        # l3 = Dropout(self.dropout)(l3)
       # l3 = Dense(500, activation=self.act_fun, kernel_initializer=initilization_method,
        #            kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        l4 = Dense(100, activation=self.act_fun, kernel_initializer=initilization_method,
                    kernel_regularizer=regularizers.l2(regularzation_penalty))(l3)
        location = Dense(2, activation='linear', kernel_initializer=initilization_method)(l4)
        return Model(y, location)
  

    def train(self, train_x_s, train_y_s, val_x_s, val_y_s,\
              train_x_t,train_y_t, val_x_t, val_y_t,
              epochs, batch_size=128, save_interval=50):
    
        # self.discriminator.trainable = True
        # self.encoder.trainable = True
        half_batch = int(batch_size / 2)
    
        for epoch in range(epochs):

            # ---------------------
            #  Train floor detection
            # ---------------------
            if epoch > 0:
              self.discriminator.trainable = True
              self.decoder.trainable = True
              half_batch = int(batch_size/2)
              idx = np.random.permutation(train_x_s.shape[0])
              idx = idx[:half_batch]
              x_s = train_x_s[idx,:]
              if train_x_t.shape[0] < half_batch :
                x_t_batch = train_x_t.shape[0]
              else:
                x_t_batch = half_batch
              idx = np.random.permutation(train_x_t.shape[0])
              idx = idx[:x_t_batch]
              x_t = train_x_t[idx,:]
              encoded_s =  self.encoder.predict(x_s)
              encoded_t =  self.encoder.predict(x_t)
              
              d_loss_source = self.floor_combined.train_on_batch(encoded_s, np.ones((half_batch, 1)))
              d_loss_target = self.floor_combined.train_on_batch(encoded_t, np.ones((x_t_batch, 1)))
              d_loss = 0.5 * np.add(d_loss_source, d_loss_target)
                          
              # d_loss_source = self.floor_combined_d.train_on_batch(x_s, np.ones((half_batch, 1)))
              # d_loss_target = self.floor_combined_d.train_on_batch(x_t, np.zeros((half_batch, 1)))
              # d_loss = 0.5 * np.add(d_loss_source, d_loss_target)
            else:
              d_loss = [0,0]
              
            # ---------------------
            #  Train encoder
            # ---------------------
            self.encoder.trainable = True

            if epoch > 0:
              if train_x_t.shape[0] < batch_size :
                x_t_batch = train_x_t.shape[0]
              else:
                x_t_batch = batch_size
              idx = np.random.permutation(train_x_s.shape[0])
              idx = idx[:x_t_batch]
              x_t = train_x_s[idx,:]
              self.decoder.trainable = False
              g_loss = self.autoencoder.train_on_batch(x_t, x_t)
              # g_loss = self.floor_combined.train_on_batch(x_t, np.ones((batch_size, 1)))
              self.encoder.trainable = False

            else:
              g_loss = 0
          
            if epoch > 0:
              self.localization.trainable = True

              # ---------------------
              #  Train localization
              # ---------------------
              idx = np.random.permutation(train_x_s.shape[0])
              idx = idx[:batch_size]
              x_s = train_x_s[idx,:]
              y_s = train_y_s[idx,:]
              
              localization_error = self.localization_combined.train_on_batch(x_s, y_s)
              loc_error = np.mean(localization_error)  
            else:
              loc_error = 0
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [error: %f] [G loss: %f]" \
                    % (epoch, d_loss[0], 100*d_loss[1], loc_error, g_loss))
            # print ("%d [D loss: %f, acc.: %.2f%%]  [error: %f]" \
            #         % (epoch, d_loss[0], 100*d_loss[1], loc_error))
    
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                p_val_y_s = self.localization_combined.predict(val_x_s)
                p_val_y_t = self.localization_combined.predict(val_x_t)
                val_y_s_error = np.linalg.norm(p_val_y_s - val_y_s, axis=1)
                val_y_t_error = np.linalg.norm(p_val_y_t - val_y_t, axis=1)
                print('-'*80)
                print ("Validation errors: source: %f, \t target: %f" %\
                        (np.mean(val_y_s_error), np.mean(val_y_t_error)))
                print('-'*80)
                
                source = self.floor_combined.predict(val_x_s)
                target = self.floor_combined.predict(val_x_t)
                
                mean_s = np.mean(source)
                mean_t = 1 - np.mean(target)
                
                print('-'*80)
                print ("Validation errors: source: %f, \t target: %f" %\
                       (np.mean(mean_s), np.mean(mean_t)))
                print('-'*80)
             
        
if __name__ == '__main__':

    from NN_models import my_models
    from print_stat import print_stat, print_error_dist
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
    # =============================================================================
    
    # X_s = np.concatenate((train_x_s, val_x_s), axis = 0)
    # X_t = np.concatenate((train_x_t, val_x_t), axis = 0)
    # Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
    # Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)
    # print('Finding the metric')
    
    # from metric_learning import hisc_meme_no_labeled_data, HSIC2, hisc_meme
    # M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000)
    # # M , L = hisc_meme(X_s, X_s, X_t, X_t, p = 2, B = 1000)
    # # M , L = HSIC2(X_s, Y_s, p = 2, B = 1000)
    # # from metric_learning import hisc_meme_no_loc
    # # M , L = hisc_meme_no_loc(X_s, X_t, p = 2, B = 1000,
    # #                   threshhold = 1)
    
    
    # train_x_s = train_x_s @ L
    # val_x_s = val_x_s @ L
    
    # train_x_t = train_x_t @ L
    # val_x_t = val_x_t @ L
    
    # test_x_t = test_x_t @ L
    # =============================================================================
    dann = DANN(train_x_s.shape[1])
    dann.train(train_x_s, train_y_s, val_x_s, val_y_s, \
               train_x_t,train_y_t, val_x_t, val_y_t,
                  epochs=100000, batch_size=500, save_interval=100)
    
    predict_y = dann.localization_combined.predict(test_x_t)
    error = np.linalg.norm(predict_y - test_y_t, axis=1)    
    
    # =============================================================================    
    print_error_dist(error, 'DANN')
    
    # from plotters import error_dist
      
    # error_dist(ml_x_s, train_y_s, ml_x_t, train_y_t, error_metric,
    #            test_y_t,  title = title)
    # plt.show()
    
    # from plotters import plot_cdf
    # plot_cdf(error_metric, 100)    
    # plt.show()
    
    # del model
    # del model_obj
