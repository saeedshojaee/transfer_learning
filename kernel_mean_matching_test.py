# kernel mean matching test

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n = 1000

source = np.random.randn(n,2) * 10 + [0, 0]
target = np.random.rand(n,2) * 5 + [1, 2]

def test_kmm(source, target):
  
  kmm_kernel = 'rbf'
  B = 1000
  print('Kernel mean matching')
  # from kernel_mean_matching import eprimical_kmm as ekmm
  # coef_s,_  =  ekmm(target, source, kern = kmm_kernel, B = B)
  
  from kernel_mean_matching import kernel_mean_matching as kmm
  coef_s  =  kmm(target, source, kern = kmm_kernel, B = B)
  
  
  tr = coef_s>0.1
  tr = tr.reshape(tr.shape[0],)
  
  
  
  from embedders import embedding, variable_embedder
  n_components = 2
  embedding_type = "no_embedding"
  embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
  if embedding_type == "autoencoder":
    split = 0.3
    cut = np.floor(target.shape[0] * (1- split) ).astype(int)
    test_X = target[:cut,:]
    val_X  = target[cut:, :]
    cut = np.floor(x_s.shape[0] * (1 - split) ).astype(int)
    test_C = source[:cut,:]
    val_C  = source[cut:, :]
    emb_c, emb_val_c, emb_x, emb_val_x = variable_embedder(embedder,\
                        [test_C, test_X],[val_C, val_X])
    source = np.concatenate((emb_c, emb_val_c), axis = 0)
    target = np.concatenate((emb_x, emb_val_x), axis = 0)
  else:
    source, target, = variable_embedder(embedder,[source, target])
  
  marker_size = 5
  l1, = plt.plot(source[:,0],source[:,1], 'o', color = 'red',
              label  = 'source')
  plt.setp(l1, markersize=marker_size)
  
  
  l2, = plt.plot(target[:,0],target[:,1], 'o', color = 'blue',
              label  = 'target')
  plt.setp(l2, markersize=marker_size)
  
  marker_size = 2
  l1, = plt.plot(source[:,0],source[:,1], 'o', color = 'red',
              label  = 'source')
  plt.setp(l1, markersize=marker_size)
  marker_size = 1
  l3, = plt.plot(source[tr,0],source[tr,1], 'o', color = 'k',
              label  = 'ssbc')
  plt.setp(l3, markersize=marker_size)
  
  plt.show()

##############################################################################
##############################################################################

test_kmm(source, target)

from sklearn import preprocessing
source = preprocessing.scale(source)
target = preprocessing.scale(target)

test_kmm(source, target)

##############################################################################
n = 1000

source = np.random.randn(n,3) * 10 + [0, 0, 0]
target = np.random.rand(n,3) * 5 + [1, 2, 5]

test_kmm(source, target)










##############################################################################

from load_data import load
train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
  val_x_t, val_y_t, test_x_t, test_y_t = load(unlabele_target_percentage = 1)
  
x_s = np.concatenate((train_x_s, val_x_s), axis = 0)
x_t = np.concatenate((train_x_s, val_x_t), axis = 0)
print('Kernel mean matching')
# =============================================================================
kmm_kernel = 'rbf'
B = 1000
# =============================================================================
x_s = np.concatenate((train_x_s, val_x_s), axis = 0)
x_t = np.concatenate((train_x_s, val_x_t), axis = 0)

from kernel_mean_matching import eprimical_kmm as ekmm
ssbc_coef_s, ssbc_coef_t =  ekmm(x_t, x_s, kern = kmm_kernel, B = B)

treshhold = 1

ssbc_coef_val_s = ssbc_coef_s[train_y_s.shape[0]:]
ssbc_coef_s = ssbc_coef_s[:train_y_s.shape[0]]
tr_ssbc = ssbc_coef_s > treshhold
tr_ssbc = tr_ssbc.reshape(tr_ssbc.shape[0],)



from embedders import embedding, variable_embedder
n_components = 2
embedding_type = "tsne"
embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
##############################################################################
emb_train_x_s, emb_train_val_x_s, emb_train_x_t, emb_train_val_x_t\
 = variable_embedder(embedder,\
                     [train_x_s, val_x_s, train_x_t, val_x_t]) 


marker_size = 5;

l00 = plt.plot(emb_train_x_s[:,0],emb_train_x_s[:,1], 'o', color = scalarMap.to_rgba(0),
            label  = 'source')
plt.setp(l00, markersize=marker_size)

l01 = plt.plot(emb_train_x_t[:,0],emb_train_x_t[:,1], 'o', color = scalarMap.to_rgba(1),
            label  = 'target')
plt.setp(l01, markersize=marker_size-3)

l22, = plt.plot(emb_train_x_s[tr_ssbc,0],emb_train_x_s[tr_ssbc,1],'o', color = 'k',
            label  = 'ssbc')
plt.setp(l22, markersize=1)
plt.show()

