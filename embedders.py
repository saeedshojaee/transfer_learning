from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

# =============================================================================
def pca():
  print("PCA projection is selected")
  embedder = decomposition.TruncatedSVD(n_components=n_components)
  return embedder
## =============================================================================
def isomap():
  print("Isomap embedding is selected")
  embedder = manifold.Isomap(n_neighbors, n_components=n_components)
  return embedder
## =============================================================================
def lle():
  print("LLE embedding is selected")
  embedder = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                          method='standard')
  return embedder
## =============================================================================
def ltsa():
  print("LTSA embedding is selected")
  embedder = manifold.LocallyLinearEmbedding(n_neighbors, n_components=n_components,
                                          method='ltsa')
  return embedder
## =============================================================================
def mds():
  print("MDS embedding is selected")
  embedder = manifold.MDS(n_components=n_components, n_init=1, max_iter=100)
  return embedder
## =============================================================================
# to be fixed
def random_tree():
  print("Totally Random Trees embedding is selected")
  embedder = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                           max_depth=5)
  X_s = embedder.fit_transform(X_s)
  X_t = embedder.fit_transform(X_t)
  X_s_val = embedder.fit_transform(X_s_val)
  X_test = embedder.fit_transform(X_test)
  embedder = decomposition.TruncatedSVD(n_components=n_components)
  return embedder
## =============================================================================
def spectral():
  print("Spectral embedding is selected")
  embedder = manifold.SpectralEmbedding(n_components = n_components,
                                        random_state = 0,
                                        eigen_solver = "arpack",
                                        n_neighbors = n_neighbors)
  return embedder
# =============================================================================
def tsne():
  print("t-SNE embedding is selected")
  embedder = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
  return embedder
# =============================================================================
def no_embedding():
  print("copy data to the embedder output")
  class no_embed:
    def fit_transform(x):
      return x
  embedder = no_embed
  return embedder
# =============================================================================
def autoencoder():
  print("autoencoder is selected")
  class ae:
    def fit_transform(x , val_x):
      from autoencoder import autoencoder
      emb_x, emb_val_x, _,_ = autoencoder(x, val_x, n_components = n_components )
      return emb_x, emb_val_x
  embedder = ae
  return embedder    
# =============================================================================
def embedding(argument1, n_cmp = 3, n_ngb = 30):
  switcher = {
    "pca" : pca,
    "isomap"  : isomap,
    "lle" : lle,
    "ltsa"  : ltsa,
    "mds"   : mds,
    "random_tree"   : random_tree,
    "spectral"  : spectral,
    "tsne"  : tsne,
    "no_embedding" : no_embedding,
    "autoencoder" : autoencoder ,
  }
  
  global n_components
  global n_neighbors
  n_components = n_cmp
  n_neighbors = n_ngb
  # Get the function from switcher dictionary
  func = switcher.get(argument1, "nothing")
  # Execute the function
  return func()

def variable_embedder(embedder, variables ):
  import numpy as np
  length = [0]
  embedding_vec = np.array([]).reshape(0,variables[1].shape[1])
  output = []
  for x in variables:
    embedding_vec = np.concatenate((embedding_vec, x), axis=0)
    length = length + [embedding_vec.shape[0]]
  from sklearn import preprocessing
  embedded_vec = preprocessing.scale(embedder.fit_transform(embedding_vec))
  output = []
  
  for i in range(len(length)-1):
    output = output + [embedded_vec[length[i]:length[i+1],:]]
  return output
    
#
#emb_x_s = preprocessing.scale(embedder.fit_transform(train_x_s))
#emb_x_t = preprocessing.scale(embedder.fit_transform(train_x_t))
#emb_val_x_s = preprocessing.scale(embedder.fit_transform(val_x_s))
#emb_val_x_t = preprocessing.scale(embedder.fit_transform(val_x_t))
#emb_test_x = preprocessing.scale(embedder.fit_transform(test_x_t))

