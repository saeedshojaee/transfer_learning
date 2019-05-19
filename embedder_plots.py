
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from load_data import load

train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
  val_x_t, val_y_t, test_x_t, test_y_t = load(unlabele_target_percentage = 1)

from plotters import cmap2d
cmap_s = cmap2d(train_y_s)
cmap_t = cmap2d(train_y_t)
plt.scatter(train_y_s[:,0],train_y_s[:,1], c = cmap_s,
              label  = 'source', marker =  "o", s = 5)
plt.show()


def embedd():
  ##############################################################################  
  from embedders import embedding, variable_embedder
  n_components = 5
  embedding_type = "spectral"
  embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 30)
  ##############################################################################
  
  if  embedding_type == "autoencoder":
    emb_train_x_s, emb_train_val_x_s, emb_train_x_t, emb_train_val_x_t\
      = variable_embedder(embedder,\
                          [train_x_s, train_x_t],[val_x_s, val_x_t])
  else:
    emb_train_x_s, emb_train_val_x_s, emb_train_x_t, emb_train_val_x_t\
      = variable_embedder(embedder,\
                          [train_x_s, val_x_s, train_x_t, val_x_t])
  
  
  # if  embedding_type == "autoencoder":
  #   emb_train_x_s, emb_train_val_x_s\
  #     = embedder.fit_transform(train_x_s, val_x_s) 
  # else:
  #   emb_train_x_s, emb_train_val_x_s\
  #     = variable_embedder(embedder, [train_x_s, val_x_s]) 
   
  
  plt.scatter(emb_train_x_s[:,0],emb_train_x_s[:,1], c = cmap_s,
                label  = 'source', marker =  "o", s = 5)
  plt.show()

  plt.scatter(emb_train_x_t[:,0],emb_train_x_t[:,1], c = cmap_t,
                label  = 'source', marker =  "o", s = 5)
  plt.show()
  
  
# if __name__= "__Main__"


embedd()
