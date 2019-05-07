
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from load_data import load

train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
  val_x_t, val_y_t, test_x_t, test_y_t = load()
  
  
########### metric learning
from metric_learning import hisc_meme

X_s = np.concatenate((train_x_s, val_x_s), axis = 0)
X_t = np.concatenate((train_x_t, val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)

print('Finding the metric')
M , L = hisc_meme(X_s, Y_s, X_t, Y_t, p = 2, B = 1000,
                  threshhold = 1)

ml_x_s = train_x_s @ L
ml_val_x_s = val_x_s @ L

ml_x_t = train_x_t @ L
ml_val_x_t = val_x_t @ L

###### t/ssbc
print('Finding the transformation')

from metric_learning import hisc_meme_no_labeled_data

M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000,
                  threshhold = 1)

t_ssbc_x_s = train_x_s @ L
t_ssbc_val_x_s = val_x_s @ L

t_ssbc_x_t = train_x_t @ L
t_ssbc_val_x_t = val_x_t @ L

###########################################################
from embedders import embedding
n_components = 2
embedding_type = "tsne"
embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
##############################################################################

print("Computing t-SNE embedding")
emb_train_x_s = embedder.fit_transform(train_x_s)
emb_train_x_t = embedder.fit_transform(train_x_t)

emb_train_val_x_s = embedder.fit_transform(val_x_s)
emb_train_val_x_t = embedder.fit_transform(val_x_t)
##############################################################################
emb_ml_x_s = embedder.fit_transform(ml_x_s)
emb_ml_x_t = embedder.fit_transform(ml_x_t)

emb_ml_val_x_s = embedder.fit_transform(ml_val_x_s)
emb_ml_val_x_t = embedder.fit_transform(ml_val_x_t)
##############################################################################
emb_t_ssbc_x_s = embedder.fit_transform(t_ssbc_x_s)
emb_t_ssbc_x_t = embedder.fit_transform(t_ssbc_x_t)

emb_t_ssbc_val_x_s = embedder.fit_transform(t_ssbc_val_x_s)
emb_t_ssbc_val_x_t = embedder.fit_transform(t_ssbc_val_x_t)

##############################################################################
print("Done")
set2 = cm = plt.get_cmap('Set1') 
cNorm  = colors.Normalize(vmin=0, vmax=6)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=set2)
colorVal = scalarMap.to_rgba(1)
import seaborn as sns
sns.set_style("darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(emb_train_x_s[:,0],emb_train_x_s[:,1], color = scalarMap.to_rgba(0),
            label  = 'source', s=20)
plt.scatter(emb_train_x_t[:,0],emb_train_x_t[:,1], color = scalarMap.to_rgba(1),
            label  = 'target', s=20)

plt.scatter(emb_ml_x_s[:,0],emb_ml_x_s[:,1], color = scalarMap.to_rgba(2),
            label  = 'ML source', s=20)
plt.scatter(emb_ml_x_t[:,0],emb_ml_x_t[:,1], color = scalarMap.to_rgba(3),
            label  = 'ML target', s=20)

plt.scatter(emb_t_ssbc_x_s[:,0],emb_t_ssbc_x_s[:,1], color = scalarMap.to_rgba(4),
            label  = 'T source', s=20)
plt.scatter(emb_t_ssbc_x_t[:,0],emb_t_ssbc_x_t[:,1], color = scalarMap.to_rgba(5),
            label  = 'T target', s=20)

plt.xlim(-60,90)
plt.ylim(-30,30)

# glt.grid("")
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper right')

plt.show()

