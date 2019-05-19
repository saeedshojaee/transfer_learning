
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from load_data import load
from kernel_mean_matching import eprimical_kmm_emb as ekmm_emb
from kernel_mean_matching import kernel_mean_matching as kmm

train_x_s, train_y_s, val_x_s, val_y_s, train_x_t, train_y_t, \
  val_x_t, val_y_t, test_x_t, test_y_t = load(unlabele_target_percentage = 1)


##############################################################################  
from embedders import embedding, variable_embedder
n_components = 2
embedding_type = "no_embedding"
embedder = embedding(embedding_type, n_cmp = n_components, n_ngb = 10)
##############################################################################  
if embedding_type == "autoencoder":
  train_x_s, val_x_s, train_x_t, val_x_t\
    = variable_embedder(embedder,\
                        [train_x_s, train_x_t],[val_x_s, val_x_t])
  test_x_t = embedder.predict(test_x_t)
else:
  train_x_s, val_x_s, train_x_t, val_x_t, test_x_t \
  = variable_embedder(embedder,[train_x_s, val_x_s,
                                train_x_t, val_x_t, test_x_t])
##############################################################################
treshhold = 0.5
##############################################################################  
###### ssbc
print('Kernel mean matching')
# =============================================================================
kmm_kernel = 'rbf'
B = 10
# =============================================================================
x_s = np.concatenate((train_x_s, val_x_s), axis = 0)
x_t = np.concatenate((train_x_s, val_x_t), axis = 0)

# from kernel_mean_matching import eprimical_kmm as ekmm
# ssbc_coef_s, ssbc_coef_t =  ekmm(x_t, x_s, kern = kmm_kernel, B = B)

ssbc_coef_s, ssbc_coef_t =  ekmm_emb(x_t, x_s, kern = kmm_kernel, B = B,
                            embedder_type = 'autoencoder', n_components = 10)

# x_s = np.concatenate((train_x_s, emb_train_val_x_s), axis = 0)
# x_t = np.concatenate((train_x_t, emb_train_val_x_t), axis = 0)
# ssbc_coef_s =  kmm(x_t, x_s, kern = kmm_kernel, B = B, sigma = 10.0)


ssbc_coef_val_s = ssbc_coef_s[train_y_s.shape[0]:]
ssbc_coef_s = ssbc_coef_s[:train_y_s.shape[0]]

##############################################################################  
# from plotters import plot_embeding
# plot_embeding(train_x_s, train_x_t, ssbc_coef_s, train_y_s, train_y_t)
##############################################################################  
########### metric learning
from metric_learning import hisc_meme

X_s = np.concatenate((train_x_s, val_x_s), axis = 0)
X_t = np.concatenate((train_x_t, val_x_t), axis = 0)
Y_s =  np.concatenate((train_y_s, val_y_s), axis = 0)
Y_t =  np.concatenate((train_y_t, val_y_t), axis = 0)

##############################################################################
print('Finding the metric')
M , L = hisc_meme(X_s, Y_s, X_t, Y_t, p = 2, B = 1000,
                  threshhold = 1)

ml_x_s = train_x_s @ L
ml_val_x_s = val_x_s @ L

ml_x_t = train_x_t @ L
ml_val_x_t = val_x_t @ L
print('Done')

##############################################################################
print('Kernel mean matching')
ml_x_s_tmp = np.concatenate((ml_x_s, ml_val_x_s), axis = 0)
ml_x_t_tmp = np.concatenate((ml_x_t, ml_val_x_t), axis = 0)

kmm_kernel = 'rbf'
B = 10
ml_coef_s, ml_coef_t =  ekmm_emb(ml_x_t_tmp, ml_x_s_tmp, kern = kmm_kernel, B = B,
                           embedder_type = 'autoencoder', n_components = 10)
print('Done')
# =============================================================================
ml_coef_val_s = ml_coef_s[ml_x_s.shape[0]:]
ml_coef_s = ml_coef_s[:ml_x_s.shape[0]]
##############################################################################
from plotters import plot_embeding
# plot_embeding(ml_x_s, ml_x_t, ml_coef_s, train_y_s, train_y_t)
##############################################################################
###### t/ssbc
print('Finding the transformation')
##############################################################################
from metric_learning import hisc_meme_no_labeled_data

M , L = hisc_meme_no_labeled_data(X_s, Y_s, X_t, p = 2, B = 1000,
                  threshhold = 1)

t_ssbc_x_s = train_x_s @ L
t_ssbc_val_x_s = val_x_s @ L

t_ssbc_x_t = train_x_t @ L
t_ssbc_val_x_t = val_x_t @ L


t_ssbc_x_s_tmp = np.concatenate((t_ssbc_x_s, t_ssbc_val_x_s), axis = 0)
t_ssbc_x_t_tmp = np.concatenate((t_ssbc_x_t, t_ssbc_val_x_t), axis = 0)

kmm_kernel = 'rbf'
B = 10
coef_s, coef_t =  ekmm_emb(t_ssbc_x_t_tmp, t_ssbc_x_s_tmp, kern = kmm_kernel, B = B,
                            embedder_type = 'autoencoder', n_components = 10)

# coef_s = kmm(t_ssbc_x_t_tmp, t_ssbc_x_s_tmp, kern = kmm_kernel, B = B, sigma = 10.0)

print('Done')
# =============================================================================
coef_val_s = coef_s[t_ssbc_x_s.shape[0]:]
coef_s = coef_s[:t_ssbc_x_s.shape[0]]

from plotters import plot_embeding
plot_embeding(train_x_s, train_x_t, ssbc_coef_s, train_y_s, train_y_t, fig_name = 'ssbc')

plot_embeding(ml_x_s, ml_x_t, ml_coef_s, train_y_s, train_y_t, fig_name = 'ml')

plot_embeding(t_ssbc_x_s, t_ssbc_x_t, coef_s, train_y_s, train_y_t, fig_name = 't-ssbc')