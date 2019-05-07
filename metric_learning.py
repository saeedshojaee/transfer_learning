import numpy as np 
def hisc_meme(X_s, Y_s , X_t, Y_t , p = 3, B = 1 , threshhold = 1):
  X = np.concatenate((X_s, X_t), axis = 0)
  n = X.shape[0]
  Y = np.concatenate((Y_s, Y_t), axis = 0)
  
#  from sklearn.metrics import pairwise_distances
#  dist = pairwise_distances(Y, metric='euclidean')
#  K = kernel(dist, threshhold)
  
#  K = np.dot(Y, Y.T)
  
  from kernel_mean_matching import compute_rbf
  K_y = compute_rbf(Y, Y, sigma=1000)
  r = 2
  K_y = r*K_y - r/2

  # centering matrix
  H = np.eye(n) - 1/n * np.ones((n,n))
  # K_h =  H @ K @ H
  A = X.T @  H @ K_y @ H @ X
  
#  A = X.T @ X
 
  eig_values, eig_vectors = np.linalg.eig(A) 
  eig_values[eig_values < 0] = 0
  

  if p == 1:
    A_1 = eig_values[0] * eig_vectors @ eig_vectors.T
    M = B * A_1
  else:
    tr_p = trace_p(eig_values, eig_vectors, p = p/(p-1))
    A_p_1 = eig_vectors @ np.diag(eig_values ** (1/(p-1))) @ eig_vectors.T
    M = (B / tr_p) ** (1/p) * A_p_1
  
  import scipy
  lu, d, perm = scipy.linalg.ldl(M, lower = 0, hermitian=False)
  d_p = np.diag(d)
  d_p.setflags(write = 1)
  d_p[d_p<0] = 0
  d_half = d_p**0.5
  
  L = lu @ np.diag(d_half)
  return M.real, L.real


def kernel(dist, threshhold):
  K = np.ones(dist.shape)
  K[dist > threshhold] = -1
  return K
  
def trace_p(eig_values , eig_vectors, p = 1):
  delta_p = eig_values ** p 
  tr_p = np.sum(delta_p)
  return tr_p



def hisc_meme_no_labeled_data(X_s, Y_s , X_t, p = 3, B = 1 , threshhold = 1):
  X = np.concatenate((X_s, X_t), axis = 0)
  n = X.shape[0]
  n_t = X_t.shape[0]
  n_s = X_s.shape[0]
  
  from kernel_mean_matching import compute_rbf
  K_y_s = compute_rbf(Y_s, Y_s, sigma=1000)
  r = 2
  K_y_s = r * K_y_s - r/2
  
  K_y = np.ones((n,n))
  K_y[0:n_s ,0:n_s] = K_y_s
  
  # centering matrix
  H = np.eye(n) - 1/n * np.ones((n,n))
  # K_h =  H @ K_y @ H
  A = X.T @  H @ K_y @ H @ X
  
#  A = X.T @ X
 
  eig_values, eig_vectors = np.linalg.eig(A) 
  eig_values[eig_values < 0] = 0
  

  if p == 1:
    A_1 = eig_values[0] * eig_vectors @ eig_vectors.T
    M = B * A_1
  else:
    tr_p = trace_p(eig_values, eig_vectors, p = p/(p-1))
    A_p_1 = eig_vectors @ np.diag(eig_values ** (1/(p-1))) @ eig_vectors.T
    M = (B / tr_p) ** (1/p) * A_p_1
  
  import scipy
  lu, d, perm = scipy.linalg.ldl(M, lower = 0, hermitian=False)
  d_p = np.diag(d)
  d_p.setflags(write = 1)
  d_p[d_p<0] = 0
  d_half = d_p**0.5
  
  L = lu @ np.diag(d_half)
  return M.real, L.real


