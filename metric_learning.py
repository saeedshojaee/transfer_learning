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
  lu, d, perm = scipy.linalg.ldl(M, lower = 1, hermitian=False)
  d_p = np.diag(d)
  d_p.setflags(write = 1)
  d_p[d_p<0.0001] = 0
  d_half = d_p**0.5
  idx = d_half != 0
  diag_mat = np.diag(d_half)
  L = lu @ diag_mat[:,idx] 
      
  # beta = eig_vectors[:,:p] 
  # L = beta
  # M = L.T @ L
  
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
  
  # K_y = np.zeros((n,n))
  # K_y = np.ones((n,n))
  K_y = np.eye(n)
  
  K_y[0:n_s ,0:n_s] = K_y_s
  
  
  # K_y[n_s: ,n_s:] = compute_rbf(X_t, X_t, sigma=1000)
  # K_y[:n_s ,n_s:] = compute_rbf(X_s, X_t, sigma=1000)
  # K_y[n_s: ,:n_s] = compute_rbf(X_t, X_s, sigma=1000)
  
  # centering matrix
  H = np.eye(n) - 1/n * np.ones((n,n))
  # K_h =  H @ K_y @ H
  # A = X.T @  H @ K_y @ H @ X
  A = X.T @ H @ K_y @ H @ X
 
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
  d_p[d_p<0.0001] = 0
  # sort_idx = np.argsort(d_p)
  # d_p_tmp = d_p[sort_idx]
  # d_p_tmp[:-p] = 0
  # d_p[sort_idx] = d_p_tmp 
  d_half = d_p**0.5
  idx = d_half != 0
  diag_mat = np.diag(d_half)
  L = lu @ diag_mat[:,idx] 
  
  
  
# =============================================================================
#   beta = eig_vectors[:,:p] 
#   L = X.T @ beta
#   M = L.T @ L
# =============================================================================
  
  
  
  return M.real, L.real

def hisc_meme_no_loc(X_s, X_t, p = 3, B = 1 , threshhold = 1):
  X = np.concatenate((X_s, X_t), axis = 0)
  n = X.shape[0]
  n_t = X_t.shape[0]
  n_s = X_s.shape[0]
  
# =============================================================================
#   from kernel_mean_matching import compute_rbf
#   K_y_s = compute_rbf(Y_s, Y_s, sigma=1000)
#   r = 2
#   K_y_s = r * K_y_s - r/2
#   
#   K_y = np.ones((n,n))
#   K_y[0:n_s ,0:n_s] = K_y_s
# 
# =============================================================================
  K_y = np.eye(n)

  
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
  d_p[d_p<0.0001] = 0
  d_half = d_p**0.5
  idx = d_half != 0
  diag_mat = np.diag(d_half)
  L = lu @ diag_mat[:,idx] 
  return M.real, L.real


def kernel_hisc_meme(X_s, Y_s , X_t, Y_t , p = 3, B = 1 , threshhold = 1):
  X = np.concatenate((X_s, X_t), axis = 0)
  Y = np.concatenate((Y_s, Y_t), axis = 0)
  return HSIC(X, Y, p = p, B = B )

def HSIC(X, Y, p = 3, B = 1 ):
  m, n = X.shape
  #  from sklearn.metrics import pairwise_distances
  #  dist = pairwise_distances(Y, metric='euclidean')
  #  K = kernel(dist, threshhold)
  #  K = np.dot(Y, Y.T)
  from kernel_mean_matching import compute_rbf
  K_y = compute_rbf(Y, Y, sigma=10)
  r = 2
  K_y = r*K_y - r/2

  # centering matrix
  H = np.eye(m) - 1/n * np.ones((m,m))
  # K_h =  H @ K @ H
  K_x = compute_rbf(X,X, sigma = 100)
  # K_x = X @ X.T
  A = K_x @ H @ K_y @ H @ K_x
  eig_values, eig_vectors = np.linalg.eig(A) 
  eig_values[eig_values < 0] = 0
  eig_values[p:] = 0
  if p == 1:
    A_1 = eig_values[0] * eig_vectors @ eig_vectors.T
    M_tilde = B * A_1
  else:
    tr_p = trace_p(eig_values, eig_vectors, p = p/(p-1))
    A_p_1 = eig_vectors @ np.diag(eig_values ** (1/(p-1))) @ eig_vectors.T
    M_tilde = (B / tr_p) ** (1/p) * A_p_1
    
  import scipy
  
  M = X.T @ M_tilde @ X
  
  lu, d, perm = scipy.linalg.ldl(M, lower = 0, hermitian=False)
  d_p = np.diag(d)
  d_p.setflags(write = 1)
  d_p[d_p<0.0001] = 0
  d_half = d_p**0.5
  idx = d_half != 0
  diag_mat = np.diag(d_half)
  L = lu @ diag_mat[:,idx] 
 
  return  M.real, L.real

def HSIC2(X, Y, p = 3, B = 1 ):
  n = X.shape[0]
  #  from sklearn.metrics import pairwise_distances
  #  dist = pairwise_distances(Y, metric='euclidean')
  #  K = kernel(dist, threshhold)
  #  K = np.dot(Y, Y.T)
  from kernel_mean_matching import compute_rbf
  K_y = compute_rbf(Y, Y, sigma=1)
  # r = 2
  # K_y = r*K_y - r/2

  # centering matrix
  H = np.eye(n) - 1/n * np.ones((n,n))
  # K_h =  H @ K @ H
  # K_x = compute_rbf(X,X, sigma = 1)
  # K_x = X @ X.T
  A = X.T @ H @ K_y @ H @ X
  eig_values, eig_vectors = np.linalg.eig(A) 
  eig_values[eig_values < 0] = 0
  eig_values[p:] = 0
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
  d_p[d_p<0.0001] = 0
  d_half = d_p**0.5
  idx = d_half != 0
  diag_mat = np.diag(d_half)
  L = lu @ diag_mat[:,idx] 
 
  return  M.real, L.real

def HSIC3(X, Y, p = 3, B = 1 ):
  n = Y.shape[1]
  m = X.shape[1]
  
  # if n < m :
  #   import numpy as np
  #   idx = np.random.randint(0, m-1 , n)
  #   X = X[idx,:]
  # else:
  #   idx = np.random.randint(0, n-1 , m)
  #   Y = Y[idx,:]
  #   n = m 
    
  
  from kernel_mean_matching import compute_rbf
  K_y = compute_rbf(Y.T, Y.T, sigma=1)
  
  K_x = compute_rbf(X.T, X.T, sigma=1)

  # K_y = Y @ Y.T
  # centering matrix
  H = np.eye(n) - 1/n * np.ones((n,n))

  A = H @ K_y @ H @ K_x
  eig_values, eig_vectors = np.linalg.eig(A) 
  eig_values[eig_values < 0] = 0
  sort_idx = np.argsort(eig_values)
  d_p = eig_values[sort_idx]
  B = eig_vectors[:,sort_idx]
  # L = L[:,::-1]
  L = B
  # eig_values[p:] = 0
  # if p == 1:
  #   A_1 = eig_values[0] * eig_vectors @ eig_vectors.T
  #   M = B * A_1
  # else:
  #   tr_p = trace_p(eig_values, eig_vectors, p = p/(p-1))
  #   A_p_1 = eig_vectors @ np.diag(eig_values ** (1/(p-1))) @ eig_vectors.T
  #   M = (B / tr_p) ** (1/p) * A_p_1
    
  # import scipy
   
  # lu, d, perm = scipy.linalg.ldl(M, lower = 0, hermitian=False)
  # d_p = np.diag(d)
  # d_p.setflags(write = 1)
  # # d_p[d_p<0.0001] = 0
  # d_half = d_p**0.5
  # idx = d_half != 0
  # diag_mat = np.diag(d_half)
  # L = lu @ diag_mat 
 
  return  L.real

