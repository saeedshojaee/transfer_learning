import os
import sys
import numpy as np
import numpy.matlib as npm
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def eprimical_kmm(target_samples, source_samples, kern = 'rbf', B = 1):
  X = target_samples
  Z = np.concatenate((source_samples, target_samples), axis=0)
  coef = kernel_mean_matching(X, Z, kern=kern, B=B)
  coef_s = coef[0:source_samples.shape[0]]
  coef_t = coef[source_samples.shape[0]:]
  return coef_s, coef_t

def eprimical_kmm_emb(target_samples, source_samples, kern = 'rbf', B = 1,
                      embedder_type = 'autoencoder', n_components = 30):
  from embedders import embedding, variable_embedder
  embedder = embedding(embedder_type, n_cmp = n_components, n_ngb = 10)
  if embedder_type != "autoencoder":
    X, C = variable_embedder(embedder, [target_samples, source_samples])
  else:  
    from autoencoder import autoencoder
    split = 0.3
    cut = np.floor(target_samples.shape[0] * (1- split) ).astype(int)
    test_X = target_samples[:cut,:]
    val_X  = target_samples[cut:, :]
    cut = np.floor(source_samples.shape[0] * (1 - split) ).astype(int)
    test_C = source_samples[:cut,:]
    val_C  = source_samples[cut:, :]
    test_X, val_X, test_C, val_C\
    = variable_embedder(embedder,\
                        [test_X, test_C],[val_X, val_C])

    X = np.concatenate((test_X, val_X), axis = 0)
    C = np.concatenate((test_C, val_C), axis = 0)
   
# =============================================================================
#   from embedders import variable_embedder
#   embeded_output = variable_embedder(embedder, (source_samples, target_samples))
#   C = embeded_output[0]
#   X = embeded_output[1]
# =============================================================================
  if embedder_type == 'tsne' or embedder_type == "autoencoder":
    X = X.astype(np.double)
    C = C.astype(np.double)
  
  coef_s = kernel_mean_matching(X, C, kern=kern, B=B)
  coef_t = []
  
  # coef_s, coef_t = eprimical_kmm(X, C , kern=kern, B=B)
  
  return coef_s, coef_t


def eprimical_kmm_difference(target_samples, source_samples, kern = 'rbf', B = 1):
  
  X = np.diff(target_samples,axis = 1)
  Z = np.concatenate((np.diff(source_samples,axis = 1)
                      , np.diff(target_samples,axis = 1)),
                      axis=0)
  coef = kernel_mean_matching(X, Z, kern=kern, B=B)
  coef_s = coef[0:source_samples.shape[0]]
  coef_t = coef[source_samples.shape[0]:]
  return coef_s, coef_t

# an implementation of Kernel Mean Matchin
# referenres:
#  1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
#  2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
def kernel_mean_matching(X, Z, kern='lin', B=1.0, eps=None, sigma = 1.0):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B/math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T)*float(nz)/float(nx),axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z,Z, sigma = sigma)
        kappa = np.sum(compute_rbf(Z,X),axis=1)*float(nz)/float(nx)
    else:
        raise ValueError('unknown kernel')
        
    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    sol = solvers.qp(K, -kappa, G, h, kktsolver="ldl")
#    print(sol['x'])
    coef = np.array(sol['x'])
    return coef

def compute_rbf(X, Z, sigma=1.0 ):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K
