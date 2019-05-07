#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:30:30 2019

@author: shojaee
"""
from kernel_mean_matching import kernel_mean_matching as kmm
import numpy as np


def db2pow(x):
  x = 10**(x/10)
  return x

def db2mag(x):
  x = 10**(x/20)
  return x

def pow2db(x):
  x = 10 * np.log10(x)
  return x

def mag2db(x):
  x = 20 * np.log10(x)
  return x
  
def sum_power(X):
  X = db2mag(X) 
   
  rssi = np.matmul(X, X.T)
  return pow2db(rssi)
  