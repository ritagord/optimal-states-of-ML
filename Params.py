#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:49:34 2024

@author: rita
"""

bRecompute = True # Recompute all numerical experiments
bFullRun = False # Full or small (faster) run
bPartFit = False # Only works for MLP

C = 10


 # l2 regularization level for LReg in sklearn

ModelCloudMaxHours = 5.0 # maximum hours to build model cloud
Resol = 500  # image resolution
Scale = 100.0  # Zooming in/out
nIterMax = 5000 # maximum number of model iterations to converge
nInitialModelCount = 1_000_000 # initial model cloud size

# Parameters for faster run
if bFullRun == False:
    Resol = 150 #50 or 150
    Scale = 10
    nInitialModelCount = 100 #5000
    ModelCloudMaxHours = 5.0/60 #2.0/60
    
    