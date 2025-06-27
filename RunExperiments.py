#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:19:20 2024

@author: rita
"""

import numpy as np
import warnings

import Experiments
import Params
#import LoadOSData


DataNames = ["WisBreast_Orth"]   
ModelNames = ['MLPTorch'] 

#regularization is for LReg
C = Params.C

Experiments.ComputeExperiments(DataNames, ModelNames, intercept = False)
    
    
DataNames = ["WisBreast_Orth"]   
ModelNames = ['MLP'] 

#regularization is for LReg
C = Params.C

Experiments.ComputeExperiments(DataNames, ModelNames, intercept = False)
        


# =============================================================================
# All possible models
# =============================================================================


#ModelNames = ['LReg', 'LRegTorch', 'MLP', 'MLPTorch'] 
#DataNames = ['WisBreast', 'Census_Orth' ]

