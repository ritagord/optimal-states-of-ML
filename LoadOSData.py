# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:19:06 2024

@author: op038
"""
import os
# Set the MKL instruction set before importing libraries that use MKL
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
import pandas as pd
from sklearn.datasets import load_iris
from scipy.linalg import orth


###########################################################################
#
#   Main function to load data
#
###########################################################################
bCollinear = False
bFilterData = False
        

def LoadOSData(project_name, bReload=False):
      
    # Load data from different sets
    if project_name.startswith("WisBreast"): 
        try:
            data = pd.read_excel('./InputData/WisconsinBreast.xlsx') 
        except FileNotFoundError:
            data = pd.read_excel('./WisconsinBreast.xlsx') 
        data.drop('id', axis=1, inplace=True)
        data.rename(columns={"diagnosis": "Target"}, inplace=True)
        data['Target'] = (data['Target']=='M')
        # Return (X,Y) data
        Y = data['Target'].copy()
        X = data.copy(); X.drop('Target', axis=1, inplace=True)
        if bFilterData:
            numeric_cols = ['radius_se', 'smoothness_mean', 'texture_se', 'fractal_dimension_worst']
            X = X[numeric_cols].copy()
            print(X.corr())

            
    elif project_name.startswith("DivideBy30"):
        data = pd.read_excel('./InputData/DivideBy30.xlsx')
        data.rename(columns={"Div By 30": "Target"}, inplace=True)
        # Return (X,Y) data
        Y = data['Target'].copy()
        X = data.copy(); X.drop('Target', axis=1, inplace=True)
      
    elif project_name.startswith("Iris"): 
        X, Y = load_iris(return_X_y=True, as_frame = True)
        Y = (Y<1).astype(int) # conversion to binary
        # Return (X,Y) data
        Y = data['Target'].copy()
        X = data.copy(); X.drop('Target', axis=1, inplace=True)
        return (X, Y)
    
    elif project_name.startswith("Census"): 
        try: 
            data = pd.read_csv('./InputData/Census_Income.csv') 
        except FileNotFoundError:
            data = pd.read_csv('./Census_Income.csv') 
        filtered_data = data[
        (data['age'].astype(int) > 16) & 
        (data['capital-gain'].astype(int) > 100) & 
        (data['fnlwgt'].astype(int) > 1) & 
        (data['hours-per-week'].astype(int) > 0)
        ]
        filtered_data = filtered_data.dropna()
        numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week']
        data_numeric = filtered_data[numeric_cols].copy()
        X = data_numeric.copy()  
        
        income_map = {
           '<=50K': 0,
           '<=50K.': 0,
           '>50K': 1,
           '>50K.': 1
           }
        data_numeric['income'] = filtered_data['income'].map(income_map)
        
        # Return (X,Y) data
        Y = data_numeric['income'].copy()
      

    
    # Orthogonalize if required
    if "_Orth" in project_name: 
        X = pd.DataFrame(orth(X))
        #print(X.head())
        if bCollinear:
            X.iloc[:, 1] = - 2 * X.iloc[:, 2]
        else:
            X.iloc[:, 1] = X.iloc[:, 2]

        
        
    return (X, Y)






###########################################################################
#
#   Tests
#
###########################################################################
# X,Y = LoadOSData("Wisbreast")
      