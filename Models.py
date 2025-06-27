# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:58:17 2024

@author: op038
"""
import os
# Set the MKL instruction set before importing libraries that use MKL
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
from sklearn.linear_model   import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance
import numpy as np
import torch
import pandas as pd

import Params

########################################################################
#
#
#       Custom pytorch models
#
#
########################################################################

class LogisticRegressionTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=None):
       super().__init__()
       torch.manual_seed(random_state)
       self.linear = None
       self.l2_lambda = l2_lambda
       
   def _init_model(self, n_features): #this is so we can initilize the model without data
       if self.linear is None:
           self.linear = torch.nn.Linear(n_features, 1)
       
   def init_weights(self): # set initial weights and biases

        n_in = self.linear.weight.size(1)
        torch.nn.init.uniform_(self.linear.weight, -(1/n_in)**(1/2), (1/n_in)**(1/2)) # as sklearn
        torch.nn.init.zeros_(self.linear.bias)   
       
   def forward(self, x):
       return torch.sigmoid(self.linear(x))
   
   def fit(self, X, y, max_iter=1000, epochs=150, batch_size=32):
       weights_history = []

       if isinstance(X, pd.DataFrame):
           X = torch.FloatTensor(X.values)
       elif not isinstance(X, torch.Tensor):
           X = torch.FloatTensor(X)
        
       if isinstance(y, pd.Series):
           y = torch.FloatTensor(y.values)
       elif not isinstance(y, torch.Tensor):
           y = torch.FloatTensor(y)
    
       y = y.reshape(-1, 1)
       self._init_model(X.shape[1])
       self.init_weights()           
       criterion = torch.nn.BCELoss()
       
       
       #### LBFGS
       optimizer = torch.optim.LBFGS(self.parameters(), max_iter=max_iter)

       def closure():
           optimizer.zero_grad()
           output = self(X)
           loss = criterion(output, y)
           
           if self.l2_lambda is not None:
                l2_reg = torch.norm(self.linear.weight, 2)
                loss += self.l2_lambda * l2_reg 
                
           #### monitor weights     
           weights_history.append(self.linear.weight.detach().numpy())

           ####
           
           loss.backward()
           return loss
           
       optimizer.step(closure)
       return weights_history
       
   def predict_prob(self, X):
       if not isinstance(X, torch.Tensor):
           if hasattr(X, 'values'): # pd.DataFrames are not supported in torch
               X = X.values

       with torch.no_grad(): # disable gradient calculation for higher speed
           return self(torch.FloatTensor(X)).numpy()
           
   def predict(self, X):
       return (self.predict_prob(X) > 0.5).astype(int)
   
   def score(self, X, y):
       with torch.no_grad(): # disable gradient calculation for higher speed
           y_pred = self.predict(X)
           
           if isinstance(y, pd.Series):
               y = torch.FloatTensor(y.values)
           elif not isinstance(y, torch.Tensor):
               y = torch.FloatTensor(y)
        
           y = y.reshape(-1, 1)
           return np.mean(y_pred == np.array(y))


class MLPTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=0.01):
       super().__init__()
       
       random_state = torch.seed()
       self.layers = None
       self.l2_lambda = l2_lambda
       self.random_state = random_state

       
   def _init_model(self, n_features): #this is so we can initilize the model without data
       if self.layers is None:
            self.layers = torch.nn.Sequential( #Os.Ver1 architecture
                torch.nn.Linear(n_features, 20),  
                torch.nn.Tanh(),
                torch.nn.Linear(20, 10),          
                torch.nn.Tanh(),
                torch.nn.Linear(10, 5),           
                torch.nn.Tanh(),
                torch.nn.Linear(5, 1)            
            )
          
   def init_weights(self): # set initial weights and biases
       for layer in self.layers:
           if isinstance(layer, torch.nn.Linear):
                #n_in = layer.weight.size(1)
                #torch.nn.init.uniform_(layer.weight, -(1/n_in)**(1/2), (1/n_in)**(1/2)) # as sklearn
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)     
        
   def forward(self, x):
       return torch.sigmoid(self.layers(x))
       
   
   def fit(self, X, y, max_iter=4000, epochs=250):
        
        loss_history = []
        weights_history = []
        

        if isinstance(X, pd.DataFrame):
           X = torch.FloatTensor(X.values)
        elif not isinstance(X, torch.Tensor):
           X = torch.FloatTensor(X)
        
        if isinstance(y, pd.Series):
           y = torch.FloatTensor(y.values)
        elif not isinstance(y, torch.Tensor):
           y = torch.FloatTensor(y)
    
        y = y.reshape(-1, 1)
       
       
        self._init_model(X.shape[1])
        self.init_weights() 
        print("initialized weights")
                
        criterion = torch.nn.BCELoss()

        #optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=self.l2_lambda)

        #### LBFGS
        #optimizer = torch.optim.LBFGS(self.parameters(), max_iter=2000)
        # Detailed parameters from sklearn default
        optimizer = torch.optim.LBFGS(self.parameters(), 
                                max_iter=1000, #20
                                max_eval=500,  #50
                                tolerance_grad=1e-6,      
                                tolerance_change=1e-9,    
                                line_search_fn='strong_wolfe')
                
        def closure():
             optimizer.zero_grad()
             output = self(X)
             loss = criterion(output, y)
            
             if self.l2_lambda is not None:
                 for layer in self.layers:
                     if isinstance(layer, torch.nn.Linear):
                         l2_reg = torch.norm(layer.weight, 2)
                         loss += self.l2_lambda * l2_reg / X.shape[0] #sklearn divides by sample size

             weights_history.append(self.layers[-1].weight.detach().clone().squeeze().numpy())
             loss_history.append(loss)
             loss.backward()
            
             return loss
            
        optimizer.step(closure)
        return np.array(weights_history)

       
   def predict_prob(self, X):
       if not isinstance(X, torch.Tensor):
           if hasattr(X, 'values'): # pd.DataFrames are not supported in torch
               X = X.values

       with torch.no_grad(): # disable gradient calculation for higher speed
           return self(torch.FloatTensor(X)).numpy()
           
   def predict(self, X):
       return (self.predict_prob(X) > 0.5).astype(int)
   
   def score(self, X, y):
       with torch.no_grad(): # disable gradient calculation for higher speed
           y_pred = self.predict(X)
           
           if isinstance(y, pd.Series):
               y = torch.FloatTensor(y.values)
           elif not isinstance(y, torch.Tensor):
               y = torch.FloatTensor(y)
        
           y = y.reshape(-1, 1)
           return np.mean(y_pred == np.array(y))

   
########################################################################
#
#
#       Create model from its name
#
#
########################################################################

def CreateModel(ModelName, nIterMax = 5000):
    
    model = None
    
    
    
    if ModelName=="LReg":

        ###
        C = Params.C
        ###
        
        model = LogisticRegression(penalty='l2', solver = 'lbfgs', C = C, max_iter=nIterMax, n_jobs = -1, random_state=0)
        
        #sw_sum = n_samples
        #l2_reg_strength = 1.0 / (C * sw_sum)
        #C = 1.0 - default, C = 10 comparable with pytorch l2_lambda = 0.002
    elif ModelName=="LRegTorch":
        
        #l2_lambda=None
        l2_lambda=10/(1.0 * 100)
        model = LogisticRegressionTorch(random_state=0, l2_lambda=l2_lambda)
        if l2_lambda == None:
            print("\n Run without regularization")
        else:
            print("\n Strength of l2 regularization ", l2_lambda)
        
        
    elif ModelName=="LReg_Pen": # Penalized logistic regression
       model = LogisticRegression(C=100, max_iter=nIterMax, n_jobs = -1, random_state=0)
       
       
    elif ModelName=="MLP":
        model = MLPClassifier(hidden_layer_sizes = (20,10,5), activation = 'tanh', max_iter=nIterMax, alpha = 0, random_state=0)
   
    elif ModelName=="MLPTorch":
        print("init model")
        model = MLPTorch(random_state=0, l2_lambda=0.01)
        

        
    elif ModelName=="SVC":
        model = svm.LinearSVC(C=1, max_iter=nIterMax) # using linear for now, to support coef_
        
    
    else:
        raise Exception("Invalid model")
        
    return model

########################################################################
#
#
#       Get and return model state
#
#
########################################################################
def ModelState(bGet, model, intercept, state=None):
    # model_name = type(model).__name__
    # print(model_name)
    
    if hasattr(model, 'coef_'): #Logistic Regression sklearn
        if intercept == True:     
            if bGet:
                state = np.column_stack((model.coef_, model.intercept_))      
                return (state, model)
            else:
                model.coef_ = state[:, :-1].copy()
                model.intercept_ = state[:, -1].copy()
        else:
            if bGet:
                state = model.coef_.copy()
                return (state, model)
            else:
                model.coef_ = state.copy()
                           
    elif hasattr(model, 'coefs_'):   # 'MLPClassifier', 'MLPRegressor'
        
        if intercept == True:
            if bGet:                
                state = np.concatenate([w.flatten() for w in model.coefs_ + model.intercepts_])
            else:
                sizes = [w.size for w in model.coefs_ + model.intercepts_]
                splits = np.cumsum(sizes)[:-1]
                params = np.split(state, splits)
                
                n_layers = len(model.coefs_)
                model.coefs_ = [params[i].reshape(model.coefs_[i].shape) for i in range(n_layers)]
                model.intercepts_ = [params[i+n_layers].reshape(model.intercepts_[i].shape) for i in range(n_layers)]   
               
        else:
            #if bGet:
            #    state = model.coefs_
            #else:
            #    model.coefs_ = state.copy()
            if bGet:                
                state = np.concatenate([w.flatten() for w in model.coefs_])
            else:
                sizes = [w.size for w in model.coefs_]
                splits = np.cumsum(sizes)[:-1]
                params = np.split(state, splits)          
                model.coefs_ = [params[i].reshape(model.coefs_[i].shape) for i in range(len(model.coefs_))]
                 
    elif hasattr(model, 'layers'):  # MLP from PyTorch

           if bGet:
               weights_biases = []
               for layer in model.layers:
                   if isinstance(layer, torch.nn.Linear):
                       weights = layer.weight.data.numpy().T.flatten()
                       if intercept:
                           bias = layer.bias.data.numpy().flatten()
                           weights_biases.extend([weights, bias])
                       else:
                           weights_biases.append(weights)
        
               state = np.concatenate(weights_biases)
               return (state, model)
    
           else:  
        
                sizes = []
                shapes = []
                for layer in model.layers:
                    if isinstance(layer, torch.nn.Linear):
                        weights_shape = layer.weight.shape
                        shapes.append(weights_shape)
                        sizes.append(np.prod(weights_shape))
                        if intercept:
                            bias_shape = layer.bias.shape
                            shapes.append(bias_shape)
                            sizes.append(np.prod(bias_shape))
        
        
                splits = np.cumsum(sizes)[:-1]
                params = np.split(state, splits)
        
        
                param_idx = 0
                for layer in model.layers:
                    if isinstance(layer, torch.nn.Linear):
                        layer.weight.data = torch.FloatTensor(
                        params[param_idx].reshape(shapes[param_idx])
                        )
                        param_idx += 1
                        if intercept:
                            layer.bias.data = torch.FloatTensor(
                                params[param_idx].reshape(shapes[param_idx])
                                )
                            param_idx += 1
                    
           return (state, model)
    
    elif hasattr(model, 'linear'): # Logistic Regression from PyTorch
            
            if bGet:
                weights = model.linear.weight.data.numpy()
                bias = model.linear.bias.data.numpy()
                
                if intercept:
                    state = np.column_stack((weights, bias.reshape(-1,1)))
                else:
                    state = weights
                return (state, model)
            
            else:
                if intercept:
                    model.linear.weight.data = torch.FloatTensor(state[:, :-1])
                    model.linear.bias.data = torch.FloatTensor(state[:, -1])
                else:
                    model.linear.weight.data = torch.FloatTensor(state)
            return (state, model)
                        
                    
    else:
        raise Exception('Unknown model state atribute')
            
    # Return model and its state
    return (state, model)
    
# Short versions with Get/Set
def ModelStateGet(model, intercept):
    (state, _) = ModelState(True, model, intercept)
    return state

def ModelStateSet(model, intercept, state):
    (_, model) = ModelState(False, model, intercept, state)
    return model


########################################################################
#
#
#       Global optimization
#
########################################################################
def GlobalOptimization(model, Xeval, Yeval):
    
    # Get model state
    state = ModelStateGet(model)
    
    # Define model error funciton
    def ErrFunc(var):
        ModelStateSet(model, var)
        score = model.score(Xeval, Yeval)
        return (1-score)
    
    # Optimize model globally

########################################################################
#
#
#       Compute feature imporantce
#       ( https://scikit-learn.org/stable/modules/permutation_importance.html )
#
#
########################################################################
def GetFeatureImportance(model, X, Y):
    
    # Initialize feature importance list
    fi = []
    
    # For regression-like models, use their coefficients
    if hasattr(model, 'coeff_'):
        for i in range(len(model.coeff_)):
            fi.append( (i, model.coeff_[i]) )
            
    else: # use permutation importance, 
        res = permutation_importance(model, X, Y, n_repeats=50, n_jobs = -1, 
                                     max_samples = 500, random_state=0)
        for i in range( len(res.importances_mean) ):
            # if abs(res.importances_mean[i]) - 2 * res.importances_std[i] < 0:
            #     continue # consider non-significant
            fi.append( (i,res.importances_mean[i]) )
       
    # Exit if no important features found
    if len(fi) < 1:
        return (-1,-1)
    
    # Find the most important feature index and importance value
    fi.sort(key=lambda tup: -tup[0])
    
    return fi[0] # best feature index and its importance
    

########################################################################
#
#
#       Any test runs here
#
#
########################################################################
# model = RidgeClassifier()

# ModelStateGet(model)

