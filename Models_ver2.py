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
#       Custom pytorch models that are capable of weight tracking
#
#
########################################################################

class MLPTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=0.01, manual_init=False, manual_weights=None, loss_monitoring = False, optimizer_type = "Adam"):
       super().__init__()
       
       random_state = torch.seed()
       self.layers = None
       self.l2_lambda = l2_lambda
       self.random_state = random_state
       self.manual_init = manual_init
       self.manual_weights = manual_weights 
       self.loss_monitoring = loss_monitoring
       self.optimizer_type = optimizer_type
       
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
                if self.manual_init and self.manual_weights is not None:
                
                    layer_idx = 0
                    for layer in self.layers:
                        if isinstance(layer, torch.nn.Linear):
                            if layer_idx < len(self.manual_weights):
                                # Get weight and bias for this layer
                                weight_data, bias_data = self.manual_weights[layer_idx]
                                
                                # Check dimensions match
                                if weight_data.shape == layer.weight.shape:
                                    layer.weight.data = torch.FloatTensor(weight_data)
                                else:
                                    print(f"Warning: Manual weight shape mismatch for layer {layer_idx}. Using default initialization.")
                                    torch.nn.init.xavier_normal_(layer.weight)
                                
                                if bias_data.shape == layer.bias.shape:
                                    layer.bias.data = torch.FloatTensor(bias_data)
                                else:
                                    print(f"Warning: Manual bias shape mismatch for layer {layer_idx}. Using default initialization.")
                                    torch.nn.init.zeros_(layer.bias)
                                
                                layer_idx += 1
                else:    
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

        if self.optimizer_type == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), 
                                        max_iter=100,
                                        max_eval=500,
                                        tolerance_grad=1e-3,      
                                        tolerance_change=1e-5,    
                                        line_search_fn='strong_wolfe')
            
            def closure():
                optimizer.zero_grad()
                output = self(X)
                loss = criterion(output, y)
                
                if self.l2_lambda is not None:
                    for layer in self.layers:
                        if isinstance(layer, torch.nn.Linear):
                            l2_reg = torch.norm(layer.weight, 2)
                            loss += self.l2_lambda * l2_reg / X.shape[0]

                weights_history.append(self.layers[-1].weight.detach().clone().squeeze().numpy())
                loss_history.append(loss.detach().numpy())
                loss.backward()
                return loss
                
            optimizer.step(closure)

        else:
            # Standard optimizers (SGD, Adam)
            if self.optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            elif self.optimizer_type == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self(X)
                loss = criterion(output, y)
                
                if self.l2_lambda is not None:
                    for layer in self.layers:
                        if isinstance(layer, torch.nn.Linear):
                            l2_reg = torch.norm(layer.weight, 2)
                            loss += self.l2_lambda * l2_reg / X.shape[0]
                
                loss.backward()
                optimizer.step()
                
                loss_history.append(loss.detach().numpy().copy())
                weights_history.append(self.layers[-1].weight.detach().clone().squeeze().numpy())

        if self.loss_monitoring == True:
            return np.array(weights_history), np.array(loss_history)
        else:
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
       
       

class LogisticRegressionTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=None, manual_init = False, manual_weights=None, loss_monitoring = False, optimizer_type = "SGD"):
       super().__init__()
       torch.manual_seed(random_state)
       self.linear = None
       self.l2_lambda = l2_lambda
       self.manual_weights = manual_weights 
       self.manual_init = manual_init
       self.loss_monitoring = loss_monitoring
       self.optimizer_type = optimizer_type
       
       
   def _init_model(self, n_features): #this is so we can initilize the model without data
       if self.linear is None:
           self.linear = torch.nn.Linear(n_features, 1)
       
   def init_weights(self):  
        
        if self.manual_init == True:

            if len(self.manual_weights) > 0:
                
                weight_data, bias_data = self.manual_weights

                if weight_data.shape == self.linear.weight.shape:
                    self.linear.weight.data = torch.FloatTensor(weight_data)
                else:
                    print(f"Warning: Manual weight shape mismatch. Using default initialization.")
                    n_in = self.linear.weight.size(1)
                    torch.nn.init.uniform_(self.linear.weight, -(1/n_in)**(1/2), (1/n_in)**(1/2))
                
                if bias_data.shape == self.linear.bias.shape:
                    self.linear.bias.data = torch.FloatTensor(bias_data)
                else:
                    print(f"Warning: Manual bias shape mismatch. Using default initialization.")
                    torch.nn.init.zeros_(self.linear.bias)
        else:
            # Default initialization
            n_in = self.linear.weight.size(1)
            torch.nn.init.uniform_(self.linear.weight, -(1/n_in)**(1/2), (1/n_in)**(1/2))  # as sklearn
            torch.nn.init.zeros_(self.linear.bias)
       
   def forward(self, x):
       return torch.sigmoid(self.linear(x))
   
   def fit(self, X, y, max_iter=1000, epochs=150, batch_size=32):
    weights_history = []
    loss_history = []
    

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
    
    #### SGD
    #optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    
    if self.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    elif self.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = self(X)
        loss = criterion(output, y)
        
        if self.l2_lambda is not None:
            l2_reg = torch.norm(self.linear.weight, 2)
            loss += self.l2_lambda * l2_reg 
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().numpy().copy())
        #### monitor weights     
        weights_history.append(self.linear.weight.detach().numpy().copy())
    if self.loss_monitoring == True:
        return np.array(weights_history), np.array(loss_history)
    else:
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

