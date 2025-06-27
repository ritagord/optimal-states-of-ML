import os
# Set the MKL instruction set before importing libraries that use MKL
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import time

warnings.filterwarnings('ignore')

import LoadOSData
import torch
class MLPTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=0.01, manual_init=False, manual_weights=None, loss_monitoring = False):
       super().__init__()
       
       random_state = torch.seed()
       self.layers = None
       self.l2_lambda = l2_lambda
       self.random_state = random_state
       self.manual_init = manual_init
       self.manual_weights = manual_weights 
       self.loss_monitoring = loss_monitoring
       
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

        #optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=self.l2_lambda)

        #### LBFGS
        #optimizer = torch.optim.LBFGS(self.parameters(), max_iter=2000)
        # Detailed parameters from sklearn default
        optimizer = torch.optim.LBFGS(self.parameters(), 
                                #max_iter=1000, #20
                                #max_eval=500,  #50
                                #tolerance_grad=1e-6,      
                                #tolerance_change=1e-9,  
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
                         loss += self.l2_lambda * l2_reg / X.shape[0] #sklearn divides by sample size

             weights_history.append(self.layers[-1].weight.detach().clone().squeeze().numpy())
             loss_history.append(loss.detach().numpy())
             loss.backward()
            
             return loss
            
        optimizer.step(closure)
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


def CreateModel(ModelName, nIterMax = 5000):    
    model = None    
    if ModelName=="MLPTorch":
        print("init model")
        model = MLPTorch(random_state=0, l2_lambda=0.01)
 
    else:
        raise Exception("Invalid model")
        
    return model


X, Y = LoadOSData.LoadOSData("WisBreast")
#X, Y = LoadOSData.LoadOSData("CensusOrth")
X = (X-X.mean())/X.std()


rand_rows = np.random.rand(X.shape[0])
bTrainFlag = (rand_rows < 0.7)
X_train = X.loc[bTrainFlag]
Y_train = Y.loc[bTrainFlag]


num_models = 10
all_weights_history = []
all_final_weights = []
model_scores = []
predictions = []

manual_weights = [
    (np.random.randn(20, X.shape[1]), np.zeros(20)),  
    (np.random.randn(10, 20), np.zeros(10)),              
    (np.random.randn(5, 10), np.zeros(5)),               
    (np.random.randn(1, 5), np.zeros(1))                
]


for i in range(num_models):
    perturbed_weights = []
    for weights, biases in manual_weights:
        weight_perturbation = np.random.randn(*weights.shape) * 10**(-3)
        bias_perturbation = np.random.randn(*biases.shape) * 10**(-3)
        new_weights = weights + weight_perturbation
        new_biases = biases + bias_perturbation
        
        perturbed_weights.append((new_weights, new_biases))

    model = MLPTorch(manual_init=True, manual_weights=perturbed_weights)

    #model = CreateModel(ModelName="MLPTorch", nIterMax=2000)
    
    weights = model.fit(X_train, Y_train)
    
    all_weights_history.append(weights)
    all_final_weights.append(weights[-1])
    
    score = model.score(X, Y)
    model_scores.append(score)
    predictions.append(model.predict(X))
    
    

plt.figure(figsize=(12, 10))
plt.title("Model scores")
plt.plot(model_scores, "o")
plt.show()

dist = np.zeros((num_models, num_models))
for i in range(num_models):
    for k in range(i+1, num_models):
        if model_scores[i] != model_scores[k]:
            model_idx1 = i
            model_idx2 = k

            weights1 = all_weights_history[model_idx1]
            weights2 = all_weights_history[model_idx2]
            
            dist[i, k] = (np.linalg.norm(weights1[-1, :] - weights2[-1, :]))
            
            preds1 = predictions[model_idx1]
            preds2 = predictions[model_idx2]

            model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
            model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


            print("model1_right_model2_wrong", np.where(model1_right_model2_wrong)[0])
            print("model1_wrong_model2_right", np.where(model1_wrong_model2_right)[0])


print("Distance matrix", dist)
max_index = np.unravel_index(np.argmax(dist), dist.shape)
model_idx1, model_idx2 = max_index
print("model_idx1: ", model_idx1)
print("model_idx2: ", model_idx2)

preds1 = predictions[model_idx1]
preds2 = predictions[model_idx2]

model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


print("model1_right_model2_wrong", np.where(model1_right_model2_wrong)[0])
print("model1_wrong_model2_right", np.where(model1_wrong_model2_right)[0])


plt.figure(figsize=(12, 10))
plt.suptitle("Weight Evolution")

plt.subplot(2, 1, 1)
plt.title(f"Model {model_idx1} (Score: {model_scores[model_idx1]:.4f})")
weights1 = all_weights_history[model_idx1]
for j in range(min(5, weights1.shape[1])):
    plt.plot(weights1[:, j], ".-", label=f"Weight {j+1}")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title(f"Model {model_idx2} (Score: {model_scores[model_idx2]:.4f})")


for j in range(min(5, weights2.shape[1])):
    plt.plot(weights2[:, j], ".-", label=f"Weight {j+1}")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



model = MLPTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True)
   
weights, losses = model.fit(X_train, Y_train, epochs=500)
weights_reshaped = weights.reshape(weights.shape[0], -1)  # (epochs, features)
weight_gradients = np.gradient(weights_reshaped, axis=0)

# L2 norm
gradient_magnitude = np.linalg.norm(weight_gradients, axis=1)

plt.figure(figsize=(10, 8))
epochs = len(losses)
scatter = plt.scatter(gradient_magnitude, losses, c=range(epochs), 
                     cmap='viridis', alpha=0.7, s=30)

plt.plot(gradient_magnitude, losses, '-', alpha=0.3, linewidth=0.8, color='gray')

plt.xlabel('Weight Gradient Magnitude')
plt.ylabel('Loss')
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()





class LogisticRegressionTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=None, manual_init = False, manual_weights=None, loss_monitoring = False):
       super().__init__()
       torch.manual_seed(random_state)
       self.linear = None
       self.l2_lambda = l2_lambda
       self.manual_weights = manual_weights 
       self.manual_init = manual_init
       self.loss_monitoring = loss_monitoring
       
       
   def _init_model(self, n_features): #this is so we can initilize the model without data
       if self.linear is None:
           self.linear = torch.nn.Linear(n_features, 1)
       
   def init_weights(self):  # set initial weights and biases
        #print("weights shape1", np.shape(self.linear.weight))
        #print("biases shape1", np.shape(self.linear.bias))
        
        if self.manual_init == True:

            if len(self.manual_weights) > 0:
                
                weight_data, bias_data = self.manual_weights
                
                #print("weights shape from manual", np.shape(weight_data))
                #print("biases shape from manual", np.shape(bias_data))
                
                # Check dimensions match
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
    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    
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


def CreateModel(ModelName, nIterMax = 5000):    
    model = None    
    if ModelName=="MLPTorch":
        print("init model")
        model = MLPTorch(random_state=0, l2_lambda=0.01)
 
    else:
        raise Exception("Invalid model")
        
    return model


X, Y = LoadOSData.LoadOSData("WisBreast")
#X, Y = LoadOSData.LoadOSData("CensusOrth")
X = (X-X.mean())/X.std()


rand_rows = np.random.rand(X.shape[0])
bTrainFlag = (rand_rows < 0.7)
X_train = X.loc[bTrainFlag]
Y_train = Y.loc[bTrainFlag]


num_models = 10
all_weights_history = []
all_final_weights = []
model_scores = []
predictions = []


manual_weights = [np.random.randn(1, X.shape[1]), 
                  np.random.randn(1)             
]

weights, biases = manual_weights
#print("manual weights shape", np.shape(weights))
#print("manual biases shape", np.shape(biases))

model = LogisticRegressionTorch(manual_init=True, manual_weights=manual_weights)
model.fit(X_train, Y_train)

#print("FIRST MODEL TRAINED")

    
for i in range(num_models):
    seed = np.random.randint(1000)
    np.random.seed(seed)
    print("Seed: ", seed)
    weight_perturbation = np.random.randn(*weights.shape) * 5*10**(-1)
    bias_perturbation = np.random.randn(*biases.shape) * 5*10**(-1)
    new_weights = weights + weight_perturbation
    new_biases = biases + bias_perturbation
    
    perturbed_weights = [new_weights, new_biases]

    model = LogisticRegressionTorch(manual_init=True, manual_weights=perturbed_weights)
    
    # Use SGD or Adam instead of LBFGS for better weight tracking
    weights1 = model.fit(X_train, Y_train, epochs=500)
    
    all_weights_history.append(weights1)
    all_final_weights.append(weights1[-1])
    
    score = model.score(X, Y)
    model_scores.append(score)
    predictions.append(model.predict(X))    

plt.figure(figsize=(12, 10))
plt.title("Model scores")
plt.plot(model_scores, "o")
plt.show()



dist = np.zeros((num_models, num_models))
for i in range(num_models):
    for k in range(i+1, num_models):
        if model_scores[i] != model_scores[k]:
            model_idx1 = i
            model_idx2 = k

            weights1 = all_weights_history[model_idx1]
            weights2 = all_weights_history[model_idx2]
            
            dist[i, k] = (np.linalg.norm(weights1[-1, :] - weights2[-1, :]))
            
            preds1 = predictions[model_idx1]
            preds2 = predictions[model_idx2]

            model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
            model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


            print("model1_right_model2_wrong", np.where(model1_right_model2_wrong)[0])
            print("model1_wrong_model2_right", np.where(model1_wrong_model2_right)[0])

print("Distance matrix", dist)
#print(dist)
max_index = np.unravel_index(np.argmax(dist), dist.shape)
model_idx1, model_idx2 = max_index
#print("maximal distance between weights")
#print("model_idx1: ", model_idx1)
#print("model_idx2: ", model_idx2)

#plt.figure(figsize=(12, 10))
#plt.suptitle("Weight Evolution")

#for i in range(num_models):
#    plt.subplot(num_models, 1, i+1)
#    plt.title(f"Model {i} (Score: {model_scores[i]:.4f})")
#    weights1 = all_weights_history[i]xs
#    #print(np.shape(weights1))
#    for j in range(5):
#        plt.plot(weights1[:, 0, j], ".-", label=f"Weight {j+1}")
#    plt.legend()
#    plt.grid(True)

#plt.show()

plt.figure(figsize=(12, 10))
plt.suptitle("Weight Evolution")
plt.subplot(2, 1, 1)
plt.title(f"Model {model_idx1} (Score: {model_scores[model_idx1]:.4f})")
weights1 = all_weights_history[model_idx1]
for j in range(5):
        plt.plot(weights1[:, 0, j], ".-", label=f"Weight {j+1}")
plt.legend()
plt.grid(True)
plt.subplot(2, 1, 2)        
plt.title(f"Model {model_idx2} (Score: {model_scores[model_idx2]:.4f})")
weights2 = all_weights_history[model_idx2]
for j in range(5):
        plt.plot(weights2[:, 0, j], ".-", label=f"Weight {j+1}")        
plt.legend()
plt.grid(True)
plt.show()


preds1 = predictions[model_idx1]
preds2 = predictions[model_idx2]

model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


print("model1_right_model2_wrong", np.where(model1_right_model2_wrong)[0])
print("model1_wrong_model2_right", np.where(model1_wrong_model2_right)[0])


model = LogisticRegressionTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True)
   
weights, losses = model.fit(X_train, Y_train, epochs=500)
weights_reshaped = weights.reshape(weights.shape[0], -1)  # (epochs, features)
weight_gradients = np.gradient(weights_reshaped, axis=0)

# L2 norm
gradient_magnitude = np.linalg.norm(weight_gradients, axis=1)

plt.figure(figsize=(10, 8))
epochs = len(losses)
scatter = plt.scatter(gradient_magnitude, losses, c=range(epochs), 
                     cmap='viridis', alpha=0.7, s=30)

plt.plot(gradient_magnitude, losses, '-', alpha=0.3, linewidth=0.8, color='gray')

plt.xlabel('Weight Gradient Magnitude')
plt.ylabel('Loss')
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()




