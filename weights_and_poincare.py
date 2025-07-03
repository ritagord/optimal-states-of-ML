
import numpy as np
import LoadOSData
import torch
import pandas as pd
#from Models_ver2 import MLPTorch
import Plot_Weights

X, Y = LoadOSData.LoadOSData("WisBreastOrth")
#X, Y = LoadOSData.LoadOSData("CensusOrth")
X = (X-X.mean())/X.std()


rand_rows = np.random.rand(X.shape[0])
bTrainFlag = (rand_rows < 0.7)
X_train = X.loc[bTrainFlag]
Y_train = Y.loc[bTrainFlag]

manual_weights = [
    (np.random.randn(20, X.shape[1]), np.zeros(20)),  
    (np.random.randn(10, 20), np.zeros(10)),              
    (np.random.randn(5, 10), np.zeros(5)),               
    (np.random.randn(1, 5), np.zeros(1))                
]

class MLPTorch(torch.nn.Module):
   def __init__(self, random_state=0, l2_lambda=0.01, manual_init=False, manual_weights=None, loss_monitoring = False, optimizer_type = "SGD"):
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
            print("not lbfgs")
            # Standard optimizers (SGD, Adam)
            if self.optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            elif self.optimizer_type == 'Adam':
                print("adam optimizer")
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
                
                weights_closure = np.zeros((855, ))
                q = 0
                for layer in self.layers:
                        if isinstance(layer, torch.nn.Linear):
                            w = layer.weight.detach().clone().squeeze().numpy()
                            
                            weights_closure[q: q+(len(w.flatten()))] = w.flatten()
                            q += len(w.flatten())
                
                loss_history.append(loss.detach().numpy().copy())
                #weights_history.append(self.layers[-1].weight.detach().clone().squeeze().numpy())
                weights_history.append(weights_closure)
                    
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
       
       
def poincare(weights, losses, plot = False):
    weights_reshaped = weights.reshape(weights.shape[0], -1)  # (epochs, features)
    weight_gradients = np.gradient(weights_reshaped, axis=0)
    gradient_magnitude = np.linalg.norm(weight_gradients, axis=1)
    if plot == True:
        plt.figure(figsize=(10, 8))
        epochs = len(losses)
        scatter = plt.scatter(gradient_magnitude, losses, c=range(epochs), 
                            cmap='viridis', alpha=0.7, s=30)

        plt.plot(gradient_magnitude, losses, '-', alpha=0.3, linewidth=0.8, color='gray')

        plt.xlabel('Weight Gradient Magnitude')
        plt.ylabel('Loss')
        plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, SGD Optimizer, MLP Classifier')
        plt.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Epochs')
        plt.show()
    return gradient_magnitude

import matplotlib.pyplot as plt

num_models = 20
epochs = 100

weight_evolutions = np.zeros((855, num_models))
weights_full_arr = []
losses_full_arr = []
weight_gradients_arr = []

for i in range(num_models):
    perturbed_weights = []
    for weights, biases in manual_weights:
        weight_perturbation = np.random.randn(*weights.shape) * 10**(-2)
        bias_perturbation = np.random.randn(*biases.shape) * 10**(-2)
        new_weights = weights + weight_perturbation
        new_biases = biases + bias_perturbation        
        perturbed_weights.append((new_weights, new_biases))
        
    model = MLPTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True)
    weights, losses = model.fit(X_train, Y_train, epochs = epochs)
    grad_mag = poincare(weights, losses)
    
    weights_full_arr.append(weights)
    losses_full_arr.append(losses)
    weight_gradients_arr.append(grad_mag)
    
    weight_evolutions[:, i] = weights[epochs-1, :] - weights[0, :]


# for i in range(num_models):
#     weights_this_model = weights_full_arr[i]
#     plt.figure()
#     plt.title("Model number "+str(i))
#     for j in indexes:
#         plt.plot(weights_this_model[:, j], label = "weight "+str(j))
#     plt.legend()
#     plt.show()


#### Plot weight evolutions for weights most different between models

indexes = np.argmax(abs(np.diff(weight_evolutions, axis = 1)), axis = 0)   
indexes = np.unique(indexes)

weights_full_arr = np.array(weights_full_arr)
# for j in indexes:
#     plt.figure()
#     plt.title("Weight number "+str(j))   
#     for i in range(num_models):
#         plt.plot(weights_full_arr[i, :, j], label = "model "+str(i)) 
#     plt.legend()
#     plt.show()
    
    
fig, axes = plt.subplots(len(indexes), 1, figsize=(10, 4*len(indexes)))
if len(indexes) == 1:
   axes = [axes]  

fig.suptitle("Evolution of weights most different across models", fontsize = "x-large")

for idx, j in enumerate(indexes):
   axes[idx].set_title(f"Weight number {j}")
   for i in range(num_models):
       axes[idx].plot(weights_full_arr[i, :, j], label=f"model {i}")
   #axes[idx].legend()

plt.show()


weight_gradients_arr = np.array(weight_gradients_arr)
losses_full_arr = np.array(losses_full_arr)

plt.figure(figsize=(10, 8))

for i in range(num_models):
    epochs = len(losses)
    scatter = plt.scatter(weight_gradients_arr[i], losses_full_arr[i],
                        c=range(epochs), cmap='viridis', alpha=0.7, s=2)

    plt.plot(weight_gradients_arr[i], losses_full_arr[i], '-', 
             alpha=0.3, linewidth=0.8, color='gray')

plt.xlabel('Weight Gradient Magnitude')
plt.ylabel('Loss')
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, SGD Optimizer, MLP Classifier')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()

#plt.tight_layout()
plt.show()




#start with a ball of coefs

