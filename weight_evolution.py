import os
# Set the MKL instruction set before importing libraries that use MKL
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'SSE4_2'
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import time

warnings.filterwarnings('ignore')

import LoadOSData
import torch
from Models_ver2 import MLPTorch, LogisticRegressionTorch
import Plot_Weights

X, Y = LoadOSData.LoadOSData("WisBreastOrth")
#X, Y = LoadOSData.LoadOSData("CensusOrth")
X = (X-X.mean())/X.std()


rand_rows = np.random.rand(X.shape[0])
bTrainFlag = (rand_rows < 0.7)
X_train = X.loc[bTrainFlag]
Y_train = Y.loc[bTrainFlag]


num_models = 20
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
        weight_perturbation = np.random.randn(*weights.shape) * 10**(-2)
        bias_perturbation = np.random.randn(*biases.shape) * 10**(-2)
        new_weights = weights + weight_perturbation
        new_biases = biases + bias_perturbation
        
        perturbed_weights.append((new_weights, new_biases))

    model = MLPTorch(manual_init=True, manual_weights=perturbed_weights)
    
    weights = model.fit(X_train, Y_train, epochs = 100)
    
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



plt.figure(figsize = (6,6))
mask = np.triu(np.ones_like(dist, dtype=bool))
sns.heatmap(dist, mask=~mask, cmap='viridis', square=True)
plt.title("Distance matrix")
plt.show()

max_index = np.unravel_index(np.argmax(dist), dist.shape)
model_idx1, model_idx2 = max_index


preds1 = predictions[model_idx1]
preds2 = predictions[model_idx2]

model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


print("model1_right_model2_wrong", len(np.where(model1_right_model2_wrong)[0]))
print("model1_wrong_model2_right", len(np.where(model1_wrong_model2_right)[0]))


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



model = MLPTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True, optimizer_type = "Adam")
   
weights, losses = model.fit(X_train, Y_train, epochs=100)
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
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, Adam Optimizer, MLP Classifier')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()

model = MLPTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True, optimizer_type = "SGD")
   
weights, losses = model.fit(X_train, Y_train, epochs=100)
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
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, SGD Optimizer, MLP Classifier')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()



X, Y = LoadOSData.LoadOSData("WisBreast")
#X, Y = LoadOSData.LoadOSData("CensusOrth")
X = (X-X.mean())/X.std()


rand_rows = np.random.rand(X.shape[0])
bTrainFlag = (rand_rows < 0.7)
X_train = X.loc[bTrainFlag]
Y_train = Y.loc[bTrainFlag]


num_models = 20
all_weights_history = []
all_final_weights = []
model_scores = []
predictions = []


manual_weights = [np.random.randn(1, X.shape[1]), 
                  np.random.randn(1)             
]

weights, biases = manual_weights

model = LogisticRegressionTorch(manual_init=True, manual_weights=manual_weights, optimizer_type = "Adam")
model.fit(X_train, Y_train)

    
for i in range(num_models):
    seed = np.random.randint(1000)
    np.random.seed(seed)
    print("Seed: ", seed)
    weight_perturbation = np.random.randn(*weights.shape) * 5*10**(-2)
    bias_perturbation = np.random.randn(*biases.shape) * 5*10**(-2)
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

plt.figure(figsize=(6, 6))
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



plt.figure(figsize = (6,6))
mask = np.triu(np.ones_like(dist, dtype=bool))
sns.heatmap(dist, mask=~mask, cmap='viridis', square=True)
plt.title("Distance matrix")
plt.show()

max_index = np.unravel_index(np.argmax(dist), dist.shape)
model_idx1, model_idx2 = max_index


preds1 = predictions[model_idx1]
preds2 = predictions[model_idx2]

model1_right_model2_wrong = (preds1[:, 0] == Y) & (preds2[:, 0]  != Y)
model1_wrong_model2_right = (preds1[:, 0]  != Y) & (preds2[:, 0]  == Y)


print("model1_right_model2_wrong", len(np.where(model1_right_model2_wrong)[0]))
print("model1_wrong_model2_right", len(np.where(model1_wrong_model2_right)[0]))



#plt.figure(figsize=(12, 10))
#plt.suptitle("Weight Evolution")

#for i in range(num_models):
 #   plt.subplot(num_models, 1, i+1)
 #   plt.title(f"Model {i} (Score: {model_scores[i]:.4f})")
#    weights1 = all_weights_history[i]
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


model = LogisticRegressionTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True, optimizer_type = "Adam")
   
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
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, Adam Optimizer, Logistic Regression')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()


model = LogisticRegressionTorch(manual_init=True, manual_weights=perturbed_weights, loss_monitoring = True, optimizer_type = "SGD")
   
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
plt.title('Poincaré Map: Weight Gradient Magnitude vs Loss, SGD Optimizer, Logistic Regression')
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Epochs')
plt.show()



