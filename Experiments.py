# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:29:05 2024

@author: op038
"""

import sys
import numpy as np
import pandas as pd
import LoadOSData
import Models
import Plots
import time
from sklearn.decomposition import PCA
from copy import deepcopy
import pickle
import Params


########################################################################
#
#
#       Main parameters
#
#
########################################################################
#bRecompute = False # Recompute all numerical experiments

#bFullRun = False # Full or small (faster) run

#ModelCloudMaxHours = 5.0 # maximum hours to build model cloud
#Resol = 500  # image resolution
#Scale = 100.0  # Zooming in/out
#nIterMax = 1000 # maximum number of model iterations to converge
#nInitialModelCount = 1_000_000 # initial model cloud size

# Parameters for faster run
#if bFullRun == False:
#    Resol = 50
#    nInitialModelCount = 5000
#    ModelCloudMaxHours = 2.0/60

bRecompute = Params.bRecompute
bFullRun = Params.bFullRun
bPartFit = Params.bPartFit
ModelCloudMaxHours = Params.ModelCloudMaxHours
Resol = Params.Resol
Scale = Params.Scale
nIterMax = Params.nIterMax
nInitialModelCount = Params.nInitialModelCount



########################################################################
#
#
#       Create model cloud: A large number of model instances
#       fit into the original data
#
#
########################################################################
def CreateModelCloud(model, X, Y, nInitialModelCount, intercept):
        
    # Get some model properties here
    bIterCount = hasattr(model, 'n_iter')
        
        
    # Create initial populaiton of models/states
    # Any reasonable method would work - the most universal
    # is getting different models by fitting 
    # subsamples of the original data
    list_states = []; list_qualities = []; list_output = []; list_niter = [];
    list_qualities_rand = []; 
    q_best = -1e100; model_best = None
    t0 = time.time()
    step = max(100, nInitialModelCount//50)
    
    rand_rows = np.random.rand(X.shape[0])
    
    if bPartFit:
        val_percent = 0.8
        bValFlag = (rand_rows < val_percent)
        print("Validation retraining with ", val_percent*100)
        X_val, y_val = X.loc[bValFlag], Y.loc[bValFlag]
        
        X, Y = X.loc[~bValFlag], Y.loc[~bValFlag]
    list_states_val = []
    list_qualities_val = []
    
    for nModel in range(nInitialModelCount):
        
        # Report progress
        if nModel % step ==0:
            sys.stdout.write("\r  ...creating models %%: %.0f" % (100*nModel/nInitialModelCount)) 
        
        # Exit if too much time was spend
        if nModel % 10 ==0:
            dt = (time.time()-t0)/3600
            if dt > ModelCloudMaxHours:
                print('\n  ...stopping cloud generation after %.1f hours' % (dt) )
                break
               
        # Fit model into a random train set
        rand_rows = np.random.rand(X.shape[0])


        bTrainFlag = (rand_rows < 0.7)
        #print(X.loc[bTrainFlag], Y.loc[bTrainFlag])
        model.fit(X.loc[bTrainFlag], Y.loc[bTrainFlag])
        
        countzero_in_model_pred = not np.any(model.predict(X))
        if countzero_in_model_pred:
            nModel -= 1
            print('failed to converge')
            break 
        
        # Get model state
        state = Models.ModelStateGet(model, intercept = intercept)
        list_states.append(state.reshape(-1))
          
        # Get model quality
        q = model.score(X, Y)
        list_qualities.append(q)
        if q>q_best: # record best observed model
            model_best = deepcopy(model)
            q_best = q
            
              
        # Get model quality on a random subset to avoid quality value artifacts
        r_thr = np.random.uniform(0.88, 0.99)
        b_rr = (np.random.rand(X.shape[0]) < r_thr)
        qr = model.score(X.loc[b_rr], Y.loc[b_rr])
        list_qualities_rand.append(qr)
        
        # Get model output
        list_output.append(model.predict(X))
        
        # Get number of iterations to converge
        list_niter.append(model.n_iter if bIterCount else 0)
        
        #### ONLY FOR mlp TEST
        #X_combined = pd.concat([X.loc[bTrainFlag], X_val])
        #y_combined = pd.concat([Y.loc[bTrainFlag], y_val])
        
        #model.fit(X_combined, y_combined)
        if bPartFit:
            model.partial_fit(X_val, y_val) #does not work with lreg
        
        list_qualities_val.append(model.score(X, Y))
        state = Models.ModelStateGet(model, intercept = intercept)
        list_states_val.append(state.reshape(-1))
        
        
        
        ####
        
    #print("Mean change in quality after adding validation: ", 
    #          np.mean(np.add(np.array(list_qualities_val), -np.array(list_qualities))))
    #print("Max change in quality after adding validation: ", 
    #          np.max(np.add(np.array(list_qualities_val), -np.array(list_qualities))))
    
    # Combine all results into a dataframe
    df_Results = pd.DataFrame( { 'Quality':   list_qualities,
                                 'Quality_R': list_qualities_rand,
                                 'States':    list_states,
                                 'Output':    list_output,
                                 'nIter':     list_niter,
                                 'States_val':list_states_val, 
                                 'Quality_val': list_qualities_val})
    df_Results.sort_values(by=['Quality'], ascending = False, inplace=True) # Highest quality on top
      
    # Report model initialization time
    dt = time.time()-t0
    if dt>2.0:
        print("\nModels: %d instances trained in %.2f sec" % (df_Results.shape[0], dt))
        
    # Return
    return (df_Results, state, model_best)
          
          

########################################################################
#
#
#       Run all experiments: models and datasets
#
#
########################################################################
def ComputeExperiments(DataNames, ModelNames, intercept):
    print(0)  
    # Initialize All_Experiments dictionary to hold all experiments
    All_Experiments = dict()
    t_all = time.time()
        
      
    # Run all experiments
    for DataName in DataNames:
    #for DataName in ["WisBreast_Orth", "WisBreast"]:
    # for DataName in ["WisBreast", "DivideBy30"]:
        
        # Load the original data
        X0, Y0 = LoadOSData.LoadOSData(DataName)
        
        print("loaded data")
        # Normalize X data
        X0 = (X0-X0.mean())/X0.std()
        feature_names = X0.columns
        if intercept:
            feature_names = pd.Index(list(feature_names) + ['intercept'])
        #print(X0, Y0)
     
        for ModelName in ModelNames:
            
            # Get a copy of the original data, to avoid changing it
            X = X0.copy();  Y = Y0.copy()
            
            # Set experiment name
            ExperimentName = ModelName + '_' + DataName
            print('\n')
            print(ExperimentName)
            
            # Initialize the results for this experiment
            strTime =  time.strftime('%b%d_%Hh%Mm')
            ExperimentResults = { 'Name': ExperimentName,
                                  'Resol': Resol, 
                                  'Time': strTime
                                }
            
            # Set model type
            
            model = Models.CreateModel(ModelName, nIterMax)
            print("created first model")
            np.random.seed(10) # making sure we can reproduce each model-data experiment
            
            # Create model cloud
            df_Results, state, model_best = CreateModelCloud(model, X, Y, nInitialModelCount, intercept)
            print("got model cloud")
            states = np.stack(df_Results['States'].values)
            
            
            
            pca = PCA(n_components=3) 


            states_in_pca = pca.fit_transform(states) 
            print('\nPCA variances: ',  pca.explained_variance_ratio_) 
            
            ###
            import pickle
            with open('pca_lreg.pkl', 'wb') as file:
                pickle.dump(pca, file)
            
            
            #with open('pca_lreg.pkl', 'rb') as file:
            #    loaded_pca = pickle.load(file)

            ###

            df_Results['pca_x'] = states_in_pca[:,0] 
            df_Results['pca_y'] = states_in_pca[:,1]   
            df_Results['pca_z'] = states_in_pca[:,2]  
            
            states_val = np.stack(df_Results['States_val'].values)
            states_in_pca_val  = pca.transform(states_val)            
                        
            df_Results['pca_x_val'] = states_in_pca_val[:,0] 
            df_Results['pca_y_val'] = states_in_pca_val[:,1]   


            ######### Top contributions of features to the first principal component
            #pc1_loadings = pca.components_[0]
            #top_10_indices = np.abs(pc1_loadings).argsort()[-10:][::-1]
            #for idx in top_10_indices:
            #    print(f"Feature '{feature_names[idx]}': {pc1_loadings[idx]:.3f}")
            #ExperimentResults['PCA'] = pca
           
               
            #########
            
            # Save results in experiments
            ExperimentResults['df_Results'] = df_Results.copy()
            
            # Set correct shapes
            state_best = Models.ModelStateGet(model_best, intercept = intercept)
            origin = state_best.reshape(state.shape)
            
            model_best_test = Models.ModelStateSet(model_best, intercept, state_best)
            print("Origin quality", model_best_test)
            #origin = states[0, :].reshape(state.shape) # best model
            
            
            vect_x = pca.components_[0,:].reshape(state.shape)
            vect_y = pca.components_[1,:].reshape(state.shape)
            #vect_z = pca.components_[2,:].reshape(state.shape)
            
            # Build 2D distribution of optimal state parameters
            image_quality = np.zeros( (Resol, Resol) )
            image_trained = np.zeros( (Resol, Resol) )
            image_feat    = np.zeros( (Resol, Resol) )
            t0 = time.time()
            step = max(5, Resol//10)
            for nx in range(Resol):
                x = (2.0*nx/Resol)-1.0
                
                # Report progress
                if nx % step ==0:
                    sys.stdout.write("\r  ...evaluating plane models: {}%".format((100*nx)//Resol)) 
                
                for ny in range(Resol):
                    y = (2.0*ny/Resol)-1.0
                    
                    # Set new model state at this point
                    point_state = origin + Scale*(x*vect_x + y*vect_y)
                    
                    # Compute model at the new state 
                    point_model = Models.ModelStateSet(model_best, intercept = intercept, state = point_state)
                    
                    # Retrain this model from this point, if supported
                    # if ModelName in ['MLP']:
                    #     mdl = MLPClassifier(hidden_layer_sizes = (10,10,10), activation = 'tanh', 
                    #                         max_iter=1000, random_state=0,
                    #     point_model = Models.ModelStateSet(deepcopy(model_best), point_state)
                    #     point_model.fit()
                    
                    # Record model quality at this point
                    q = point_model.score(X,Y)
                    image_quality[nx, ny] = q
                    
                    # Record model feature importance. Takes time!
                    # ff = Models.GetFeatureImportance(point_model, X, Y) 
                    # image_feat[nx, ny] = ff[0]
            
                
            # Record all results
            ExperimentResults['Image_Quality'] = image_quality
            ExperimentResults['Image_Quality_trained'] = image_trained
            ExperimentResults['Image_Features'] = image_feat
                                 
            # Report processing time
            dt = time.time()-t0
            if dt>2.0:
                print("Models processed in %.2f sec" % (dt))
                
                
            # Update experiments dictionary with the new experiment result 
            All_Experiments[ExperimentName] = ExperimentResults
            
            # Save all experiments into a file  
            with open('./Output/All_Experiments.pickle', 'wb') as handle:
                pickle.dump(All_Experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(10)
                
            # Plot this particular result
            Plots.PlotAll(ExperimentResults)     
            
    # Report total compute time and exit
    print('\nAll experiments computed in %.1f hours' % ( (time.time()-t_all)/3600 ) )
    return ExperimentResults


########################################################################
#
#
#       Run
#
#
#######################################################################
# if bRecompute: 
#     ComputeExperiments()
# else:
#     Plots.PlotAll()


# Global optimization experiment
    
X0, Y0 = LoadOSData.LoadOSData("WisBreast")
model = Models.CreateModel('LReg', 1000)
model.fit(X0, Y0)
stop = 1

# Define model error function
def ErrFunc(var):
    global model
    global intercept
    state0 = Models.ModelStateGet(model, intercept = intercept)
    v = var.reshape(state0.shape)
    Models.ModelStateSet(model, intercept = intercept, state = v)
    score = model.score(X0, Y0)
    # print('ErrFunc score: ', score)
    return (1-score)

# Define callback function
Nfeval = 0
def callbackF(var):
    global Nfeval
    print("Progress: ", 1-ErrFunc(var), var) 
    Nfeval += 1
    

#score0 = model.score(X0, Y0)
#state0 = Models.ModelStateGet(model, intercept = True)
#print("Initial score", score0)
#ef = ErrFunc(state0)



#from scipy import optimize

# Set the bounds
#x0 = state0.reshape(-1)
#bmax = 100*np.amax(abs(state0)); Bounds = [(-bmax,bmax)]*x0.size

# ef = ErrFunc(x0)
#nworkers = 1
#print('Start optimizing')
#res = optimize.differential_evolution(ErrFunc, bounds = Bounds, x0 = x0)
# res = optimize.dual_annealing(ErrFunc, bounds = Bounds, maxiter = 2000, x0 = x0)
#print('Optimized score 1: ', 1-res.fun)

# res = optimize.shgo(ErrFunc, bounds=Bounds, n=100, callback=callbackF, workers=nworkers)
# print('Optimized score 2: ', 1-res.fun)


