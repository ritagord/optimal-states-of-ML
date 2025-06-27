# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 09:33:33 2024

@author: op038
"""

import numpy as np
import pickle
import time
import random
import matplotlib.pyplot as plt


########################################################################
#
#
#       Main parameters
#
#
########################################################################
bImage = True

bScatter2D = True

import Params
                
bPartFit = Params.bPartFit

########################################################################
#
#
#       Plot all experiments from saved data
#
#
########################################################################
def PlotAll(ExperimentResults = None):
    
    
    # Get the data first
    if ExperimentResults is None:
        # Read all experiments from the saved file
        with open('./Output/All_Experiments.pickle', 'rb') as handle:
            All_Experiments = pickle.load(handle)
        time.sleep(2)
    else: # we are plotting a specific result
        All_Experiments = { ExperimentResults['Name'] : ExperimentResults.copy() }
    
    
    for ExperimentName, ExperimentDict in All_Experiments.items():
        
        strTime = ''
        alpha = 0.2
        
        for key, value in ExperimentDict.items():
            
            # Plot 2D images
            if key.startswith('Image_') and bImage:  

                title = key.replace('Image_', ExperimentName)
                
                # Skip empty images
                if np.amax(value) <= np.amin(value) + 1.0e-10:
                    continue
        
                # Plot the array
                fig = plt.figure(figsize=(8,8), dpi=600)
                plt.title(title + '.Image')
                plt.imshow(ExperimentDict[key])
                plt.colorbar(orientation='vertical')
                plt.show()
                
                # Save plot image
                strTime =  ''
                # strTime =  time.strftime('%b%d_%Hh%Mm')
                filename = './Output/' + title + '.' + strTime + '.png'
                fig.savefig(filename, bbox_inches='tight')
                # plt.close(fig)
                
                title = key.replace('Image_', ExperimentName)
                
                # Skip empty images
                if np.amax(value) <= np.amin(value) + 1.0e-10:
                    continue
        
                # Plot the array
                fig = plt.figure(figsize=(8,8), dpi=600)
                
                if ExperimentName.startswith("LReg"):
                    C = Params.C
                    plt.title(title + '_Reg_'+str(C)+'.Contour')
                else:
                    plt.title(title + '.Contour')
                
                levels = np.arange(0.2, 1, 0.01)
               
                #levels = np.arange(np.min(ExperimentDict[key]), np.max(ExperimentDict[key]), 0.01)
                plt.contour(ExperimentDict[key], levels = levels)
                plt.colorbar(orientation='vertical')
                plt.show()
                
                # Save plot image
                strTime =  ''
                # strTime =  time.strftime('%b%d_%Hh%Mm')
                filename = './Output/' + title + '.Contour' + strTime + '.png'
                fig.savefig(filename, bbox_inches='tight')
                print("finished plotting images")
                # plt.close(fig)
            
            
            # Plot scatter plots for model density
            elif key == 'df_Results':
                print("plotting df_results")
                bQuality = True
                
                if bQuality:
                    ##################### Density of good models
                    # Set title
                    title = ExperimentName + '_' + 'GoodModelDensity'
                    
                    # Get best models
                    qbest = value['Quality'].quantile(.9)
                    df_best = value[value['Quality']>=qbest]
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    plt.title(title)
                    plt.scatter(df_best['pca_x'], df_best['pca_y'], 
                                c="green", 
                                alpha=0.3, 
                                edgecolor='none')
                    
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
      
                    plt.show()
                    filename = './Output/' + title + '.' + strTime + '.png'
                    fig.savefig(filename, bbox_inches='tight')
                                
                    ##################### Quality Scatter
                    # Set title
                    title = ExperimentName + '_' + 'QualityScatter'
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
    
                    plt.scatter(value['pca_x'],   value['pca_y'],   
                                c=value['Quality'],  
                                cmap='viridis', 
                                alpha=0.8, 
                                edgecolor='none')
                    
                    plt.colorbar()
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
      
                    plt.show()
                    filename = './Output/' + title + '.' + strTime + '.png'
                    fig.savefig(filename, bbox_inches='tight')
                
                    
                if bPartFit:
                    ##################### Quality Scatter
                    # Set title
                    title = ExperimentName + '_' + 'QualityScatterVal'
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
    
                    plt.scatter(value['pca_x_val'],   value['pca_y_val'],   
                                c=value['Quality_val'],  
                                cmap='viridis', 
                                alpha=0.8, 
                                edgecolor='none')
                    
                    plt.colorbar()
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
      
                    plt.show()
                    filename = './Output/' + title + '.' + strTime + '.png'
                    fig.savefig(filename, bbox_inches='tight')
                    
                    ####Vector space for retraining with validation set
                    # Set title
                    
                    from matplotlib.colors import Normalize
                    norm = Normalize(vmin=value['Quality'].min(), vmax=value['Quality'].max())
                    title = ExperimentName + '_' + 'ValidationVector'
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.quiver(value['pca_x'], value['pca_y'],                          # starting points
                    value['pca_x_val'] - value['pca_x'],                      # x direction
                    value['pca_y_val'] - value['pca_y'],                      # y direction
                    angles='xy', scale_units='xy', scale=1,                   # maintain aspect ratio
                    norm=norm, alpha = 0.5,                                               # add the normalizer
                    color=plt.cm.viridis(norm(value['Quality'])))
    
                    
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
                    plt.colorbar(label='Quality')
                    plt.show()
                    #filename = './Output/' + title + '.' + strTime + '.png'
                    #fig.savefig(filename, bbox_inches='tight')
                    title = ExperimentName + '_' + 'ValidationVector'
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.quiver(value['pca_x'], value['pca_y'],                          # starting points
                    value['pca_x_val'] - value['pca_x'],                      # x direction
                    value['pca_y_val'] - value['pca_y'],                      # y direction
                    angles='xy', scale_units='xy', scale=1,                   # maintain aspect ratio
                    norm=norm, alpha = 0.5,                                              # add the normalizer
                    color=plt.cm.viridis(norm(value['Quality_val'])))
    
                    
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
                    plt.colorbar(label='Quality with validation')
                    plt.show()

                
                    ###combination of original and validation sets
                    title = ExperimentName + '_' + 'ValidationVectorScatter'
    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.quiver(value['pca_x'], value['pca_y'],                          # starting points
                    value['pca_x_val'] - value['pca_x'],                      # x direction
                    value['pca_y_val'] - value['pca_y'],                      # y direction
                    angles='xy', scale_units='xy', scale=1,                   # maintain aspect ratio
                    norm=norm, headwidth = 10,                       # add the normalizer
                    color=plt.cm.viridis(norm(value['Quality'])), alpha = 0.5)
                    
                    plt.scatter(value['pca_x'],   value['pca_y'],   
                                c=value['Quality'],  
                                cmap='viridis', 
                                alpha=0.9, s = 5,  
                                edgecolor='none')
                    
                    plt.scatter(value['pca_x_val'],   value['pca_y_val'],   
                                c=value['Quality_val'],  
                                cmap='viridis', 
                                alpha=0.9, s = 5, 
                                edgecolor='none')
                    
                    plt.xlabel("pca_1"); plt.ylabel("pca_2");
                    plt.colorbar(label='Quality')
                    plt.show()
                
                
                ##################### dState-dQuality plot
                # Set title
                #title = ExperimentName + '_' + 'dState-dQuality_Scatter'
                
                # Compute points
                #Q = value['Quality_R'].to_numpy()
                #Q = value['Quality'].to_numpy()
                #states = np.stack(value['States'].values)
                #nRows = value.shape[0]
                #dstate_list = [] 
                #dq_list = []
                #for k in range (1000):
                #    i1 = random.randrange(nRows)
                #    i2 = random.randrange(nRows)
                #    dist = np.linalg.norm(states[i1,:]-states[i2,:])
                #    dstate_list.append(dist)
                #    dq_list.append(abs(Q[i1]-Q[i2]))
                    
                
                # Plot the array
                #fig = plt.figure(figsize=(8,8), dpi=600)
                
                #plt.title(title)
                #plt.scatter(dstate_list, dq_list, s = 4, alpha=0.2, edgecolor='none')     
                #plt.xlabel('Model distance')
                #plt.xlabel('Quality distance')
                #plt.show()
                
                ##################### State-Quality plot
                
                # Set title
                #title = ExperimentName + '_' + 'QualityScatter'
                
                # Get best models
                #qbest = value['Quality'].quantile(.9)
                #df_best = value[value['Quality']>=qbest]
                
                # Plot the array
                #fig = plt.figure(figsize=(8,8), dpi=600)
                
                #plt.title(title)
                
                #if bScatter2D: # 2D scatter
                #    plt.scatter(value['pca_x'],   value['pca_y'],   alpha=alpha/2, edgecolor='none')
                    # plt.scatter(df_best['pca_x'], df_best['pca_y'], color = 'red', alpha=alpha, edgecolor='none')
                #    plt.xlabel("pca_1"); plt.ylabel("pca_2");
                #else:          # 3D scatter
                #    ax = plt.axes(projection ="3d")
                #    tdf = value
                    # tdf = value[value['pca_z'].abs()<0.05]
                #    ax.scatter(tdf['pca_x'], tdf['pca_y'], tdf['pca_z'], s=4, alpha=0.01, edgecolor='none')
                    # ax.scatter(df_best['pca_x'], df_best['pca_y'], value['pca_z'],  s=1, color = 'red', alpha=alpha, edgecolor='none')
                #    ax.set_xlabel('pca_1'); ax.set_ylabel('pca_2'); ax.set_zlabel('pca_3')
                    
                #plt.show()
                #filename = './Output/' + title + '.' + strTime + '.png'
                #fig.savefig(filename, bbox_inches='tight')


######################     my tests
                bcollin = False
                if bcollin:
                    title = ExperimentName + '_' + 'Collinear_features'
                    
                    # Compute points
                    #Q = value['Quality_R'].to_numpy()
                    Q = value['Quality'].to_numpy()
                    states = np.stack(value['States'].values)
                    
                    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.scatter(states[:, 1],states[:, 2], c = value['Quality'], s = 2, alpha=0.6, edgecolor='none')     
                    plt.xlabel('C1')
                    plt.ylabel('C2')
                    plt.colorbar(orientation='vertical')
                    plt.show()
                    
                    qbest = 0.85
                    df_best = value[value['Quality']>=qbest]
                    states_best = np.stack(df_best['States'].values)
                    plt.title(title+"best")
                    plt.scatter(states_best[:, 1],states_best[:, 2], c = df_best['Quality'], s = 2, alpha=0.6, edgecolor='none')     
                    plt.xlabel('C1')
                    plt.ylabel('C2')
                    plt.colorbar(orientation='vertical')
                    plt.show()
                    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.scatter(states[:, 0],states[:, 3], c = value['Quality'], s = 2, alpha=0.6, edgecolor='none')     
                    plt.xlabel('C0')
                    plt.ylabel('C3')
                    plt.colorbar(orientation='vertical')
                    plt.show()
                    
                    qbest = 0.99
                    df_best = value[value['Quality']>=qbest]
                    states_best = np.stack(df_best['States'].values)
                    plt.title(title+"best")
                    plt.scatter(states_best[:, 0],states_best[:, 3], c = df_best['Quality'], s = 2, alpha=0.6, edgecolor='none')     
                    plt.xlabel('C0')
                    plt.ylabel('C3')
                    plt.colorbar(orientation='vertical')
                    plt.show()
                    
                    
    # for state with highest variance, plot quality and coef difference from the mean 
                extra_plots = False
                if extra_plots == True:
                    title = ExperimentName + '_' + 'State-Quality_From_mean'
                    
                    # Compute points
                    #Q = value['Quality_R'].to_numpy()
                    Q = value['Quality'].to_numpy()
                    states = np.stack(value['States'].values)
                    meanq = np.mean(Q)
                    highest_var_idx = np.argmax(np.var(states, axis=0))
                    highest_var_param = states[:, highest_var_idx]
                    meanp = np.mean(highest_var_param)
                    nRows = value.shape[0]
                    dstate_list = [] 
                    dq_list = []
                    
                    for k in range (1000):
                        i1 = random.randrange(nRows)
                        dstate_list.append(highest_var_param[i1] - meanp)
                        dq_list.append(Q[i1]- meanq)
                        
                    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.scatter(dstate_list, dq_list, s = 4, alpha=0.8, edgecolor='none')     
                    plt.xlabel('Highest variance coef distance from mean')
                    plt.ylabel('Quality distance from mean')
                    plt.show()
                    
    # for state with highest contribution to the first pca component, plot quality and coef difference from the mean 
    
                    title = ExperimentName + '_' + '1st_PC-Quality_From_mean'
                    
                    # Compute points
                    Q = value['Quality'].to_numpy()
                    states = np.stack(value['States'].values)
                    meanq = np.mean(Q)
                    
                    top10features = ExperimentResults['PrincipalComponent1contributors']
                    
                    highest_pc1_contribution_param = states[:, top10features[0]]
                    
                    meanp = np.mean(highest_pc1_contribution_param)
                    nRows = value.shape[0]
                    dstate_list = [] 
                    dq_list = []
                    
                    for k in range (1000):
                        i1 = random.randrange(nRows)
                        dstate_list.append(highest_pc1_contribution_param[i1] - meanp)
                        dq_list.append(Q[i1]- meanq)
                        
                    
                    # Plot the array
                    fig = plt.figure(figsize=(8,8), dpi=600)
                    
                    plt.title(title)
                    plt.scatter(dstate_list, dq_list, s = 4, alpha=0.8, edgecolor='none')     
                    plt.xlabel('Feature contributing most to PC1 distance from mean')
                    plt.ylabel('Quality distance from mean')
                    plt.show()                
                    
