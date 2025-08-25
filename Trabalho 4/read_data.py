# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:48:35 2024

@author: Deivid
"""

excels_page="Database"
target='Bio-oil yield'
dataset="Biomass pyrolysis dataset"
seed=1

def read_data_mendeley(plot=False, target=target, dataset=dataset, seed=seed, boxplot=False):
    
    import sys
    import requests
    import numpy as np
    import pandas as pd
    import pylab as pl
    import matplotlib as mpl
    import seaborn as sns
    import os
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    #from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
    from io import BytesIO
    
    

    # Caminho relativo para o arquivo dentro da pasta 'dataset' no diretório raiz
    file_path = os.path.join(os.getcwd(), 'dataset', 'Biomass_pyrolysis_dataset.xlsx')
    
    # Leitura do arquivo Excel
    dfs = pd.read_excel(file_path)
    X = dfs.copy()
    X.columns
      
    cols_to_drop=[
                  'Biomass',
                  'Origin',
                  'Type of reactor', #fixed-bed constant
                  'Reference',
                  # 'Cellulose', #[%]
                  # 'Hemicellulose', #[%]
                  # 'Lignin', #[%]
                  # 'Pyrolysis temperature [°C]',
                  # 'Heating rate [°C/min]',
                  # 'N2 flow rate [mL/min]',
                  # 'Biomass particle size', #micrometer
                  # 'Bio-oil yield', #[%]
              ]
      
    X['Biomass particle size'] = [float(str(valor).replace('b', '')) for valor in X['Biomass particle size']] #dealing with final b
    X[target] = [float(str(valor).replace('c', '').replace(',','.')) for valor in X[target]] #dealing with final c
      
    X.drop(cols_to_drop, axis=1, inplace=True)
    variable_names = list(X.columns.drop(target))
    X.dropna(inplace=True)
      
      
    df = X.copy()
      
    X_train, X_test, y_train, y_test = train_test_split(X[variable_names], X[target], test_size=0.2, shuffle=True, random_state=seed)
      
    n_samples, n_features = X_train.shape
      
    df_train=X_train.copy()
    df_train[target] = y_train
    stat_train = df_train.describe().T
      
    if plot:
        stat_train.to_latex(buf=(f'./tex/{dataset}_train_{target}.tex').lower(), index=True, caption=f'Basic statistics for dataset {dataset}.')
      
    regression_data = {
        'task' : 'regression',
        'name' : dataset,
        'feature_names' : np.array(variable_names),
        'target_names' : target,
        'n_samples' : n_samples,
        'n_features' : n_features,
        'X_train' : X_train.values,
        'y_train' : y_train.values.T,
        'X_test' : X_test.values,
        'y_test' : y_test.values.T,
        'predicted_lables': None,
        'true_labels' : None,
        'descriptions' : 'None',
        'reference' : "10.17632/bx88ymgbbv.1",
        'items' : None,
        'normalize' : None,
    }
      
    return regression_data