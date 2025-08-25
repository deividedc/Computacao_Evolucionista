#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
np.int = np.int64
import pandas as pd
import pygmo as pg
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     cross_val_predict,train_test_split)
from sklearn.metrics import r2_score, max_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, Ridge, PassiveAggressiveRegressor, LogisticRegression, BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor#, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import  XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from pyearth import Earth as MARS


import re
import random


from sklearn.kernel_approximation import RBFSampler,SkewedChi2Sampler

from util.ELM import  ELMRegressor, ELMRegressor
from util.MLP import MLPRegressor as MLPR
from util.RBFNN import RBFNNRegressor, RBFNN
from util.LSSVR import LSSVR

from scipy import stats
from hydroeval import kge, nse


import warnings
import sys, getopt

import json
from pathlib import Path

#%%----------------------------------------------------------------------------   
# Importa função para ler datasets específicos
from read_data import *


#%%----------------------------------------------------------------------------
# Configura o pandas para exibir números decimais com 3 casas após o ponto
pd.options.display.float_format = '{:.3f}'.format

# Ignora todos os avisos (warnings) que possam ser gerados durante a execução
warnings.filterwarnings('ignore')

# Obtém o nome do programa/script que está sendo executado
program_name = sys.argv[0]

# Obtém os argumentos passados ao script na linha de comando (excluindo o nome do script)
arguments = sys.argv[1:]

# Conta quantos argumentos foram passados
count = len(arguments)

# Verifica se algum argumento foi passado
if len(arguments) > 0:
    # Se o primeiro argumento for '-r', então pega o segundo argumento como número de execuções
    if arguments[0] == '-r':
        run0 = int(arguments[1])  # Converte o argumento para inteiro
        n_runs = run0             # Define o número de execuções
else:
    # Se nenhum argumento foi passado, define valores padrão
    run0, n_runs = 45, 50

#%%----------------------------------------------------------------------------   
# =============================================================================
#                       Funções de avaliação de modelo
# =============================================================================

def accuracy_log(y_true, y_pred):
    """
    Calcula a acurácia em escala logarítmica.
    A acurácia é definida como a porcentagem de previsões cujo erro logarítmico absoluto é menor que 0.3.
    
    Args:
        y_true: array-like, valores reais
        y_pred: array-like, valores previstos
    
    Returns:
        Acurácia (%) baseada no log10 das previsões
    """
    y_true = np.abs(np.array(y_true))  # garante valores absolutos
    y_pred = np.abs(np.array(y_pred))
    
    return (np.abs(np.log10(y_true / y_pred)) < 0.3).sum() / len(y_true) * 100

#------------------------------------------------------------------------------
def rms(y_true, y_pred):
    """
    Calcula a raiz do erro quadrático médio (RMS) em escala logarítmica.
    
    Args:
        y_true: array-like, valores reais
        y_pred: array-like, valores previstos
    
    Returns:
        RMS log10
    """
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    
    return ( (np.log10(y_pred / y_true)**2).sum() / len(y_true) )**0.5

#------------------------------------------------------------------------------
def lhsu(xmin, xmax, nsample):
    """
    Gera amostras usando Latin Hypercube Sampling (LHS) uniforme.
    
    Args:
        xmin: lista ou array com valores mínimos para cada variável
        xmax: lista ou array com valores máximos para cada variável
        nsample: número de amostras desejadas
    
    Returns:
        s: array (nsample x nvar) contendo as amostras LHS
    """
    nvar = len(xmin)  # número de variáveis
    ran = np.random.rand(nsample, nvar)  # números aleatórios uniformes entre 0 e 1
    s = np.zeros((nsample, nvar))  # inicializa array de amostras
    
    # Loop em cada variável para distribuir amostras
    for j in range(nvar):
        idx = np.random.permutation(nsample)  # permutação aleatória dos índices
        P = (idx.T - ran[:, j]) / nsample     # cálculo do deslocamento dentro da célula LHS
        s[:, j] = xmin[j] + P * (xmax[j] - xmin[j])  # escala para o intervalo da variável
    
    return s

#------------------------------------------------------------------------------   
def RMSE(y, y_pred):
    """
    Calcula o RMSE entre valores reais e previstos.
    
    Args:
        y: array-like de valores reais
        y_pred: array-like de valores previstos
    
    Returns:
        RMSE: float
    """
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()  # garante vetores 1D
    error = y - y_pred  # vetor de erros
    return np.sqrt(np.mean(np.power(error, 2)))  # raiz do quadrado médio

#------------------------------------------------------------------------------   
def RRMSE(y, y_pred):
    """
    Calcula o RMSE relativo, ou seja, RMSE em porcentagem da média dos valores reais.
    
    Args:
        y: array-like de valores reais
        y_pred: array-like de valores previstos
    
    Returns:
        RRMSE (%)
    """
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    return RMSE(y, y_pred) * 100 / np.mean(y)

#------------------------------------------------------------------------------   
def MAPE(y_true, y_pred):    
    """
    Calcula o erro percentual absoluto médio (MAPE).
    
    Args:
        y_true: array-like de valores reais
        y_pred: array-like de valores previstos
    
    Returns:
        MAPE (%)
    """
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    return np.mean(np.abs(y_pred - y_true) / np.abs(y_true)) * 100

# =============================================================================
#               Função principal de avaliação do modelo
# =============================================================================
def model_base_evaluation(x, data_args, estimator_args,
                          normalizer_args, transformer_args):
    """
    Avalia um modelo de aprendizado de máquina com diferentes normalizadores,
    transformadores e estimadores.
    
    Pode executar duas ações dependendo do flag:
        - 'eval': realiza cross-validation e retorna a métrica de erro
        - 'run' : treina o modelo e retorna previsões e parâmetros do modelo

    Args:
        x: array de parâmetros do modelo e seleção de features
        data_args: tupla com dados de treino e teste e parâmetros gerais
        estimator_args: tupla com informações sobre o estimador
        normalizer_args: tupla indicando o tipo de normalização
        transformer_args: tupla indicando o tipo de transformação e kernel
    
    Returns:
        - Em 'eval': valor do cross-validation
        - Em 'run' : dicionário com previsões, parâmetros e informações do modelo
    """
    
    # Desempacota argumentos de dados
    (X_train_, y_train, X_test_, y_test, flag, task, n_splits, 
     random_seed, scoring, target, 
     n_samples_train, n_samples_test, n_features) = data_args
    
    # Desempacota normalizador
    (normalizer_type,) = normalizer_args
    
    # Desempacota transformador
    (transformer_type, n_components, kernel_type) = transformer_args
    
    # Desempacota estimador
    (clf_name, n_decision_variables, p, clf) = estimator_args

    # =========================
    # Dicionário de normalizadores
    # =========================
    normalizer = { 
        'None'           : FunctionTransformer(),             # sem transformação
        'MinMax'         : MinMaxScaler(),                   # escala entre 0 e 1
        'MaxAbs'         : MaxAbsScaler(),                   # escala por valor absoluto máximo
        'Standard'       : StandardScaler(),                # padronização (média=0, std=1)
        'Log'            : FunctionTransformer(np.log1p),   # log(1+x)
        'Quantile Norm.' : QuantileTransformer(n_quantiles=n_features, output_distribution='normal'),
        'Quantile Unif.' : QuantileTransformer(n_quantiles=n_features, output_distribution='uniform'),
        'Poly'           : PolynomialFeatures(),            # features polinomiais
    }
    
    # mapeamento de índices para nomes de normalizadores
    normalizer_dict = {0:'None', 1:'MinMax', 3:'MaxAbs', 2:'Standard', 4:'Log', 5:'Quantile Norm.', 6:'Quantile Unif.', 7:'Poly'}
    n = normalizer_dict[normalizer_type]

    # Dicionário de kernels e transformadores
    kernel = {0:"linear", 1:"poly", 2:"rbf", 3:"sigmoid", 4:"cosine"}
    
    if transformer_type == 'Identity':
        n_components = None  # não aplica redução de dimensão
    if transformer_type == 'KPCA':
        k = kernel[kernel_type]
    else:
        k = None
    
    transformer = {
        'Identity' : FunctionTransformer(),  # identidade
        'PCA'      : KernelPCA(kernel='linear', n_components=n_components, random_state=random_seed),
        'KPCA'     : KernelPCA(kernel=k, n_components=n_components, random_state=random_seed),
    }
    
    transformer_dict = {0:'Identity', 1:'PCA', 2:'KPCA'}
    t = transformer_dict[transformer_type]

    # Pipeline do modelo
    model = Pipeline([
        ('normalizer', normalizer[n]),
        ('tranformer', transformer[t]),
        ('estimator', clf),
    ])

    # Seleção de features
    if len(x) <= n_decision_variables:
        clfnme = clf_name
        ft = np.ones(n_features)  # todas as features ativas
        ft = np.where(ft > 0.5)[0]
    else:
        clfnme = clf_name + '-FS'  # FS = feature selection
        ft = np.array([1 if k > 0.5 else 0 for k in x[n_decision_variables:]])
        if sum(ft) == 0:
            ft[np.random.choice(n_features)] = 1  # garante pelo menos 1 feature
        ft = np.where(ft > 0.5)[0]

    # Configuração de cross-validation
    if task == 'regression':
        if n_splits == -1:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
    elif task == 'forecast':
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        sys.exit('Cross-validation does not defined for estimator ' + clf_name)

    # Avaliação do modelo
    if flag == 'eval':
        if task == 'regression':
            r = -cross_val_score(model, X_train_[:, ft], y_train, cv=cv, scoring=scoring, n_jobs=-1).mean()
        elif task == 'forecast':
            r = cross_val_score(model, X_train_[:, ft], y_train, cv=cv, n_jobs=1, scoring=scoring)
            r = np.abs(r).mean()
        else:
            sys.exit('Cross-validation does not defined for estimator ' + clf_name)
        return r
    
    # Treinamento do modelo e geração de previsões
    elif flag == 'run':
        model.fit(X_train_[:, ft], y_train)
        
        # Previsão para dados de treino
        if task == 'regression':
            if n_samples_test == 0:
                y_p = cross_val_predict(model, X_train_[:, ft], y_train, cv=cv, n_jobs=1)
            else:
                y_p = model.predict(X_train_[:, ft])
        else:
            y_p = model.predict(X_train_[:, ft])

        # Previsão para dados de teste
        if n_samples_test > 0:
            y_t = model.predict(X_test_[:, ft])
        else:
            y_t = np.array([None for i in range(len(y_test))])

        return {
            'Y_TRAIN_TRUE': y_train, 'Y_TRAIN_PRED': y_p,
            'Y_TEST_TRUE': y_test, 'Y_TEST_PRED': y_t,
            'EST_PARAMS': p, 'PARAMS': x, 'EST_NAME': clfnme,
            'SCALES_PARAMS': {'scaler': n},
            'TRANSF_PARAMS': {'tranformer': t, 'kernel': k, 'n_components': n_components},
            'ACTIVE_VAR': ft, 'SCALER': n,
            'SEED': random_seed, 'N_SPLITS': n_splits,
            'OUTPUT': target
        }
    else:
        sys.exit('Model evaluation does not performed for estimator ' + clf_name)

      
#------------------------------------------------------------------------------
def fun_en_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='EN' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 6
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = ElasticNet(random_state=int(random_seed),max_iter=1000,)
  p={
     'alpha': x[4],
     'l1_ratio': x[5],
     #'positive': x[6]<0.5
    }
  clf.set_params(**p)  
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)

#------------------------------------------------------------------------------
def fun_mars_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='MARS'
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = MARS()
  p={
   'allow_linear'             : True, 
   'allow_missing'            : False, 
   'check_every'              : None,
   'enable_pruning'           : True, 
   'endspan'                  : None, 
   'endspan_alpha'            : None, 
   'fast_K'                   : None,
   'fast_h'                   : None, 
   'feature_importance_type'  : None, 
   'max_degree'               : np.round(x[4]),
   'max_terms'                : 1000, 
   'min_search_points'        : 100, 
   'minspan'                  : None,
   'minspan_alpha'            : None, 
   'penalty'                  : x[5], 
   'smooth'                   : False, 
   'thresh'                   : 0.001,  
   'use_fast'                 : True, 
   'verbose'                  : 0, 
   'zero_tol'                 : 1e-12,
    }
  clf.set_params(**p)
  p={
   'max_degree'               : np.round(x[4]),
   'penalty'                  : x[5],
   'max_terms'                : int(round(x[6])),
    }
  ##         0     1  2   3         4     5     6
  #lb_mars= [0,    0, 0., 0,        0,    1,    1,        ] #+ [0.0]*n_features
  #ub_mars= [0,    0, 1., 4,        3,  1e3, 1000,        ] #+ [1.0]*n_features
  
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)

#%%----------------------------------------------------------------------------     
def fun_rf_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='RF' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 6
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = RandomForestRegressor(random_state=int(random_seed),  n_jobs=1,)
  p={
     'n_estimators': int(x[4]+0.995),
     'max_depth': int(x[5]+0.995),
    }
  clf.set_params(**p)  
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_ab_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='AB' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 6
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = AdaBoostRegressor(random_state=int(random_seed),)
  p={
     'n_estimators'   : int(x[4]+0.995),
     'learning_rate'  : x[5],
     #'base_estimator' : DecisionTreeRegressor(max_depth=3),
    }
  clf.set_params(**p)  
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_xgb_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='XGB' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 8
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
 
  cr ={0:'reg:squarederror', 1:'reg:logistic', 2:'binary:logistic',}
  clf = XGBRegressor(random_state=int(random_seed), objective=cr[0], n_jobs=1)
  p={
     'learning_rate'        : int(x[4]*1000)/1000.,
     'n_estimators'         : int(x[5]+0.99), 
     'max_depth'            : int(x[6]+0.99),
     #'reg_alpha'           : x[3],
     'reg_lambda'           : int(x[7]*1000)/1000.,
     #'subsample'           : int(x[5]*1000)/1000,
     #'alpha'               : x[6],
     #'presort'             : ps[0],
     #'max_iter'            : 1000,
     }
    
  clf.set_params(**p)  
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_svr_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='SVR' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = SVR(kernel='rbf', max_iter=1000)
  kernel = {
            0:'rbf', 
            1:'sigmoid', 
            2:'chi2',
            3:'poly', 
            4:'linear', 
            5:'laplacian', 
            }  
  
  _gamma = x[4]#int(x[4]*1000)/1000.
  p={
     'gamma'        :'scale' if _gamma<=0 else _gamma, 
     'C'            : x[5],  
     'epsilon'      : x[6], 
     #'kernel'      : kernel[0],
     #'tol'         : 1e-6,
     #'max_iter'    : 10000,
     #'shrinking'   : False,
     }

  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_lsv_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='SVR-L' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = LinearSVR(max_iter=1000, loss='epsilon_insensitive', random_state=random_seed)
  m=1e4
  p={
     'C'            : int(x[5]*m)/m,  
     'epsilon'      : int(x[4]*m)/m, 
     'loss'         : 'epsilon_insensitive' if x[6]<0.5 else 'squared_epsilon_insensitive'
     #'tol'         : 1e-6,
     #'max_iter'    : 10000,
     }

  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_svm_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='SVM'
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 10
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = SVR(kernel='rbf', max_iter=10000)
  kernel = {
            0:'rbf', 
            2:'sigmoid', 
            4:'chi2',
            5:'laplacian', 
            3:'poly', 
            1:'linear', 
            }  
  
  p={
     'kernel':kernel[int(round(x[4]))], 
     'degree':int(round(x[5])),
     'gamma': 'scale' if x[6]<0 else x[6],
     'coef0':x[7],
     'C':x[8],
     'epsilon':x[9],
     #'tol':1e-6,
     #'max_iter':5000,
  }

  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_gpr_fs(x,*data_args):
  #%%
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='GPR' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 9
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  k1 = int(x[8]*100000)/100000.
  clf = GaussianProcessRegressor(random_state=int(random_seed), optimizer=None, normalize_y=False)
  kernel = {
        0: k1**2 * Matern(length_scale=int(x[5]*1000)/1000., length_scale_bounds=(1e-1, 100.0), nu=int(x[6]*1000)/1000.),
        1: k1**2 * RationalQuadratic(length_scale=int(x[5]*1000)/1000., length_scale_bounds=(1e-1, 100.0), alpha=int(x[6]*1000)/1000.),
        2: k1**2 * RBF(length_scale=int(x[5]*1000)/1000.),
        3: k1**2 * ExpSineSquared(length_scale=int(x[5]*1000)/1000., length_scale_bounds=(1e-1, 100.0), periodicity=int(x[6]*1000)/1000., periodicity_bounds=(1, 1000)),
        4: ConstantKernel(constant_value=k1)*RBF(length_scale=int(x[5]*1000)/1000.)*ExpSineSquared(length_scale=1.0, periodicity=int(x[6]*1000)/1000., periodicity_bounds=(1, 1000)),
        5: ConstantKernel(constant_value=k1)*RBF(length_scale=int(x[5]*1000)/1000.)*RationalQuadratic(length_scale=1., length_scale_bounds=(1e-1, 100.0), alpha=int(x[6]*1000)/1000.),
        }
  p={'kernel': kernel[int(x[4]+0.995)], 'alpha':x[7]}#int(x[7]*1000)/1000.}
  #print(int(x[4]+0.5), x[7], '\t>>\t', p)

  clf.set_params(**p)
  p['k1']=k1
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  #model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
  #%%
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_rbf_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='RBFNN' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = RBFNNRegressor()
  af = {
      0 : 'linear', 
      1 : 'multiquadric', 
      2 : 'inverse',
      3 : 'gaussian', 
      4 : 'cubic', 
      5 : 'quintic', 
      6 : 'thin_plate',     
      7 : 'sigmoid',     
      8 : 'relu',     
      9 : 'swish',     
  }
  mval=1e6
  p={
     'func'    : af[int(x[4]+0.995)],
     'epsilon' : int(x[5]*mval)/mval,
     'smooth'  : int(x[6]*mval)/mval,
    }
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )
  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_lss_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='LSSVR' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 6
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = LSSVR(kernel='rbf')
  
  p={
     'C'     : x[4],
     'gamma' : x[5],#int(x[5]*1000)/1000.,
    }
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_rbn_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='RBN' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 6
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  clf = RBFNN(random_state=random_seed)
  
  p={
     'n_hidden' : int(x[4]+0.995),
     'epsilon'  : x[5],#int(x[5]*1000)/1000.,
    }
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%----------------------------------------------------------------------------     
def fun_elm_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='ELM' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 9
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  
  clf = ELMRegressor(random_state=int(random_seed))
  af = {
      #0 :'tribas', 
      0 :'identity', 
      4 :'relu', 
      5 :'swish',
      #4 :'inv_tribase', 
      #5 :'hardlim', 
      #6 :'softlim', 
      6 :'sigmoid',
      1 :'gaussian', 
      2 :'multiquadric', 
      3 :'inv_multiquadric',
  }

  _alpha=int(x[8]*1000)/1000.
  regressor = None if _alpha<1e-4 else Ridge(alpha=_alpha,random_state=int(random_seed))
  m=1e4
  p={'n_hidden'         : int(round(x[4])), #'alpha':1, 'rbf_width':1,
     'activation_func'  : af[int(x[5]+0.995)], #'alpha':0.5, 
     'alpha'            : int(x[6]*m)/m, 
     'rbf_width'        : int(x[7]*m)/m,
     'regressor'        : regressor,
     }
  clf.set_params(**p)
  p['l2_penalty']=_alpha
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#%%---------------------------------------------------------------------------- 
def fun_mlp_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='MLP' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 10
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  n_hidden = int(round(x[8]))
  hidden_layer_sizes = tuple( int(round(x[9])) for i in range(n_hidden))
  #print(hidden_layer_sizes)
  af = {
          0 :'logistic', 
          1 :'identity', 
          2 :'relu', 
          3 :'tanh',
      }  
  
  s = {
        0: 'lbfgs',
        1: 'sgd',
        2: 'adam',
      }
  
  p={
     'activation': af[int(round(x[4]))],
     'hidden_layer_sizes':hidden_layer_sizes,
     #'alpha':1e-5, 'solver':'lbfgs',
     'solver': s[2],#s[round((x[7]))],
     'alpha': x[6],
     #'learning_rate': 'adaptive',
     'learning_rate_init': x[5],
     }
  
  #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=int(random_seed))
  clf = MLPRegressor(random_state=int(random_seed), warm_start=True, 
                     #early_stopping=True, validation_fraction=0.20,
                     learning_rate='adaptive',  solver='adam',
                      #max_iter=1100,
                      )
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#------------------------------------------------------------------------------
def fun_ann_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='ANN' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 13
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  n_hidden = int(round(x[8]))
  hidden_layer_sizes = tuple( int(round(x[9+i])) for i in range(n_hidden))
  af = {
          0 :'logistic', 
          1 :'identity', 
          2 :'relu', 
          3 :'tanh',
      }  
  
  s = {
        0: 'lbfgs',
        1: 'adam',
        2: 'sgd',
      }
  
  p={
     'activation': af[int(round(x[4]))],
     'hidden_layer_sizes':hidden_layer_sizes,
     #'alpha':1e-5, 'solver':'lbfgs',
     'solver': s[1],#s[int((x[7]))],
     'alpha': x[6],
     #'learning_rate': 'adaptive',
     'learning_rate_init': x[5],
     }
  
  #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=int(random_seed))
  clf = MLPRegressor(random_state=int(random_seed), warm_start=True, 
                     early_stopping=True, validation_fraction=0.20,
                     learning_rate='adaptive',  solver='adam',
                      #max_iter=1100,
                      )
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#------------------------------------------------------------------------------
def fun_knn_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='KNN' 
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 7
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  w = {0 :'uniform', 1 :'distance', }   
  p={
      'p': int(round(x[6])), 
      'n_neighbors': int(round(x[4])),
      'weights':w[int(round(x[5]))],
  }
     
  clf = KNeighborsRegressor()
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)

#------------------------------------------------------------------------------
def fun_mcn_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='MCN'
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 9
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  n_hidden = int(round(x[6]))   
  hidden_layer_sizes = [ int(round(x[7+i])) for i in range(n_hidden)]
  #hidden_layer_sizes = [ int(round(x[2])) for i in range(n_hidden)]
  con = {0: 'mlgraph', 1:'tmlgraph',}
  p={
     'connectivity': con[int(round(x[4]))],
     'bias':bool(round(x[5])),
     #'renormalize':bool(round(x[2])),
     'n_hidden':hidden_layer_sizes, 
     #'algorithm':['tnc', 'l-bfgs', 'sgd', 'rprop', 'genetic'],
     }
  clf = MLPR(algorithm = 'sgd',max_iter=None, renormalize=True)  
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


#------------------------------------------------------------------------------
def fun_krr_fs(x,*data_args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = data_args
    
  clf_name              ='KRR'
  normalizer_type       = int(x[0]+0.995)
  transformer_type      = int(x[1]+0.995)
  n_components          = int(x[2]*n_features+1)
  kernel_type           = int(x[3]+0.995)
  n_decision_variables  = 10
  
  normalizer_args       = (normalizer_type,)
  transformer_args      = (transformer_type, n_components, kernel_type)
  
  kernel = {2:'linear', 3:'poly', 0:'rbf', 1:'sigmoid', 4:'laplacian', 5:'chi2'}  
  p={
     'alpha':x[4],
     'kernel':kernel[int(round(x[5]))], 
     'gamma':x[6],
     'degree':int(round(x[7])),
     'coef0':x[8],
     'kernel_params':{'C':x[9], 'max_iter':15000},
     }
  clf = KernelRidge(kernel='rbf',)
  clf.set_params(**p)
  estimator_args=(clf_name, n_decision_variables, p, clf, )

  return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)
#------------------------------------------------------------------------------

def fun_pr_fs(x,*args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = args
    
  clf_name='PR' 
  normalizer_type = round(x[0]+0.495)#; sys.exit('exit '+clf_name)
  n_decision_variables = 6

  p={
     'alpha': x[3],
     'l1_ratio': x[4],
     'positive': x[5]<0.5,
    }  
    
  _clf = ElasticNet(random_state=int(random_seed),max_iter=5000)
  _clf.set_params(**p)

  p['degree']= int(x[1]+0.5)
  p['interaction_only']= x[2]<0.50
  clf = Pipeline([
                  ('poly', PolynomialFeatures(degree=p['degree'], interaction_only=p['interaction_only'])),
                  ('linear', _clf),
                 ])
  return model_base_evaluation(x, p, clf, clf_name, n_decision_variables, normalizer_type, *args)        
#------------------------------------------------------------------------------
def fun_krr_fs_(x,*args):
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = args
    
  clf_name='KRR' 
  normalizer_type = round(x[0]+0.495)#; sys.exit('exit '+clf_name)
  n_decision_variables = 6
  
  clf = KernelRidge(kernel='rbf',)
  
  kernel = {2:'linear', 3:'poly', 0:'rbf', 1:'sigmoid', 4:'laplacian', 5:'chi2'}  
  p={
     'alpha':x[1],
     'kernel':kernel[int(round(x[2]))], 
     'gamma':x[3],
     'degree':int(round(x[4])),
     'coef0':x[5],
     'kernel_params':{'C':x[6], 'max_iter':15000},
     }
  clf.set_params(**p)
 
  return model_base_evaluation(x, p, clf, clf_name, n_decision_variables, normalizer_type, *args)   
#------------------------------------------------------------------------------      

#------------------------------------------------------------------------------      
def fun_dt_fs(x,*args):
  X, y, flag, n_splits, random_seed, scoring = args 
  n_samples, n_var = X.shape
  _clf = DecisionTreeRegressor(random_state=int(random_seed),)
  #clf = RandomForestRegressor(random_state=int(random_seed), n_estimators=100)
  p={
     'criterion': 'mse' if x[2] < 0.5 else 'mae',
     #'min_samples_split': int(x[2]),
     'max_depth': None if x[3]<1 else int(x[3]),
     #'n_estimators': int(x[1]),
    }
  _clf.set_params(**p)
  p['degree']= int(x[0]+0.5)
  p['interaction_only']= x[1]<0.50
  clf = Pipeline([
                  ('poly', PolynomialFeatures(degree=p['degree'], interaction_only=p['interaction_only'])),
                  ('linear', _clf),
                 ])
    
    
  if len(x)<=4:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)[0]
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)[0]

  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).mean()
    except:
        r=1e12
    
    print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'DT',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'N_SPLITS':n_splits,}
#------------------------------------------------------------------------------
def fun_vr_fs(x,*args):
  X, y, flag, n_splits, random_seed, scoring = args 
  n_samples, n_var = X.shape
  kernel = {
            0:'rbf', 
            1:'sigmoid', 
            2:'chi2',
            3:'laplacian', 
            4:'poly', 
            5:'linear', 
            }
    
  af = {
      #0 :'tribas', 
      0 :'identity', 
      1 :'relu', 
      2 :'swish',
      #4 :'inv_tribase', 
      #5 :'hardlim', 
      #6 :'softlim', 
      3 :'gaussian', 
      4 :'multiquadric', 
      5 :'inv_multiquadric',
    }


  _clf=[
         ElasticNet(random_state=int(random_seed),),
         DecisionTreeRegressor(random_state=int(random_seed),),
         #SVR(kernel='rbf', max_iter=5000),
         #ELMRegressor(random_state=int(random_seed)),
     ]
  
  _p=[
        {
        'alpha': x[0],
        'l1_ratio': x[1],
        'positive': x[2]<0.5,
        },
        {
        'max_depth': None,
        'criterion': 'mse',
        },
#        {
#        'gamma':'scale' if x[3]<0 else x[3], 
#        'C':x[4],  
#        'epsilon':x[5], 
#        'kernel':kernel[0],
#        },          
#        {
#        'n_hidden':int(x[6]), #'alpha':1, 'rbf_width':1,
#        'activation_func': af[int(x[7]+0.5)], #'alpha':0.5, 
#        'alpha':x[8], 
#        'rbf_width':x[9],
#        },
     ]
  
  
  for k in range(len(_clf)):
      _clf[k].set_params(**_p[k])
     
  _estimators=[]      
  for k in range(len(_clf)):
      _estimators.append( ('reg_'+str(k), _clf[k]) )
      
     
  clf = VotingRegressor(estimators=_estimators)
  p={'weights':None} 
  clf.set_params(**p)
  
  if len(x)<=10:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)[0]
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)[0]

  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).max()
    except:
        r=1e12
    
    #print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'VR',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'N_SPLITS':n_splits,}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------   
def fun_rxe_fs(x,*args):
  X, y, flag, n_splits, random_seed, scoring = args 
  clf = ELMRegressor(random_state=int(random_seed),  alpha=0)
  n_samples, n_var = X.shape
  
  _alpha=int(x[2]*10000)/10000.
  regressor = None if _alpha<1e-4 else Ridge(alpha=_alpha,random_state=int(random_seed))
  p={'n_hidden':int(x[0]/1.)*1, #'alpha':1, 'rbf_width':1,
     'rbf_width':int(x[1]*100)/100.,
     'regressor':regressor,
     }
  clf.set_params(**p)
  
  p['l2_penalty']=_alpha
    
  #x[2::] = [1 if k>0.5 else 0 for k in x[4::]]
  if len(x)<=3:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)[0]
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)[0]
      
     
  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).max()
    except:
        r=1e12
    
    #print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'RXE',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'N_SPLITS':n_splits,}
#------------------------------------------------------------------------------   
def fun_cat_fs(x,*args):
  X, y, flag, n_splits, random_seed, scoring = args 
  clf = CatBoostRegressor(random_state=int(random_seed),verbose=0)
  
  n_samples, n_var = X.shape

#  cr ={
#        0:'reg:linear',
#        1:'reg:logistic',
#        2:'binary:logistic',
#       }
       
  #x=[0.1, 200, 5, 2.5, 10.0, 0.8, ]
  p={
     'learning_rate': x[0],
     'n_estimators':int(round(x[1])), 
     'depth':int(round(x[2])),
     'loss_function':'RMSE',
     'l2_leaf_reg':x[3],
     'bagging_temperature':x[4],
     #'boosting_type':'Pĺain',
     #'colsample_bytree':x[3],
     #'min_child_weight':int(round(x[4])),
     #'bootstrap_type':'Bernoulli',
     #'subsample':int(x[5]*1000)/1000,
     ##'alpha':x[6],
     #'objective':cr[0],
     ##'presort':ps[0],
     }
    
  clf.set_params(**p)
  if len(x)<=6:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)[0]
      
  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X[:,ft].squeeze(), y.ravel(), cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).max()
    except:
        r=1e12
    
    print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'CAT',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'N_SPLITS':n_splits,}

#------------------------------------------------------------------------------   
def fun_hgb_fs(x,*args):
  #X, y, flag, n_splits, random_seed, scoring = args
  (X_train, y_train, X_test, y_test, flag, task,  n_splits, 
                    random_seed, scoring, target,
                    n_samples_train, n_samples_test, n_features) = args
   
  clf = HistGradientBoostingRegressor(random_state=int(random_seed), 
                                  loss='least_squares',)
  n_samples, n_var = X_train.shape
  p={
     'learning_rate': x[0],
     'max_iter': int(round(x[1])),
     'l2_regularization': x[2],
     }
    
  
  clf.set_params(**p)
  if len(x)<=3:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)[0]
  clf.set_params(**p)
  if len(x)<=6:
      ft = np.array([1 for i in range(n_var)])
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])

  ft = np.where(ft>0.5)[0]
      
  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X_train[:,ft].squeeze(), y_train.squeeze(), cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).max()
    except:
        r=1e12
    
    #print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'HGB',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X_train, 'SEED':random_seed, 'N_SPLITS':n_splits,}
#------------------------------------------------------------------------------   
def fun_krr_fs_(x,*args):
  X, y, flag, n_splits, random_seed, scoring = args 
  clf = KernelRidge(kernel='rbf',)
  n_samples, n_var = X.shape
  
  kernel = {2:'linear', 3:'poly', 0:'rbf', 1:'sigmoid', 4:'laplacian', 5:'chi2'}  
  p={
     'alpha':x[0],
     'kernel':kernel[int(round(x[1]))], 
     'gamma':x[2],
     'degree':int(round(x[3])),
     'coef0':x[4],
     'kernel_params':{'C':x[5], 'max_iter':4000},
     }
  clf.set_params(**p)
  n_param=len(p)
  if len(x)<=n_param:
      ft = np.array([1 for i in range(n_var)])
      ft = np.where(ft>0.5)[0]
  else:
      ft = np.array([1 if k>0.5 else 0 for k in x[2::]])
      ft = np.where(ft>0.5)[0]
    
  cv=KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))
  if flag=='eval':
    try:
        r=cross_val_score(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1, scoring=scoring)
        r=np.abs(r).max()
    except:
        r=1e12
    
    print(r,'\t',p,)  #'\t',ft)  
    return r
  else:
    y_p  = cross_val_predict(clf,X[:,ft].squeeze(), y, cv=cv, n_jobs=1)
    return {'Y_TRUE':y, 'Y_PRED':y_p, 'EST_PARAMS':p, 'PARAMS':x, 'EST_NAME':'KRR',
              'ESTIMATOR':clf, 'ACTIVE_VAR':ft, 'DATA':X, 'SEED':random_seed, 'N_SPLITS':n_splits,}

#%%----------------------------------------------------------------------------                
import pygmo as pg
class evoML:
    def __init__(self, args, fun, lb, ub):
         self.args = args
         self.obj = fun
         self.lb, self.ub= lb, ub
         
    def fitness(self, x):     
        self.res=self.obj(x,*self.args)
        return [self.res]
    
    def get_bounds(self):
         return (self.lb, self.ub)  
     
    def get_name(self):
         return "evoML"
        
#%%----------------------------------------------------------------------------   
# Importa bibliotecas para otimização, manipulação de arquivos, gráficos e sistema operacional
from scipy.optimize import differential_evolution as de, shgo, dual_annealing
import glob as gl
import pylab as pl
import os

# Prefixo para salvar arquivos
basename='petlm__'


     
#%%----------------------------------------------------------------------------   
pd.options.display.float_format = '{:.3f}'.format

# Parâmetros gerais para otimização e validação
pop_size    = 50           # tamanho da população em algoritmos evolutivos
max_iter    = 50           # número máximo de gerações
n_run       = 30           # numero máximo de execuções independetes
n_splits    = 3            # número de folds para validação cruzada
scoring     = 'neg_root_mean_squared_error'  # métrica de avaliação
random.seed(42)            # semente aleatória

# Loop sobre múltiplas execuções (seed diferentes)
for run in range(n_run):
    random_seed = random.randint(0,1000)  # semente pseudo-aleatória para reprodutibilidade
    dataset = read_data_char(seed=random_seed, target='Biochar Yield' ) # target='Biochar HHV'
    # dataset = read_data_char(seed=random_seed, target='Biochar HHV' )
    
    # Preparar diretório para salvar resultados
    dr = dataset['name'].replace(' ','_').replace("'","").lower()
    path = './pkl_petlm_'+dr+'/'
    os.system('mkdir  '+path)  # cria pasta se não existir
    
        
    # Inicializa variáveis do dataset
    dataset_name = dataset['name']
    target       = dataset['target_names']
    y_train, y_test = dataset['y_train'], dataset['y_test']
    X_train, X_test = dataset['X_train'], dataset['X_test']
    n_samples_train, n_features = dataset['n_samples'], dataset['n_features']
    task, normalize = dataset['task'], dataset['normalize']
    n_samples_test = len(y_test)
    
    # Exibe informações do dataset
    s = '\n' + '='*80 + '\n'
    s += f'Dataset                    : {dataset_name} -- {target}\n'
    s += f'Number of training samples : {n_samples_train}\n'
    s += f'Number of testing  samples : {n_samples_test}\n'
    s += f'Number of features         : {n_features}\n'
    s += f'Normalization              : {normalize}\n'
    s += f'Task                       : {dataset["task"]}\n'
    s += '='*80 + '\n'
    print(s)

    #------------------------------------------------------------------
    # Limites inferiores e superiores de hiperparâmetros para diversos modelos     
    #------------------------------------------------------------------ 
    lb_en = [0, 0, 0., 0, 1e-6, 0, ] #+ [0.0]*n_features 
    ub_en = [0, 0, 1., 4, 2e+0, 1, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_elm = [0, 0, 0., 0, 1e-0, 0, 1, 1., 0.0] #+ [0.0]*n_features 
    ub_elm = [0, 0, 1., 4, 5e+2, 6, 1, 10., 1e4] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_lsv = [0, 0, 0., 0, 1e-4, 1e-4, 0, ] #+ [0.0]*n_features 
    ub_lsv = [0, 0, 1., 4, 1e+0, 1e+3, 1, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------         
    lb_mars= [0,    0, 0., 0,        0,    1,    1,        ] #+ [0.0]*n_features
    ub_mars= [0,    0, 1., 4,        3,  1e3, 1000,        ] #+ [1.0]*n_features
    #------------------------------------------------------------------ 
    lb_mlp = [0, 0, 0., 0, 0, 1e-6, 1e-8, 0, 1, 1] #+ [0.0]*n_features 
    ub_mlp = [0, 0, 1., 4, 3, 1.0, 1e+0, 2, 3, 50] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_gpr = [0, 0, 0., 0, 0, 1e-0, 1e-3, 1e-8, 0, ] #+ [0.0]*n_features 
    ub_gpr = [0, 0, 1., 4, 0, 5e+2, 1e+2, 1., 1e1, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_xgb = [0, 0, 0., 0, 1e-6, 10, 1, 0., ] #+ [0.0]*n_features 
    ub_xgb = [0, 0, 1., 4, 1e+0, 100, 10, 5., ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_svr = [0, 0, 0., 0, 1e-5, 1e-1, 1e-5, ] #+ [0.0]*n_features 
    ub_svr = [0, 0, 1., 4, 1e+0, 3e+3, 1e+1, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_rf = [0, 0, 0., 0, 1, 1, ] #+ [0.0]*n_features 
    ub_rf = [0, 0, 1., 4, 100, 20, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_ab = [0, 0, 0., 0, 1, 0, ] #+ [0.0]*n_features 
    ub_ab = [0, 0, 1., 4, 100, 1, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_rbf = [0, 0, 0., 0, 0 , 1e-5, 1e-5, ] #+ [0.0]*n_features 
    ub_rbf = [0, 0, 1., 4, 9 , 2e+2, 10, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_rbn = [0, 0, 0., 0, 1, 0, ] #+ [0.0]*n_features 
    ub_rbn = [0, 0, 1., 4, 100, 1, ] #+ [1.0]*n_features 
    #----------------------------------------------------------------- 
    lb_lss = [0, 0, 0., 0, 1, 0, ] #+ [0.0]*n_features 
    ub_lss = [0, 0, 1., 4, 1e3, 1e2, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_ann = [0, 0, 0., 0, 0, 1e-6, 1e-8, 0, 1, 1, 1, 1, 1] #+ [0.0]*n_features 
    ub_ann = [2, 0, 1., 4, 3, 1.0, 1e+0, 2, 2,100,100,100,100] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_knn = [0, 0, 0., 0, 1, 0, 1, ] #+ [0.0]*n_features 
    ub_knn = [0, 0, 1., 4, 50, 1, 3, ] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_svm = [0, 0, 0., 0, 0, 1, -0.1, 1e-6, 1e-6, 1e-6,] #+ [0.0]*n_features 
    ub_svm = [0, 0, 1., 4, 3, 5, 2, 1e+4, 1e+4, 4,] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_mcn = [0, 0, 0., 0, 0, 0, 1, 1, 1, 1,] #+ [0.0]*n_features 
    ub_mcn = [0, 0, 1., 4, 1, 1, 3, 20, 20, 20,] #+ [1.0]*n_features 
    #------------------------------------------------------------------ 
    lb_krr = [0, 0, 0., 0, 0., 0, 0., 1, 0, 1e-6] #+ [0.0]*n_features 
    ub_krr = [0, 0, 1., 4, 1., 4, 10., 5, 1e2, 1e+3] #+ [1.0]*n_features
    #-------------------------------------------------------------------
    lb_rxe = [0, 0, 0., 0, 0., 0, 0., 1, 0, 1e-6] #+ [0.0]*n_features 
    ub_rxe = [0, 0, 1., 4, 1., 4, 10., 5, 1e2, 1e+3] #+ [1.0]*n_features
    #-------------------------------------------------------------------
    lb_pr = [0, 0, 0., 0, 0., 0, 0., 1, 0, 1e-6] #+ [0.0]*n_features 
    ub_pr = [0, 0, 1., 4, 1., 4, 10., 5, 1e2, 1e+3] #+ [1.0]*n_features
    #-------------------------------------------------------------------
    lb_hgb = [0, 0, 0., 0, 0., 0, 0., 1, 0, 1e-6] #+ [0.0]*n_features 
    ub_hgb = [0, 0, 1., 4, 1., 4, 10., 5, 1e2, 1e+3] #+ [1.0]*n_features
    #-------------------------------------------------------------------
    lb_vr = [0, 0, 0., 0, 0., 0, 0., 1, 0, 1e-6] #+ [0.0]*n_features 
    ub_vr = [0, 0, 1., 4, 1., 4, 10., 5, 1e2, 1e+3] #+ [1.0]*n_features
    #------------------------------------------------------------------
    

    #------------------------------------------------------------------
    # Argumentos gerais passados para funções de otimização
    args = (X_train, y_train, X_test, y_test, 'eval', task, n_splits, 
            int(random_seed), scoring, target, 
            n_samples_train, n_samples_test, n_features)
    
    #------------------------------------------------------------------
    # Lista de estimadores a serem otimizados
    optimizers=[ 
        # ('EN' , lb_en, ub_en, fun_en_fs, args, random_seed,), 
        ('ELM' , lb_elm, ub_elm, fun_elm_fs, args, random_seed,), # OK 
        # ('SVR-L', lb_lsv, ub_lsv, fun_lsv_fs, args, random_seed,), # OK 
        # ('MLP' , lb_mlp, ub_mlp, fun_mlp_fs, args, random_seed,), # OK 
        ('MARS' ,lb_mars,ub_mars,fun_mars_fs, args, random_seed,),    # OK
        ('XGB' , lb_xgb, ub_xgb, fun_xgb_fs, args, random_seed,), # OK # 
        # ('KNN' , lb_knn, ub_knn, fun_knn_fs, args, random_seed,), # OK 
        # ('GPR' , lb_gpr, ub_gpr, fun_gpr_fs, args, random_seed,), # OK 
        # ('KRR' , lb_krr, ub_krr, fun_krr_fs, args, random_seed,), # OK 
        # ('AB' , lb_ab, ub_ab, fun_ab_fs, args, random_seed,), # OK 
        ('RF' , lb_rf, ub_rf, fun_rf_fs, args, random_seed,), # OK 
        ('SVR' , lb_svr, ub_svr, fun_svr_fs, args, random_seed,), # OK 
        # ('RBF' , lb_rbf, ub_rbf, fun_rbf_fs, args, random_seed,), # OK 
        # ('LSSVR', lb_lss, ub_lss, fun_lss_fs, args, random_seed,), # OK # 
        # ('RBN' , lb_rbn, ub_rbn, fun_rbn_fs, args, random_seed,), # OK 
        ('ANN' , lb_ann, ub_ann, fun_ann_fs, args, random_seed,), # OK #
        # ('SVM' , lb_svm, ub_svm, fun_svm_fs, args, random_seed,), # OK 
        # ('MCN' , lb_mcn, ub_mcn, fun_mcn_fs, args, random_seed,), # 
        # ('PR' , lb_pr, ub_pr, fun_pr_fs, args, random_seed,), # OK 
        # ('DT' , lb_dt, ub_dt, fun_dt_fs, args, random_seed,), 
        # ('KRR' , lb_krr, ub_krr, fun_krr_fs, args, random_seed,), # OK 
        # ('HGB' , lb_hgb, ub_hgb, fun_hgb_fs, args, random_seed,), # OK 
        # ('VR' , lb_vr, ub_vr, fun_vr_fs, args, random_seed,), # OK 
        #('VC' , lb_vc , ub_vc , fun_vc_fs , args, random_seed,), 
        #('BAG' , lb_bag, ub_bag, fun_bag_fs, args, random_seed,), 
        # ('RXE' , lb_rxe, ub_rxe, fun_rxe_fs, args, random_seed,), # OK 
        # ('CAT' , lb_cat, ub_cat, fun_cat_fs, args, random_seed,), 
        ]
    
    #------------------------------------------------------------------
    # Loop sobre cada estimador
    for (clf_name, lb, ub, fun, args, random_seed) in optimizers:
        np.random.seed(random_seed)
        
        # Mostra informações do estimador e dataset
        s = '-'*80 + '\n'
        s += f'Estimator                  : {clf_name}\n'
        s += f'Function                   : {str(fun)}\n'
        s += f'Dataset                    : {dataset_name} -- {target}\n'
        s += f'Run                        : {run}\n'
        s += f'Random seed                : {random_seed}\n'
        
        #------------------------------------------------------------------
        # Lista de algoritmos evolutivos a serem aplicados
        algorithms = [
            # ('DE', pg.algorithm(pg.de(gen=max_iter, variant=2, seed=random_seed))),
            # ('PSO', pg.algorithm(pg.pso(gen=max_iter, seed=random_seed))),
            # ('GWO', pg.algorithm(pg.gwo(gen=max_iter, seed=random_seed))),
            # ('ABC', pg.algorithm(pg.bee_colony(gen=int(max_iter/2), seed=random_seed))),
            ('GA', pg.algorithm(pg.sga(gen=max_iter, m=0.10, cr=0.95, crossover="single", mutation="uniform", seed=random_seed))),
            ('xNES', pg.algorithm(pg.xnes(gen=max_iter, memory=False, force_bounds=True, seed=random_seed))),
        ]
        
        #------------------------------------------------------------------
        # Loop sobre algoritmos evolutivos
        for algo_name, algo in algorithms:
            list_results = []  # lista para armazenar resultados
            print(algo_name)
            
            s += f'Optimizer                  : {algo.get_name()}\n'
            s += '-'*80 + '\n'
            print(s)
            
            # Configura problema para otimização
            algo.set_verbosity(16)
            prob = pg.problem(evoML(args, fun, lb, ub))  # função objetivo
            
            # População inicial
            pop = pg.population(prob=prob, size=pop_size, seed=random_seed)
            
            # Executa otimização
            pop = algo.evolve(pop)
            
            # Recupera melhor solução
            xopt = pop.champion_x
            
            # Avalia solução no dataset real
            args1 = (X_train, y_train, X_test, y_test, 'run', task, n_splits, 
                     int(random_seed), scoring, target,
                     n_samples_train, n_samples_test, n_features)
            sim = fun(xopt, *args1)
            
            # Adiciona informações do algoritmo e dataset
            sim['ALGO'] = algo.get_name()
            sim['OUTPUT'] = sim['TARGET'] = target
            sim['ACTIVE_VAR_NAMES'] = dataset['feature_names'][sim['ACTIVE_VAR']]
            sim['RUN'] = run
            sim['DATASET_NAME'] = dataset_name
            sim['SEED'] = random_seed
            list_results.append(sim)
            
            # Salva resultados em arquivo pickle
            data = pd.DataFrame(list_results)
            ds_name = dataset_name.replace('/','_').replace("'","").lower()
            tg_name = target.replace('/','_').replace("'","").lower()
            algo_name_clean = sim['ALGO'].split(':')[0] 
            pk = (path + basename + '_run_' + f"{run:02d}_" +
                  f"{ds_name}" + "_" + f"{sim['EST_NAME']}" + "_" +
                  f"{algo_name_clean}" + "_" + f"{tg_name}" + '.pkl')
            pk = pk.replace(' ','_').replace("'","").lower()
            data.to_pickle(pk)
            
