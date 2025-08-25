#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
np.int = np.int64  # compat: alguns pacotes esperam np.int
import pandas as pd

# Modelagem e avaliação (scikit-learn)
from sklearn.model_selection import (
    GridSearchCV, KFold, cross_val_predict, TimeSeriesSplit, cross_val_score,
    LeaveOneOut, KFold, StratifiedKFold, cross_val_predict, train_test_split
)
from sklearn.metrics import (
    r2_score, mean_squared_error, max_error, mean_squared_error,
    mean_absolute_error, median_absolute_error, accuracy_score,
    f1_score, precision_score,
)
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import (
    MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler,
    MaxAbsScaler, FunctionTransformer, QuantileTransformer,
)
from sklearn.pipeline import Pipeline

# Estimadores (alguns usados por outros blocos)
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import (
    ElasticNet, Ridge, PassiveAggressiveRegressor, LogisticRegression,
    BayesianRidge, LinearRegression,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel,
)

import re
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler

# Estimadores custom do projeto (usados em blocos seguintes)
from util.ELM import ELMRegressor, ELMRegressor
from util.MLP import MLPRegressor as MLPR
from util.RBFNN import RBFNNRegressor, RBFNN
from util.LSSVR import LSSVR

from scipy import stats
from hydroeval import kge, nse

# ====== Configuração global simples ======
pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings('ignore')

import sys, getopt
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

# Permite escolher run inicial via argumento "-r"
if len(arguments) > 0:
    if arguments[0] == '-r':
        run0 = int(arguments[1])
        n_runs = run0
else:
    run0, n_runs = 45, 50

# =============================================================================
# MÉTRICAS & UTILITÁRIOS
# =============================================================================

def accuracy_log(y_true, y_pred):
    """Calcula a "acurácia log" (% de previsões cujo erro log10 relativo < 0.3).

    Parâmetros
    ----------
    y_true : array-like
        Valores reais/observados.
    y_pred : array-like
        Valores previstos pelo modelo.

    Retorna
    -------
    float
        Percentual de acertos (0–100).
    """
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true / y_pred)) < 0.3).sum() / len(y_true) * 100


def rms(y_true, y_pred):
    """RMS do erro em escala log10.

    Retorna o erro quadrático médio (root mean square) do log10(y_pred/y_true).
    Útil quando a escala dos dados é multiplicativa.
    """
    y_true = np.abs(np.array(y_true))
    y_pred = np.abs(np.array(y_pred))
    return ((np.log10(y_pred / y_true) ** 2).sum() / len(y_true)) ** 0.5


def RMSE(y, y_pred):
    """Root Mean Squared Error (RMSE) em escala original.

    Parâmetros
    ----------
    y : array-like
        Valores reais.
    y_pred : array-like
        Valores previstos.
    """
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y - y_pred
    return np.sqrt(np.mean(np.power(error, 2)))


def RRMSE(y, y_pred):
    """Relative RMSE (%): RMSE normalizado pela média de *y* vezes 100."""
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    return RMSE(y, y_pred) * 100 / np.mean(y)


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    return np.mean(np.abs(y_pred - y_true) / np.abs(y_true)) * 100


def lhsu(xmin, xmax, nsample):
    """Latin Hypercube Sampling (uniforme).

    Gera *nsample* amostras no hipercubo definido por *xmin* e *xmax*,
    com estratificação por dimensão.

    Parâmetros
    ----------
    xmin : array-like, shape (n_var,)
        Limites inferiores de cada variável.
    xmax : array-like, shape (n_var,)
        Limites superiores de cada variável.
    nsample : int
        Número de amostras desejado.

    Retorna
    -------
    np.ndarray, shape (nsample, n_var)
        Amostras geradas.
    """
    nvar = len(xmin)
    ran = np.random.rand(nsample, nvar)
    s = np.zeros((nsample, nvar))
    for j in range(nvar):
        idx = np.random.permutation(nsample)
        P = (idx.T - ran[:, j]) / nsample
        s[:, j] = xmin[j] + P * (xmax[j] - xmin[j])
    return s


# =============================================================================
# WRAPPER DE PROBLEMA PARA PYGMO
# =============================================================================
import pygmo as pg

class evoML:
    """Wrapper de problema (single-objective) para *pygmo*.

    Esta classe recebe uma função-objetivo *self.obj* (por exemplo, fun_en_fs,
    fun_mlp_fs, etc.), os argumentos de dados e os limites inferior/superior
    de busca. Ela expõe os métodos esperados pelo *pygmo*:
      • fitness(x): calcula e retorna a função-objetivo (lista com 1 valor)
      • get_bounds(): retorna tupla (lb, ub)
      • get_name(): nome amigável do problema
    """
    def __init__(self, args, fun, lb, ub):
        self.args = args
        self.obj = fun
        self.lb, self.ub = lb, ub

    def fitness(self, x):
        """Retorna a avaliação da função objetivo para o vetor *x*.

        *pygmo* espera uma lista (mesmo para objetivo único), por isso
        embrulhamos o escalar em uma lista.
        """
        self.res = self.obj(x, *self.args)
        return [self.res]

    def get_bounds(self):
        """Limites inferior e superior usados pelo otimizador."""
        return (self.lb, self.ub)

    def get_name(self):
        return "evoML"


# =============================================================================
# AVALIAÇÃO BASE (PIPELINE: normalização → transformação → estimador)
# =============================================================================

def model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args):
    """Núcleo de avaliação de modelos (cross‑validation e/ou predição).

    Esta função constrói um *Pipeline* com: normalizador → transformador →
    estimador, executa *cross‑validation* quando `flag == 'eval'` ou realiza
    *fit/predict* quando `flag == 'run'`. É usada pelos *wrappers* de cada
    estimador (fun_en_fs, fun_rf_fs, fun_mlp_fs, etc.).

    Parâmetros
    ----------
    x : array-like
        Vetor de decisão completo (inclui hiperparâmetros e, em algumas
        funções, máscara de seleção de variáveis ao final).
    data_args : tuple
        (X_train_, y_train, X_test_, y_test, flag, task, n_splits,
         random_seed, scoring, target, n_samples_train, n_samples_test,
         n_features)
    estimator_args : tuple
        (clf_name, n_decision_variables, p, clf)
    normalizer_args : tuple
        (normalizer_type,)
    transformer_args : tuple
        (transformer_type, n_components, kernel_type)

    Retorna
    -------
    float or dict
        • Se `flag == 'eval'`: retorna a métrica de *scoring* (com sinal
          ajustado conforme o *scoring*), para uso do otimizador.
        • Se `flag == 'run'`: retorna dicionário com previsões e metadados
          (parâmetros, features ativas, seed, etc.).
    """
    (
        X_train_, y_train, X_test_, y_test, flag, task, n_splits,
        random_seed, scoring, target, n_samples_train, n_samples_test,
        n_features,
    ) = data_args
    (normalizer_type,) = normalizer_args
    (transformer_type, n_components, kernel_type) = transformer_args
    (clf_name, n_decision_variables, p, clf) = estimator_args

    # ---------- Normalizador ----------
    normalizer = {
        'None': FunctionTransformer(),
        'MinMax': MinMaxScaler(),
        'MaxAbs': MaxAbsScaler(),
        'Standard': StandardScaler(),
        'Log': FunctionTransformer(np.log1p),
        'Quantile Norm.': QuantileTransformer(
            n_quantiles=n_features, output_distribution='normal'
        ),
        'Quantile Unif.': QuantileTransformer(
            n_quantiles=n_features, output_distribution='uniform'
        ),
        'Poly': PolynomialFeatures(),
    }
    # Mapeia o código inteiro (0..7) para rótulo humano
    normalizer_dict = {
        0: 'None', 1: 'MinMax', 3: 'MaxAbs', 2: 'Standard', 4: 'Log',
        5: 'Quantile Norm.', 6: 'Quantile Unif.', 7: 'Poly',
    }
    n = normalizer_dict[normalizer_type]

    # ---------- Transformador ----------
    kernel = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid', 4: 'cosine'}
    transformer = {
        # 0 → identidade (sem transformação)
        0: FunctionTransformer(),
        # 1 → PCA via KernelPCA, com kernel escolhido em `kernel_type`
        1: KernelPCA(n_components=n_components, kernel=kernel[kernel_type]),
    }
    t = transformer[transformer_type]
    k = kernel[kernel_type]

    # ---------- Seleção de variáveis (máscara no final de x) ----------
    # Regra do projeto: se len(x) <= n_decision_variables → usa todas as
    # features; do contrário, usa 0/1 a partir da posição n_decision_variables.
    if len(x) <= n_decision_variables:
        ft = np.ones(n_features, dtype=int)
    else:
        ft = np.array([1 if v > 0.5 else 0 for v in x[n_decision_variables:]], dtype=int)
    ft = np.where(ft > 0.5)[0]

    # ---------- Monta o pipeline final ----------
    model = Pipeline([
        ('normalizer', normalizer[n]),
        ('transformer', t),
        ('estimator', clf),
    ])

    # ---------- Validação cruzada / Execução ----------
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_seed))

    if flag == 'eval':
        # Avaliação para o otimizador (retorna escalar)
        try:
            # cross_val_score segue a convenção do sklearn de sinal: para
            # *loss* (ex.: neg RMSE) os valores são negativos; usamos abs().
            r = cross_val_score(
                model, X_train_[:, ft], y_train, cv=cv, n_jobs=1, scoring=scoring
            )
            r = np.abs(r).mean()
        except Exception:
            r = 1e12  # penaliza configurações inválidas
        return r

    elif flag == 'run':
        # Treina o pipeline com a partição de treino
        model.fit(X_train_[:, ft], y_train)

        # Previsão para treino (CV ou direto) e teste (se houver)
        if task == 'regression':
            if n_samples_test == 0:
                y_p = cross_val_predict(
                    model, X_train_[:, ft], y_train, cv=cv, n_jobs=1
                )
            else:
                y_p = model.predict(X_train_[:, ft])
        else:
            y_p = model.predict(X_train_[:, ft])

        if n_samples_test > 0:
            y_t = model.predict(X_test_[:, ft])
        else:
            y_t = np.array([None for _ in range(len(y_test))])

        # Pacote de resultados para persistência/análise posterior
        return {
            'Y_TRAIN_TRUE': y_train,
            'Y_TRAIN_PRED': y_p,
            'Y_TEST_TRUE': y_test,
            'Y_TEST_PRED': y_t,
            'EST_PARAMS': p,
            'PARAMS': x,
            'EST_NAME': clf_name,
            'SCALES_PARAMS': {'scaler': n},
            'TRANSF_PARAMS': {
                'tranformer': t,
                'kernel': k,
                'n_components': n_components,
            },
            'ACTIVE_VAR': ft,
            'SCALER': n,
            'SEED': random_seed,
            'N_SPLITS': n_splits,
            'OUTPUT': target,
        }

    else:
        # Sinaliza erro de uso
        raise SystemExit(
            f"Model evaluation not defined for estimator {clf_name} (flag={flag})."
        )


# =============================================================================
# ELASTIC NET
# =============================================================================
def fun_en_fs(x, *data_args):
    """Wrapper para Elastic Net (EN).

    Hiperparâmetros no vetor x:
        x[0] → tipo de normalizador
        x[1] → tipo de transformador
        x[2] → nº de componentes
        x[3] → kernel (se aplicável)
        x[4] → alpha
        x[5] → l1_ratio
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'EN'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = ElasticNet(random_state=int(random_seed), max_iter=1000)
    p = {
        'alpha': x[4],
        'l1_ratio': x[5]
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# RIDGE REGRESSION
# =============================================================================
def fun_ridge_fs(x, *data_args):
    """Wrapper para Ridge Regression.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº de componentes
        x[3] → kernel
        x[4] → alpha (penalização L2)
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'Ridge'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 5

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = Ridge(random_state=int(random_seed))
    p = {
        'alpha': x[4],
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# RANDOM FOREST REGRESSOR
# =============================================================================
def fun_rf_fs(x, *data_args):
    """Wrapper para Random Forest Regressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº de componentes
        x[3] → kernel
        x[4] → n_estimators
        x[5] → max_features (proporção)
        x[6] → max_depth
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'RF'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 7

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = RandomForestRegressor(random_state=int(random_seed), n_jobs=-1)
    p = {
        'n_estimators': int(x[4] * 100 + 10),
        'max_features': max(min(x[5], 1.0), 0.1),
        'max_depth': int(x[6] * 20 + 2)
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# SUPORT VECTOR REGRESSOR
# =============================================================================
def fun_svr_fs(x, *data_args):
    """Wrapper para SVR.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº de componentes
        x[3] → kernel
        x[4] → C
        x[5] → epsilon
        x[6] → gamma
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'SVR'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 7

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = SVR()
    kernel_map = {0: 'linear', 1: 'poly', 2: 'rbf', 3: 'sigmoid'}
    p = {
        'C': x[4] * 100 + 0.1,
        'epsilon': x[5],
        'gamma': x[6],
        'kernel': kernel_map.get(kernel_type, 'rbf')
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# MLP REGRESSOR
# =============================================================================
def fun_mlp_fs(x, *data_args):
    """Wrapper para MLPRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº de componentes
        x[3] → kernel
        x[4] → hidden_layer_sizes (escala → nº neurônios)
        x[5] → alpha (regularização)
        x[6] → learning_rate_init
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'MLP'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 7

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = MLPRegressor(random_state=int(random_seed), max_iter=500)
    p = {
        'hidden_layer_sizes': (int(x[4] * 100) + 5,),
        'alpha': x[5],
        'learning_rate_init': x[6] * 0.1
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# XGBOOST REGRESSOR
# =============================================================================
def fun_xgb_fs(x, *data_args):
    """Wrapper para XGBRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº de componentes
        x[3] → kernel
        x[4] → n_estimators
        x[5] → max_depth
        x[6] → learning_rate
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'XGB'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 7

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = XGBRegressor(random_state=int(random_seed), n_jobs=-1)
    p = {
        'n_estimators': int(x[4] * 200 + 50),
        'max_depth': int(x[5] * 10 + 2),
        'learning_rate': x[6] * 0.3 + 0.01
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)



# =============================================================================
# KNN REGRESSOR
# =============================================================================
def fun_knn_fs(x, *data_args):
    """Wrapper para KNeighborsRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → n_neighbors
        x[5] → weights (0: uniform, 1: distance)
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'KNN'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    weights_map = {0: 'uniform', 1: 'distance'}
    clf = KNeighborsRegressor()
    p = {
        'n_neighbors': int(x[4] * 20 + 1),
        'weights': weights_map.get(int(x[5] + 0.5), 'uniform')
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# DECISION TREE REGRESSOR
# =============================================================================
def fun_dt_fs(x, *data_args):
    """Wrapper para DecisionTreeRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → max_depth
        x[5] → min_samples_split
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'DT'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = DecisionTreeRegressor(random_state=int(random_seed))
    p = {
        'max_depth': int(x[4] * 20 + 1),
        'min_samples_split': int(x[5] * 10 + 2)
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# KERNEL RIDGE REGRESSION
# =============================================================================
def fun_kr_fs(x, *data_args):
    """Wrapper para Kernel Ridge.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → alpha
        x[5] → gamma
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'KR'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = KernelRidge()
    p = {
        'alpha': x[4],
        'gamma': x[5],
        'kernel': 'rbf'
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# BAYESIAN RIDGE
# =============================================================================
def fun_br_fs(x, *data_args):
    """Wrapper para BayesianRidge.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → alpha_1
        x[5] → alpha_2
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'BR'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = BayesianRidge()
    p = {
        'alpha_1': x[4] * 1e-6,
        'alpha_2': x[5] * 1e-6
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# NAIVE BAYES (Gaussian)
# =============================================================================
def fun_nb_fs(x, *data_args):
    """Wrapper para GaussianNB.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → var_smoothing
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'NB'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 5

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = GaussianNB()
    p = {
        'var_smoothing': x[4] * 1e-9 + 1e-9
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# PASSIVE AGGRESSIVE REGRESSOR
# =============================================================================
def fun_pa_fs(x, *data_args):
    """Wrapper para PassiveAggressiveRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → C (regularização)
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'PA'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 5

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = PassiveAggressiveRegressor(random_state=int(random_seed), max_iter=500)
    p = {
        'C': x[4]
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# GAUSSIAN PROCESS REGRESSOR
# =============================================================================
def fun_gp_fs(x, *data_args):
    """Wrapper para GaussianProcessRegressor.

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → alpha
        x[5] → comprimento da RBF
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'GPR'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    kernel = RBF(length_scale=x[5])
    clf = GaussianProcessRegressor(random_state=int(random_seed), kernel=kernel)
    p = {
        'alpha': x[4],
        'kernel': kernel
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# LEAST SQUARES SUPPORT VECTOR REGRESSION (LSSVR)
# =============================================================================
def fun_lssvr_fs(x, *data_args):
    """Wrapper para LSSVR (implementação custom em util.LSSVR).

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → gamma
        x[5] → C
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'LSSVR'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 6

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = LSSVR()
    p = {
        'gamma': x[4],
        'C': x[5]
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# EXTREME LEARNING MACHINE (ELM)
# =============================================================================
def fun_elm_fs(x, *data_args):
    """Wrapper para ELMRegressor (implementação em util.ELM).

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → hidden_layer_size
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'ELM'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 5

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = ELMRegressor()
    p = {
        'hidden_layer_size': int(x[4] * 100 + 10)
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)


# =============================================================================
# RADIAL BASIS FUNCTION NEURAL NETWORK (RBFNN)
# =============================================================================
def fun_rbfnn_fs(x, *data_args):
    """Wrapper para RBFNNRegressor (implementação em util.RBFNN).

    Hiperparâmetros:
        x[0] → normalizador
        x[1] → transformador
        x[2] → nº componentes
        x[3] → kernel
        x[4] → nº de centros
    """
    (X_train, y_train, X_test, y_test, flag, task, n_splits,
     random_seed, scoring, target, n_samples_train,
     n_samples_test, n_features) = data_args

    clf_name = 'RBFNN'
    normalizer_type = int(x[0] + 0.995)
    transformer_type = int(x[1] + 0.995)
    n_components = int(x[2] * n_features + 1)
    kernel_type = int(x[3] + 0.995)
    n_decision_variables = 5

    normalizer_args = (normalizer_type,)
    transformer_args = (transformer_type, n_components, kernel_type)

    clf = RBFNNRegressor()
    p = {
        'n_centers': int(x[4] * 50 + 5)
    }
    clf.set_params(**p)

    estimator_args = (clf_name, n_decision_variables, p, clf)
    return model_base_evaluation(x, data_args, estimator_args, normalizer_args, transformer_args)

