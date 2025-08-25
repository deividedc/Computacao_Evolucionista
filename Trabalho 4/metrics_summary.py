import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Função para calcular métricas
# -----------------------------
def calculate_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    R     = np.corrcoef(y_test, y_pred)[0, 1]
    R2    = r2_score(y_test, y_pred)
    RMSE  = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE   = mean_absolute_error(y_test, y_pred)
    MAPE  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return R, R2, RMSE, MAE, MAPE

# --------------------------------------
# Função para percorrer todas as subpastas
# --------------------------------------
def process_results(base_dir):
    columns = ['Model', 'Optimizer', 'Run', 'Seed', 'R', 'R2', 'RMSE', 'MAE', 'MAPE']
    df_results = pd.DataFrame(columns=columns)

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                
                # Extrair Model e Optimizer do caminho
                path_parts = os.path.normpath(json_path).split(os.sep)
                try:
                    idx = path_parts.index('result')
                    model = path_parts[idx + 1]
                    optimizer = path_parts[idx + 2]
                except (ValueError, IndexError):
                    model = 'Unknown'
                    optimizer = 'Unknown'
                
                # Abrir JSON
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    run    = data.get('Run', None)
                    seed   = data.get('Seed', None)
                    y_test = data.get('y_test', [])
                    y_pred = data.get('y_pred', [])
                    
                    # Calcular métricas
                    R, R2, RMSE, MAE, MAPE = calculate_metrics(y_test, y_pred)
                    
                    # Adicionar ao DataFrame
                    df_results = pd.concat([df_results, pd.DataFrame([{
                        'Model'    : model,
                        'Optimizer': optimizer,
                        'Run'      : run,
                        'Seed'     : seed,
                        'R'        : R,
                        'R2'       : R2,
                        'RMSE'     : RMSE,
                        'MAE'      : MAE,
                        'MAPE'     : MAPE
                    }])], ignore_index=True)
    
    return df_results

# --------------------------------------
# Caminho da pasta 'result'
# --------------------------------------
base_dir = './result'

# Processar resultados
df_metrics = process_results(base_dir)

# Agrupar por modelo e otimizador
df_grouped = df_metrics.groupby(['Model', 'Optimizer']).agg(
    {'R'   : ['mean', 'std'],
     'R2'  : ['mean', 'std'],
     'RMSE': ['mean', 'std'],
     'MAE' : ['mean', 'std'],
     'MAPE': ['mean', 'std']}).reset_index()

# Renomear colunas
df_grouped.columns = [
    'Model', 'Optimizer',
    'R', 'R std', 
    'R2', 'R2 std', 
    'RMSE', 'RMSE std',
    'MAE', 'MAE std',
    'MAPE', 'MAPE std'
]

# -----------------------------
# Salvar arquivo LaTeX (.tex)
# -----------------------------
output_tex_path = './analysis/metrics_summary.tex'
os.makedirs(os.path.dirname(output_tex_path), exist_ok=True)

# Salvar DataFrame como tabela LaTeX
df_grouped.to_latex(output_tex_path, index=False, float_format="%.4f", caption="Resumo das métricas por modelo e otimizador", label="tab:metrics_summary")

print(f'Tabela LaTeX salva em: {output_tex_path}')
