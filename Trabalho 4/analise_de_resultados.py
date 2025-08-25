import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Função para calcular métricas
# -----------------------------
def calculate_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    R     = np.corrcoef(y_test, y_pred)[0, 1]  # Coeficiente de correlação
    R2    = r2_score(y_test, y_pred)          # R²
    RMSE  = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
    MAE   = mean_absolute_error(y_test, y_pred)          # MAE
    MAPE  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # MAPE
    
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
                folder_name = os.path.basename(root)
                
                # Extrair modelo (antes do primeiro '_')
                model = folder_name.split('_')[0]
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    run       = data.get('Run')
                    seed      = data.get('Seed')
                    y_test    = data.get('y_test')
                    y_pred    = data.get('y_pred')
                    optimizer = data.get('Est_name')
                    
                    # Calcular métricas
                    R, R2, RMSE, MAE, MAPE = calculate_metrics(y_test, y_pred)
                    
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

# --------------------------------------
# Salvar CSV
# --------------------------------------
output_csv_path = './analysis/metrics_summary.csv'
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df_grouped.to_csv(output_csv_path, index=False)
print(f"CSV salvo em: {output_csv_path}")

# --------------------------------------
# Radar Chart - Subplots por Otimizador
# --------------------------------------
metrics = ['R', 'R2', 'RMSE', 'MAE', 'MAPE']

# Normalizar métricas (0-1)
df_norm = df_grouped.copy()
for metric in metrics:
    df_norm[metric] = (df_grouped[metric] - df_grouped[metric].min()) / (df_grouped[metric].max() - df_grouped[metric].min())


# Inverter MAE e MAPE para 1-metrica (quanto maior melhor)
df_norm['MAE']  = 1 - df_norm['MAE']
df_norm['MAPE'] = 1 - df_norm['MAPE']
df_norm['RMSE'] = 1 - df_norm['RMSE']
Model = df_norm['Model'].unique()
num_opts = len(Model)

fig, axes = plt.subplots(1, num_opts, subplot_kw=dict(polar=True), figsize=(5*num_opts, 5))
if num_opts == 1:
    axes = [axes]

for ax, opt in zip(axes, Model):
    df_opt = df_norm[df_norm['Model'] == opt]
    for idx, row in df_opt.iterrows():
        values = row[metrics].values
        values = np.append(values, values[0])
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.append(angles, angles[0])
        ax.plot(angles, values, label=row['Optimizer'])
        # ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f'Optimizer: {opt}')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()

# Salvar figura
os.makedirs('./analysis/radarchar', exist_ok=True)
fig_path = './analysis/radarchar/radar_metrics.png'
plt.savefig(fig_path, dpi=300)
print(f"Radar chart salvo em: {fig_path}")
plt.show()


# ---------------------------------------------------------
# Função para listar modelos e otimizadores
# ---------------------------------------------------------
def listar_modelos_otimizadores(base_dir="result"):
    """
    Retorna uma lista de tuplas (modelo, otimizador) disponíveis na pasta 'results'.
    """
    if not os.path.exists(base_dir):
        print(f"A pasta '{base_dir}' não existe.")
        return []

    modelos_otimizadores = []
    for modelo in os.listdir(base_dir):
        modelo_path = os.path.join(base_dir, modelo)
        if os.path.isdir(modelo_path):
            for otimizador in os.listdir(modelo_path):
                otimizador_path = os.path.join(modelo_path, otimizador)
                if os.path.isdir(otimizador_path):
                    modelos_otimizadores.append((modelo, otimizador))
    return modelos_otimizadores

# ---------------------------------------------------------
# Função para calcular métricas
# ---------------------------------------------------------
def calculate_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    R    = np.corrcoef(y_test, y_pred)[0,1]
    R2   = r2_score(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE  = mean_absolute_error(y_test, y_pred)
    MAPE = np.mean(np.abs((y_test - y_pred)/y_test))
    
    return R, R2, RMSE, MAE, MAPE

# ---------------------------------------------------------
# Função para ler resultados de JSON
# ---------------------------------------------------------
def resultados(modelo, otimizador, base_dir="result"):
    """
    Lê todos os arquivos JSON de um modelo + otimizador e retorna listas de y_test e y_pred.
    """
    y_test_list = []
    y_pred_list = []

    folder_path = os.path.join(base_dir, modelo, otimizador)
    if not os.path.exists(folder_path):
        print(f"Pasta {folder_path} não existe.")
        return [], []

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            json_path = os.path.join(folder_path, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                y_test_list.append(np.array(data.get("y_test", [])))
                y_pred_list.append(np.array(data.get("y_pred", [])))
    return y_test_list, y_pred_list

# ---------------------------------------------------------
# Função para plotar gráfico de dispersão com barras de erro
# ---------------------------------------------------------
def plot_resultados(modelo, otimizador, base_dir="result", save_dir="analysis/scatter_plot"):
    y_test_list, y_pred_list = resultados(modelo, otimizador, base_dir)

    if not y_test_list or not y_pred_list:
        print(f"Nenhum dado encontrado para {modelo} - {otimizador}.")
        return

    # Métricas
    mae = []
    rmse = []
    r = []
    r2 = []
    mape = []
    
    for y_test, y_pred in zip(y_test_list, y_pred_list):
        mae.append(mean_absolute_error(y_test, y_pred))
        rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r.append(np.corrcoef(y_test, y_pred)[0,1])
        r2.append(r2_score(y_test, y_pred))
        mape.append(np.mean(np.abs((y_test - y_pred)/y_test)))

    # Médias e desvios
    r_mean = np.mean(r); r_std = np.std(r)
    r2_mean = np.mean(r2); r2_std = np.std(r2)
    rmse_mean = np.mean(rmse); rmse_std = np.std(rmse)
    mape_mean = np.mean(mape)*100; mape_std = np.std(mape)*100

    # Concatenar arrays
    y_test = np.concatenate(y_test_list)
    y_pred = np.concatenate(y_pred_list)

    # Ordenar
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    unique_y_test = np.unique(y_test_sorted)
    mean_y_pred = np.array([np.mean(y_pred_sorted[y_test_sorted==y]) for y in unique_y_test])
    std_y_pred  = np.array([np.std(y_pred_sorted[y_test_sorted==y]) for y in unique_y_test])

    # Criar pasta de plots
    os.makedirs(save_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(8,6))
    plt.errorbar(unique_y_test, mean_y_pred, yerr=std_y_pred, fmt='o', capsize=5, mfc='#0979b0', mec='black', label="Média ± Desvio Padrão")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#0979b0', label="Linha Ideal")
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Preditos")
    plt.grid(True)
    plt.legend()
    plt.text(0.95, 0.15, f'R = {r_mean:.2f} ± {r_std:.2f}\nR² = {r2_mean:.2f} ± {r2_std:.2f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.title(f"{modelo} - {otimizador}")
    plt.savefig(f"{save_dir}/{modelo} - {otimizador}.png", dpi=300)
    plt.show()

# ---------------------------------------------------------
# Exemplo de uso
# ---------------------------------------------------------

# Listar modelos e otimizadores disponíveis
modelos_otimizadores = listar_modelos_otimizadores("result")
print("Modelos e Otimizadores encontrados:")
for m, o in modelos_otimizadores:
    print(f"{m} - {o}")

# Plotar todos
for modelo, otimizador in modelos_otimizadores:
    plot_resultados(modelo, otimizador)