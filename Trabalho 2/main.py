import numpy as np
import random
import time
import os
import json

# Importa funções do módulo AG.py
from AG import ler_arquivo_evrp, evoluir, plotar_rota

# === Número de execuções independentes do Algoritmo Genético para cada instância ===
NUM_EXECUCOES = 20  

# === Parâmetros do Algoritmo Genético ===
parametros = {
    'TAMANHO_POPULACAO':   150,      # Tamanho da população para maior diversidade genética
    'TAXA_MUTACAO':        0.02,     # Probabilidade de mutação 
    'SEMENTE_BASE':        42,       # Semente base para garantir reprodutibilidade das execuções
    'TORNEIO_K':           7,        # Número de indivíduos no torneio para seleção (maior = mais seletivo)
    'NUM_ELITISTAS':       4,        # Quantidade de melhores indivíduos mantidos (elitismo)
    'MAX_SEM_MELHORA':     100000,   # Número máximo de gerações sem melhoria antes de resetar população
    'SELECAO':             'torneio' # Método de seleção utilizado ('torneio' ou 'selecao_roleta')
}

# === Caminho da pasta contendo os arquivos .evrp ===
caminho = './evrp-benchmark-set'

# === Lista com os nomes dos arquivos .evrp que serão processados ===
arquivos = [
    "E-n23-k3.evrp",
    "E-n51-k5.evrp"
]

# === Loop para processar cada arquivo da lista ===
for arquivo_nome in arquivos:
    caminho_arquivo = os.path.join(caminho, arquivo_nome)
    print(f"\nProcessando arquivo: {caminho_arquivo}")

    # Leitura dos dados do arquivo .evrp (coordenadas dos nós e depósito)
    coordenadas, deposito = ler_arquivo_evrp(caminho_arquivo)

    # Listas para armazenar resultados de todas as execuções
    resultados = []  # Distâncias das melhores rotas
    rotas = []       # Rotas encontradas
    tempos = []      # Tempos gastos em cada execução

    # Nome base para criar pastas e arquivos de saída
    nome_base = os.path.splitext(arquivo_nome)[0]

    # Define o nome da subpasta baseado no número de elitistas
    nome_subpasta = f"Resultado_Com_Elitismo_{parametros['NUM_ELITISTAS']}"

    # Criação dos caminhos para salvar resultados e gráficos
    pasta_base = os.path.join(".", nome_base, nome_subpasta)
    pasta_plots = os.path.join(pasta_base, "plots")
    os.makedirs(pasta_plots, exist_ok=True)  # Cria as pastas caso não existam

    # Variáveis para guardar a melhor solução geral e a execução que a encontrou
    melhor_global_dist = float('inf')  # Inicializa com infinito para garantir substituição
    melhor_global_rota = None          # Guarda a melhor rota encontrada
    execucao_melhor_rota = None        # Guarda qual execução encontrou a melhor rota

    # === Loop de execuções independentes ===
    for execucao in range(NUM_EXECUCOES):
        # Define a semente do random para tornar a execução reprodutível
        random.seed(parametros['SEMENTE_BASE'] + execucao)
        np.random.seed(parametros['SEMENTE_BASE'] + execucao)

        # Marca o início do tempo para medir duração da execução
        start_time = time.time()

        # Executa o algoritmo genético e obtém melhor distância e rota
        distancia, rota = evoluir(coordenadas, deposito, parametros, verbose=False)

        # Marca o fim do tempo e calcula a duração
        end_time = time.time()
        duracao = end_time - start_time

        # Constrói a rota completa incluindo ida e volta ao depósito
        rota_completa = [deposito] + rota + [deposito]

        # Exibe os resultados da execução atual no console
        print(f"\nExecução {execucao + 1} - {nome_base}:")
        print(f"  ▸ Distância = {distancia:.2f}")
        print(f"  ▸ Rota      = {rota_completa}")
        print(f"  ▸ Tempo     = {duracao:.2f} segundos")

        # Armazena resultados para análise posterior
        resultados.append(distancia)
        rotas.append(rota_completa)
        tempos.append(duracao)

        # Salva o gráfico da rota desta execução normalmente
        nome_arquivo_fig_exec = os.path.join(pasta_plots, f"rota_execucao_{execucao+1}.png")
        plotar_rota(rota_completa, coordenadas, deposito, nome_arquivo_fig_exec)

        # Atualiza a melhor solução global se necessário
        if distancia < melhor_global_dist:
            melhor_global_dist = distancia
            melhor_global_rota = rota_completa
            execucao_melhor_rota = execucao + 1  # Guarda o número da execução com a melhor rota

    # Após todas as execuções, salva o gráfico da melhor rota global (uma única vez)
    nome_arquivo_fig_best = os.path.join(pasta_plots, "melhor_rota_global.png")
    plotar_rota(melhor_global_rota, coordenadas, deposito, nome_arquivo_fig_best)

    # === Compilação das estatísticas finais após todas as execuções ===
    estatisticas = {
        "arquivo": nome_base,
        "media_distancia": float(np.mean(resultados)),
        "desvio_padrao_distancia": float(np.std(resultados)),
        "minimo_distancia": float(np.min(resultados)),
        "maximo_distancia": float(np.max(resultados)),
        "media_tempo": float(np.mean(tempos)),
        "desvio_padrao_tempo": float(np.std(tempos)),
        "execucao_melhor_rota": execucao_melhor_rota,  # Execução com a melhor rota
        "resultados_execucoes": []
    }

    # Adiciona os resultados detalhados de cada execução
    for i in range(NUM_EXECUCOES):
        estatisticas["resultados_execucoes"].append({
            "execucao": i + 1,
            "distancia": float(resultados[i]),
            "rota": rotas[i],
            "tempo_execucao": float(tempos[i])
        })

    # Exibe no console as estatísticas consolidadas
    print("\n=== Estatísticas Finais para", nome_base, "===")
    print(f"Média Distância: {estatisticas['media_distancia']:.2f}")
    print(f"Desvio padrão Distância: {estatisticas['desvio_padrao_distancia']:.2f}")
    print(f"Mínimo Distância: {estatisticas['minimo_distancia']:.2f}")
    print(f"Máximo Distância: {estatisticas['maximo_distancia']:.2f}")
    print(f"Tempo médio de execução: {estatisticas['media_tempo']:.2f} segundos")
    print(f"Desvio padrão do tempo: {estatisticas['desvio_padrao_tempo']:.2f} segundos")

    # Salva as estatísticas em arquivo JSON para análise posterior
    nome_arquivo_json = os.path.join(pasta_base, "resultados.json")
    with open(nome_arquivo_json, "w") as f:
        json.dump(estatisticas, f, indent=4)
