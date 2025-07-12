# main.py

import numpy as np
import random
import time
import os
import json

from AG import ler_arquivo_evrp, evoluir, plotar_rota

NUM_EXECUCOES = 20           # Quantas vezes o algoritmo será executado

# === PARÂMETROS DO ALGORITMO GENÉTICO ===
parametros = {
    'BITS_POR_PRIORIDADE': 5,     # Bits usados para codificar a prioridade de cada cliente
    'TAMANHO_POPULACAO': 100,     # Número de cromossomos na população
    'TAXA_MUTACAO': 0.01,         # Probabilidade de mutação por bit
    'SEMENTE_BASE': 36,           # Semente para reprodutibilidade
    'TORNEIO_K': 10               # Tamanho do torneio usado na seleção
}

# === LEITURA DA INSTÂNCIA DO PROBLEMA ===
caminho_arquivo = "E-n23-k3.evrp"  # Caminho do arquivo .evrp
coordenadas, deposito = ler_arquivo_evrp(caminho_arquivo)  # Lê coordenadas e depósito
clientes = [i for i in coordenadas if i != deposito]        # Lista de clientes (exclui o depósito)

resultados = []  # Armazena as distâncias das melhores soluções
rotas = []       # Armazena as rotas completas
tempos = []      # Armazena os tempos de execução

# Cria a pasta para salvar os plots
nome_pasta = f"plots_{os.path.splitext(caminho_arquivo)[0]}"
os.makedirs(nome_pasta, exist_ok=True)

# === LOOP PRINCIPAL DE EXECUÇÃO DO AG ===
for execucao in range(NUM_EXECUCOES):
    random.seed(parametros['SEMENTE_BASE'] + execucao)
    np.random.seed(parametros['SEMENTE_BASE'] + execucao)

    start_time = time.time()
    distancia, rota = evoluir(coordenadas, deposito, clientes, parametros)
    end_time = time.time()
    duracao = end_time - start_time
    tempos.append(duracao)

    rota_completa = [deposito] + rota + [deposito]

    print(f"\nExecução {execucao + 1}:")
    print(f"  ▸ Distância = {distancia:.2f}")
    print(f"  ▸ Rota      = {rota_completa}")
    print(f"  ▸ Tempo     = {duracao:.2f} segundos")

    resultados.append(distancia)
    rotas.append(rota_completa)

    nome_arquivo_fig = os.path.join(nome_pasta, f"rota_execucao_{execucao+1}.png")
    plotar_rota(rota_completa, coordenadas, deposito, nome_arquivo_fig)

# Estatísticas finais
estatisticas = {
    "media_distancia": float(np.mean(resultados)),
    "desvio_padrao_distancia": float(np.std(resultados)),
    "minimo_distancia": float(np.min(resultados)),
    "maximo_distancia": float(np.max(resultados)),
    "media_tempo": float(np.mean(tempos)),
    "desvio_padrao_tempo": float(np.std(tempos)),
    "resultados_execucoes": []
}

for i in range(NUM_EXECUCOES):
    estatisticas["resultados_execucoes"].append({
        "execucao": i + 1,
        "distancia": float(resultados[i]),
        "rota": rotas[i],
        "tempo_execucao": float(tempos[i])
    })

print("\n=== Estatísticas Finais ===")
print(f"Média Distância: {estatisticas['media_distancia']:.2f}")
print(f"Desvio padrão Distância: {estatisticas['desvio_padrao_distancia']:.2f}")
print(f"Mínimo Distância: {estatisticas['minimo_distancia']:.2f}")
print(f"Máximo Distância: {estatisticas['maximo_distancia']:.2f}")
print(f"\nTempo médio de execução: {estatisticas['media_tempo']:.2f} segundos")
print(f"Desvio padrão do tempo: {estatisticas['desvio_padrao_tempo']:.2f} segundos")

# Salvar em JSON
nome_arquivo_json = os.path.join(nome_pasta, "resultados.json")
with open(nome_arquivo_json, "w") as f:
    json.dump(estatisticas, f, indent=4)


