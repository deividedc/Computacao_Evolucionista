import numpy as np
import random
import time
import os
import json

from AG import ler_arquivo_evrp, evoluir, plotar_rota

# === Número de execuções independentes do Algoritmo Genético ===
NUM_EXECUCOES = 20

# === Parâmetros do Algoritmo Genético ===
parametros = {
    'BITS_POR_PRIORIDADE': 16,       # Bits usados para codificar prioridade de clientes
    'TAMANHO_POPULACAO':   500,      # Tamanho da população
    'TAXA_MUTACAO':        0.02,     # Probabilidade de mutação por bit
    'SEMENTE_BASE':        42,       # Semente base para geração de números aleatórios
    'TORNEIO_K':           5,       # Tamanho do torneio para seleção
    'NUM_ELITISTAS':       2,        # Número de indivíduos da elite (0 para desativar)
    'MAX_SEM_MELHORA':     2000      # Número máximo de gerações sem melhora
}

# === Caminho da pasta onde estão os arquivos .evrp ===
caminho = './evrp-benchmark-set'

# === Lista com os nomes dos arquivos .evrp a serem processados ===
arquivos = [
    "E-n23-k3.evrp",
    "E-n51-k5.evrp"
]

# === Loop principal para processar cada instância do EVRP ===
for arquivo_nome in arquivos:
    # Monta o caminho completo do arquivo
    caminho_arquivo = os.path.join(caminho, arquivo_nome)
    print(f"\nProcessando arquivo: {caminho_arquivo}")

    # Lê os dados da instância (coordenadas e depósito)
    coordenadas, deposito = ler_arquivo_evrp(caminho_arquivo)

    # Lista de clientes (exclui o depósito)
    clientes = [i for i in coordenadas if i != deposito]

    # Listas para armazenar resultados de cada execução
    resultados = []
    rotas = []
    tempos = []

    # Define o nome base do arquivo (sem extensão)
    nome_base = os.path.splitext(arquivo_nome)[0]

    # Define o nome da subpasta com base no uso de elitismo
    if parametros['NUM_ELITISTAS'] == 0:
        nome_subpasta = "Resultado_Sem_Elitismo"
    else:
        nome_subpasta = f"Resultado_Com_Elitismo_{parametros['NUM_ELITISTAS']}"

    # Caminhos das pastas de resultados e de plots
    pasta_base = os.path.join(".", nome_base, nome_subpasta)
    pasta_plots = os.path.join(pasta_base, "plots")

    # Cria as pastas se não existirem
    os.makedirs(pasta_plots, exist_ok=True)

    # === Executa o Algoritmo Genético NUM_EXECUCOES vezes ===
    for execucao in range(NUM_EXECUCOES):
        # Define sementes para reprodutibilidade
        random.seed(parametros['SEMENTE_BASE'] + execucao)
        np.random.seed(parametros['SEMENTE_BASE'] + execucao)

        # Marca o tempo de início da execução
        start_time = time.time()

        # Executa o algoritmo e obtém a melhor distância e rota
        distancia, rota = evoluir(coordenadas, deposito, clientes, parametros)

        # Marca o tempo de fim da execução
        end_time = time.time()
        duracao = end_time - start_time
        tempos.append(duracao)

        # Rota completa inclui ida e volta ao depósito
        rota_completa = [deposito] + rota + [deposito]

        # Exibe os resultados da execução no console
        print(f"\nExecução {execucao + 1} - {nome_base}:")
        print(f"  ▸ Distância = {distancia:.2f}")
        print(f"  ▸ Rota      = {rota_completa}")
        print(f"  ▸ Tempo     = {duracao:.2f} segundos")

        # Armazena os resultados para estatísticas posteriores
        resultados.append(distancia)
        rotas.append(rota_completa)

        # Salva o gráfico da rota na pasta de plots
        nome_arquivo_fig = os.path.join(pasta_plots, f"rota_execucao_{execucao+1}.png")
        plotar_rota(rota_completa, coordenadas, deposito, nome_arquivo_fig)

    # === Calcula estatísticas gerais para as execuções ===
    estatisticas = {
        "arquivo": nome_base,
        "media_distancia": float(np.mean(resultados)),
        "desvio_padrao_distancia": float(np.std(resultados)),
        "minimo_distancia": float(np.min(resultados)),
        "maximo_distancia": float(np.max(resultados)),
        "media_tempo": float(np.mean(tempos)),
        "desvio_padrao_tempo": float(np.std(tempos)),
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

    # Exibe estatísticas finais no console
    print("\n=== Estatísticas Finais para", nome_base, "===")
    print(f"Média Distância: {estatisticas['media_distancia']:.2f}")
    print(f"Desvio padrão Distância: {estatisticas['desvio_padrao_distancia']:.2f}")
    print(f"Mínimo Distância: {estatisticas['minimo_distancia']:.2f}")
    print(f"Máximo Distância: {estatisticas['maximo_distancia']:.2f}")
    print(f"Tempo médio de execução: {estatisticas['media_tempo']:.2f} segundos")
    print(f"Desvio padrão do tempo: {estatisticas['desvio_padrao_tempo']:.2f} segundos")

    # Salva os resultados em um arquivo JSON dentro da subpasta correspondente
    nome_arquivo_json = os.path.join(pasta_base, "resultados.json")
    with open(nome_arquivo_json, "w") as f:
        json.dump(estatisticas, f, indent=4)
