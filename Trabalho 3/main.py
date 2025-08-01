import numpy as np
import random
import time
import os
import json

# Importa funções do módulo AG.py
from AG import ler_arquivo_evrp, evoluir, plotar_rota, calcular_cargas_por_rota

# Número de execuções independentes do Algoritmo Genético para cada instância
NUM_EXECUCOES = 20  

# Lista com valores de elitismo para testar (pode ajustar)
num_elitistas = [0,2]

for elitistas in num_elitistas:

    # Define os parâmetros do algoritmo genético para esta configuração
    parametros = {
        'TAMANHO_POPULACAO': 150,
        'TAXA_MUTACAO': 0.02,
        'SEMENTE_BASE': 42,
        'TORNEIO_K': 7,
        'NUM_ELITISTAS': elitistas,
        'TIPO_CROSSOVER': 'OX',
    }

    caminho = './evrp-benchmark-set'
    arquivos = [
        "E-n23-k3.evrp",
        "E-n51-k5.evrp"
    ]

    for arquivo_nome in arquivos:
        caminho_arquivo = os.path.join(caminho, arquivo_nome)
        print(f"\nProcessando arquivo: {caminho_arquivo}")
    
        dados = ler_arquivo_evrp(caminho_arquivo)

        resultados = []  # guarda as distâncias de cada execução
        rotas = []       # guarda as rotas de cada execução
        tempos = []      # guarda os tempos de execução
        cargas = []      # guarda as cargas por rota de cada execução

        nome_base = os.path.splitext(arquivo_nome)[0]
        nome_subpasta = f"Resultado_Com_Elitismo_{parametros['NUM_ELITISTAS']}"
        pasta_base = os.path.join(".", nome_base, nome_subpasta)
        pasta_plots = os.path.join(pasta_base, "plots")
        os.makedirs(pasta_plots, exist_ok=True)

        melhor_global_dist = float('inf')
        melhor_global_rota = None
        execucao_melhor_rota = None

        for execucao in range(NUM_EXECUCOES):
            random.seed(parametros['SEMENTE_BASE'] + execucao)
            np.random.seed(parametros['SEMENTE_BASE'] + execucao)

            start_time = time.time()

            distancia, rota = evoluir(dados, parametros)

            end_time = time.time()
            duracao = end_time - start_time
            
            carga = calcular_cargas_por_rota(rota, dados)  # lista de cargas por rota nesta execução

            print(f"\nExecução {execucao + 1} - {nome_base}:")
            print(f"  ▸ Distância = {distancia:.2f}")
            print(f"  ▸ Rota      = {rota}")
            print(f"  ▸ Tempo     = {duracao:.2f} segundos")
            print(f"  ▸ Carga     = {carga}")

            resultados.append(distancia)
            rotas.append(rota)
            tempos.append(duracao)
            cargas.append(carga)  # adiciona lista de cargas desta execução
            

            nome_arquivo_fig_exec = os.path.join(pasta_plots, f"rota_execucao_{execucao+1}.png")
            plotar_rota(rota, dados, nome_arquivo_fig_exec)

            if distancia < melhor_global_dist:
                melhor_global_dist = distancia
                melhor_global_rota = rota
                execucao_melhor_rota = execucao + 1
        
        # Exibe as cargas para todas as execuções e rotas
        print(f"\nCapacidade do veículo: {dados['capacidade']}")
        for i, carga_exec in enumerate(cargas, start=1):
            for j, carga_rota in enumerate(carga_exec, start=1):
                print(f"Execução {i}, Rota {j} = carga transportada: {carga_rota}")

        estatisticas = {
            "arquivo": nome_base,
            "media_distancia": float(np.mean(resultados)),
            "desvio_padrao_distancia": float(np.std(resultados)),
            "minimo_distancia": float(np.min(resultados)),
            "maximo_distancia": float(np.max(resultados)),
            "media_tempo": float(np.mean(tempos)),
            "desvio_padrao_tempo": float(np.std(tempos)),
            "execucao_melhor_rota": execucao_melhor_rota,
            "capacidade_veiculo": dados['capacidade'],
            "resultados_execucoes": []
        }

        # Adiciona os resultados detalhados por execução no JSON
        for i in range(NUM_EXECUCOES):
            estatisticas["resultados_execucoes"].append({
                "execucao": i + 1,
                "distancia": float(resultados[i]),
                "rota": rotas[i],
                "tempo_execucao": float(tempos[i]),
                "carga_por_rota": cargas[i],  # já é lista de cargas
            })

        print("\n=== Estatísticas Finais para", nome_base, "===")
        print(f"Média Distância: {estatisticas['media_distancia']:.2f}")
        print(f"Desvio padrão Distância: {estatisticas['desvio_padrao_distancia']:.2f}")
        print(f"Mínimo Distância: {estatisticas['minimo_distancia']:.2f}")
        print(f"Máximo Distância: {estatisticas['maximo_distancia']:.2f}")
        print(f"Tempo médio de execução: {estatisticas['media_tempo']:.2f} segundos")
        print(f"Desvio padrão do tempo: {estatisticas['desvio_padrao_tempo']:.2f} segundos")

        # Salva resultados em arquivo JSON
        nome_arquivo_json = os.path.join(pasta_base, "resultados.json")
        with open(nome_arquivo_json, "w") as f:
            json.dump(estatisticas, f, indent=4)
