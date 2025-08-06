import numpy as np
import random
import time
import os
import json

# Importa funções auxiliares do módulo AG.py (como ler_arquivo_evrp, evoluir, plotar_rota, etc)
from AG import *

# Número de execuções independentes do algoritmo para cada instância
NUM_EXECUCOES = 20

# Lista com valores de elitismo a testar (número de melhores indivíduos que vão direto para próxima geração)
num_elitistas = [2]

# Lista com os tipos de crossover que serão testados
tipos_crossover = ['CX','OX']

# Loop para testar diferentes tipos de crossover
for crossover in tipos_crossover:
    for elitistas in num_elitistas:

        # Parâmetros do algoritmo genético
        parametros = {
            'TAMANHO_POPULACAO': 150,        # Quantidade de indivíduos na população
            'TAXA_MUTACAO': 0.02,             # Probabilidade de mutação
            'SEMENTE_BASE': 42,               # Semente base para reproducibilidade
            'TORNEIO_K': 7,                   # Número de competidores no torneio de seleção
            'NUM_ELITISTAS': elitistas,       # Número de elitistas a manter
            'TIPO_CROSSOVER': crossover,      # Tipo de crossover atual do loop
            'PERC_GULOSO': 0.4,                 # Percentual de indivíduos gerados pelo heurístico guloso (vizinho mais próximo)
            'PERC_ACO': 0.4,                    # Percentual de indivíduos gerados pelo ACO (ant colony optimization)
            # O restante será gerado aleatoriamente
        }

        # Caminho base onde os arquivos EVRP estão armazenados
        caminho = './evrp-benchmark-set'
        arquivos = [
            "E-n23-k3.evrp",
            "E-n51-k5.evrp"
        ]

        # Pasta base de resultados para o tipo de crossover atual
        pasta_crossover = os.path.join(".", f"Resultados_{parametros['TIPO_CROSSOVER']}")

        # Percentuais convertidos para inteiro para nome de pasta
        perc_guloso_pct = int(parametros['PERC_GULOSO'] * 100)
        perc_aco_pct = int(parametros['PERC_ACO'] * 100)

        # Pasta raiz que inclui os percentuais de heurístico guloso e ACO usados
        pasta_percentuais = os.path.join(pasta_crossover, f"Perc_Guloso_{perc_guloso_pct}_Perc_ACO_{perc_aco_pct}")
        os.makedirs(pasta_percentuais, exist_ok=True)  # Cria a pasta, se não existir

        # Processa cada arquivo EVRP
        for arquivo_nome in arquivos:
            caminho_arquivo = os.path.join(caminho, arquivo_nome)
            print(f"\nProcessando arquivo: {caminho_arquivo}")

            # Lê dados do arquivo EVRP para usar no algoritmo
            dados = ler_arquivo_evrp(caminho_arquivo)

            # Listas para armazenar resultados de cada execução
            resultados = []
            rotas = []
            tempos = []
            cargas = []

            # Extrai nome base do arquivo para nomear pastas e arquivos
            nome_base = os.path.splitext(arquivo_nome)[0]

            # Subpasta para resultados que inclui o número de elitistas
            nome_subpasta = f"Resultado_Com_Elitismo_{parametros['NUM_ELITISTAS']}"

            # Pasta base onde serão salvos resultados para este arquivo e configuração
            pasta_base = os.path.join(pasta_percentuais, nome_base, nome_subpasta)
            os.makedirs(pasta_base, exist_ok=True)

            # Pasta para salvar os plots das rotas
            pasta_plots = os.path.join(pasta_base, "plots")
            os.makedirs(pasta_plots, exist_ok=True)

            # Variáveis para guardar a melhor solução entre todas as execuções
            melhor_global_dist = float('inf')
            melhor_global_rota = None
            execucao_melhor_rota = None

            # Loop pelas execuções independentes para esta instância
            for execucao in range(NUM_EXECUCOES):
                # Configura a semente para garantir resultados reproduzíveis
                random.seed(parametros['SEMENTE_BASE'] + execucao)
                np.random.seed(parametros['SEMENTE_BASE'] + execucao)

                start_time = time.time()

                # Chama o algoritmo genético (função evoluir deve retornar distância e rota)
                distancia, rota = evoluir(dados, parametros)

                end_time = time.time()
                duracao = end_time - start_time

                # Calcula as cargas transportadas por rota para essa solução
                carga = calcular_cargas_por_rota(rota, dados)

                # Exibe informações da execução atual
                print(f"\nExecução {execucao + 1} - {nome_base} - Crossover {crossover}:")
                print(f"  ▸ Distância = {distancia:.2f}")
                print(f"  ▸ Rota      = {rota}")
                print(f"  ▸ Tempo     = {duracao:.2f} segundos")
                print(f"  ▸ Carga     = {carga}")

                # Armazena os resultados para estatísticas futuras
                resultados.append(distancia)
                rotas.append(rota)
                tempos.append(duracao)
                cargas.append(carga)

                # Salva o plot da rota dessa execução na pasta específica
                nome_arquivo_fig_exec = os.path.join(pasta_plots, f"rota_execucao_{execucao+1}.png")
                plotar_rota(rota, dados, nome_arquivo_fig_exec)

                # Atualiza a melhor solução global, se esta execução for melhor
                if distancia < melhor_global_dist:
                    melhor_global_dist = distancia
                    melhor_global_rota = rota
                    execucao_melhor_rota = execucao + 1

            # Após todas execuções, exibe as cargas transportadas por todas as rotas
            print(f"\nCapacidade do veículo: {dados['capacidade']}")
            for i, carga_exec in enumerate(cargas, start=1):
                for j, carga_rota in enumerate(carga_exec, start=1):
                    print(f"Execução {i}, Rota {j} = carga transportada: {carga_rota}")

            # Gera estatísticas resumidas para o conjunto de execuções
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

            # Adiciona os resultados detalhados de cada execução
            for i in range(NUM_EXECUCOES):
                estatisticas["resultados_execucoes"].append({
                    "execucao": i + 1,
                    "distancia": float(resultados[i]),
                    "rota": rotas[i],
                    "tempo_execucao": float(tempos[i]),
                    "carga_por_rota": cargas[i],
                })

            # Exibe estatísticas finais no console
            print("\n=== Estatísticas Finais para", nome_base, "===")
            print(f"Média Distância: {estatisticas['media_distancia']:.2f}")
            print(f"Desvio padrão Distância: {estatisticas['desvio_padrao_distancia']:.2f}")
            print(f"Mínimo Distância: {estatisticas['minimo_distancia']:.2f}")
            print(f"Máximo Distância: {estatisticas['maximo_distancia']:.2f}")
            print(f"Tempo médio de execução: {estatisticas['media_tempo']:.2f} segundos")
            print(f"Desvio padrão do tempo: {estatisticas['desvio_padrao_tempo']:.2f} segundos")

            # Salva as estatísticas e resultados detalhados em um arquivo JSON
            nome_arquivo_json = os.path.join(pasta_base, "resultados.json")
            with open(nome_arquivo_json, "w") as f:
                json.dump(estatisticas, f, indent=4)
