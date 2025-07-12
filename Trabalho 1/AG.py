import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

# === Função para leitura do arquivo .evrp ===
def ler_arquivo_evrp(caminho_arquivo):
    """
    Lê um arquivo EVRP com seções de coordenadas e depósito
    Retorna:
    - coordenadas: dicionário {id: (x, y)}
    - deposito: ID do depósito
    """
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()

    coordenadas = {}
    deposito = None
    em_secao_coord = False

    for linha in linhas:
        if "NODE_COORD_SECTION" in linha:
            em_secao_coord = True
            continue
        if "DEMAND_SECTION" in linha or "DEPOT_SECTION" in linha:
            em_secao_coord = False
        if em_secao_coord:
            partes = linha.strip().split()
            if len(partes) == 3:
                id_no, x, y = map(float, partes)
                coordenadas[int(id_no)] = (x, y)

        if "DEPOT_SECTION" in linha:
            idx = linhas.index(linha)
            deposito = int(linhas[idx+1].strip())

    return coordenadas, deposito

# Calcula a distância euclidiana entre dois pontos
def distancia(ponto1, ponto2):
    return np.linalg.norm(np.array(ponto1) - np.array(ponto2))

# Avalia a distância total de uma rota completa
def distancia_total(rota, coordenadas, deposito):
    dist = distancia(coordenadas[deposito], coordenadas[rota[0]])
    for i in range(len(rota) - 1):
        dist += distancia(coordenadas[rota[i]], coordenadas[rota[i+1]])
    dist += distancia(coordenadas[rota[-1]], coordenadas[deposito])
    return dist

# Decodifica o cromossomo binário contínuo por blocos
def decodificar_cromossomo(cromossomo, clientes, parametros):
    n = len(clientes)
    k = parametros['BITS_POR_PRIORIDADE']
    prioridades = [int(''.join(str(bit) for bit in cromossomo[i*k:(i+1)*k]), 2) for i in range(n)]
    return [x for _, x in sorted(zip(prioridades, clientes), reverse=True)]

# Gera população inicial binária com 0s e 1s
def inicializar_populacao(n_clientes, parametros):
    k = parametros['BITS_POR_PRIORIDADE']
    tamanho = n_clientes * k
    return [[random.randint(0, 1) for _ in range(tamanho)] for _ in range(parametros['TAMANHO_POPULACAO'])]

# Crossover de ponto único
def cruzamento(pai1, pai2):
    ponto = random.randint(1, len(pai1) - 1)
    return pai1[:ponto] + pai2[ponto:], pai2[:ponto] + pai1[ponto:]

# Muta cromossomo com flip de bits
def mutacao(cromossomo, parametros):
    taxa = parametros['TAXA_MUTACAO']
    return [1 - bit if random.random() < taxa else bit for bit in cromossomo]

# Seleção por torneio
def selecao_torneio(pop, aptidoes, k):
    return min(random.sample(list(zip(pop, aptidoes)), k), key=lambda x: x[1])[0]

# Algoritmo Genético
def evoluir(coordenadas, deposito, clientes, parametros):
    n = len(clientes)
    populacao = inicializar_populacao(n, parametros)
    max_aval = 25000 * len(coordenadas)
    avaliacoes = 0
    melhor_dist = float('inf')
    melhor_rota = None

    while avaliacoes < max_aval:
        rotas = [decodificar_cromossomo(c, clientes, parametros) for c in populacao]
        aptidoes = [distancia_total(r, coordenadas, deposito) for r in rotas]
        avaliacoes += len(populacao)

        nova_pop = []
        elite_idx = np.argmin(aptidoes)
        nova_pop.append(populacao[elite_idx])

        while len(nova_pop) < parametros['TAMANHO_POPULACAO']:
            pai1 = selecao_torneio(populacao, aptidoes, parametros['TORNEIO_K'])
            pai2 = selecao_torneio(populacao, aptidoes, parametros['TORNEIO_K'])
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_pop.extend([mutacao(filho1, parametros), mutacao(filho2, parametros)])

        populacao = nova_pop[:parametros['TAMANHO_POPULACAO']]

        if min(aptidoes) < melhor_dist:
            melhor_dist = min(aptidoes)
            melhor_rota = rotas[np.argmin(aptidoes)]

    return melhor_dist, melhor_rota

# Função para plotar e salvar a rota encontrada
def plotar_rota(rota, coordenadas, deposito, caminho_salvar):
    plt.figure(figsize=(8, 6))

    # Caminho verde da rota (desenhar por baixo)
    xs = [coordenadas[i][0] for i in rota]
    ys = [coordenadas[i][1] for i in rota]
    plt.plot(xs, ys, color='green', linewidth=2, zorder=1)  # linha de rota

    # Clientes (círculos azuis por cima)
    clientes = [i for i in rota if i != deposito]
    plt.scatter([coordenadas[i][0] for i in clientes],
                [coordenadas[i][1] for i in clientes],
                c='blue', label='Clientes', zorder=2)

    # Adiciona o número do cliente próximo ao ponto azul
    for i in clientes:
        plt.text(coordenadas[i][0], coordenadas[i][1], str(i),
                 color='black', fontsize=8, fontweight='bold',
                 verticalalignment='bottom', horizontalalignment='right', zorder=4)

    # Depósito (círculo vermelho ainda mais acima)
    plt.scatter(coordenadas[deposito][0], coordenadas[deposito][1],
                c='red', s=50, label='Depósito', zorder=3)

    # Mostrar número do depósito em vermelho perto do ponto
    plt.text(coordenadas[deposito][0], coordenadas[deposito][1], str(deposito),
             color='black', fontsize=8, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='left', zorder=4)

    plt.title("Rota do Veículo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(caminho_salvar), exist_ok=True)
    plt.savefig(caminho_salvar)
    plt.close()

