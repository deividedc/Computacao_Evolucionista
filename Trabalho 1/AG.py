import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

# === Função para leitura do arquivo .evrp ===
def ler_arquivo_evrp(caminho_arquivo):
    """
    Lê um arquivo EVRP e extrai as coordenadas dos nós e o ID do depósito.
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

# === Função de distância euclidiana ===
def distancia(ponto1, ponto2):
    return np.linalg.norm(np.array(ponto1) - np.array(ponto2))

# === Avaliação da distância total de uma rota ===
def distancia_total(rota, coordenadas, deposito):
    dist = distancia(coordenadas[deposito], coordenadas[rota[0]])
    for i in range(len(rota) - 1):
        dist += distancia(coordenadas[rota[i]], coordenadas[rota[i+1]])
    dist += distancia(coordenadas[rota[-1]], coordenadas[deposito])
    return dist

# === Conversão para código Gray ===
def inteiro_para_gray(n):
    return n ^ (n >> 1)

def gray_para_inteiro(g):
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n

# === Decodifica cromossomo binário em prioridades via código Gray ===
def decodificar_cromossomo(cromossomo, clientes, parametros):
    n = len(clientes)
    k = parametros['BITS_POR_PRIORIDADE']
    prioridades = []
    for i in range(n):
        bits = cromossomo[i*k:(i+1)*k]
        valor_bin = int(''.join(str(b) for b in bits), 2)
        valor_gray = gray_para_inteiro(valor_bin)
        prioridades.append(valor_gray)
    return [x for _, x in sorted(zip(prioridades, clientes), reverse=True)]

# === Inicializa a população com valores em Gray ===
def inicializar_populacao(n_clientes, parametros):
    k = parametros['BITS_POR_PRIORIDADE']
    max_valor = 2**k - 1
    populacao = []
    for _ in range(parametros['TAMANHO_POPULACAO']):
        individuo = []
        for _ in range(n_clientes):
            valor = random.randint(0, max_valor)
            valor_gray = inteiro_para_gray(valor)
            bits = list(map(int, bin(valor_gray)[2:].zfill(k)))
            individuo.extend(bits)
        populacao.append(individuo)
    return populacao

# === Operadores genéticos ===
def cruzamento(pai1, pai2):
    """Crossover de ponto único."""
    ponto = random.randint(1, len(pai1) - 1)
    return pai1[:ponto] + pai2[ponto:], pai2[:ponto] + pai1[ponto:]

def mutacao(cromossomo, parametros):
    """Mutação por flip de bits com uma taxa definida."""
    taxa = parametros['TAXA_MUTACAO']
    return [1 - bit if random.random() < taxa else bit for bit in cromossomo]

def selecao_torneio(pop, aptidoes, k):
    """Seleção por torneio de k indivíduos."""
    return min(random.sample(list(zip(pop, aptidoes)), k), key=lambda x: x[1])[0]

# === Função principal de evolução com critério de parada por estagnação ===
def evoluir(coordenadas, deposito, clientes, parametros):
    """
    Executa o algoritmo genético para encontrar uma rota otimizada para o EVRP.
    Critérios de parada:
    - Número máximo de avaliações.
    - Número máximo de gerações sem melhora.
    """
    n = len(clientes)
    populacao = inicializar_populacao(n, parametros)
    max_aval = 25000 * len(coordenadas)
    avaliacoes = 0
    melhor_dist = float('inf')
    melhor_rota = None
    num_elitistas = parametros.get('NUM_ELITISTAS', 1)

    # Novo: controle de estagnação
    sem_melhora = 0
    MAX_SEM_MELHORA = parametros.get('MAX_SEM_MELHORA', 200)

    while avaliacoes < max_aval:
        rotas = [decodificar_cromossomo(c, clientes, parametros) for c in populacao]
        aptidoes = [distancia_total(r, coordenadas, deposito) for r in rotas]
        avaliacoes += len(populacao)

        # Ordena população por aptidão crescente (melhor = menor distância)
        pop_ordenada = [ind for _, ind in sorted(zip(aptidoes, populacao), key=lambda x: x[0])]
        rotas_ordenadas = [r for _, r in sorted(zip(aptidoes, rotas), key=lambda x: x[0])]
        aptidoes_ordenadas = sorted(aptidoes)

        nova_pop = []
        nova_pop.extend(pop_ordenada[:num_elitistas])  # elitismo

        # Geração da nova população
        while len(nova_pop) < parametros['TAMANHO_POPULACAO']:
            pai1 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])
            pai2 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])
            filho1, filho2 = cruzamento(pai1, pai2)
            filho1 = mutacao(filho1, parametros)
            filho2 = mutacao(filho2, parametros)
            if len(nova_pop) + 2 <= parametros['TAMANHO_POPULACAO']:
                nova_pop.extend([filho1, filho2])
            else:
                nova_pop.append(filho1)

        populacao = nova_pop

        # Atualiza a melhor solução e checa estagnação
        if aptidoes_ordenadas[0] < melhor_dist:
            melhor_dist = aptidoes_ordenadas[0]
            melhor_rota = rotas_ordenadas[0]
            sem_melhora = 0
        else:
            sem_melhora += 1

        # Parar se não houver melhora por muitas gerações
        if sem_melhora >= MAX_SEM_MELHORA:
            break

    return melhor_dist, melhor_rota

# === Função para plotar e salvar a rota encontrada ===
def plotar_rota(rota, coordenadas, deposito, caminho_salvar):
    """
    Gera um gráfico da rota final do veículo e salva em imagem.
    """
    plt.figure(figsize=(8, 6))

    # Trajeto da rota
    xs = [coordenadas[i][0] for i in rota]
    ys = [coordenadas[i][1] for i in rota]
    plt.plot(xs, ys, color='green', linewidth=2, zorder=1)

    # Pontos dos clientes
    clientes = [i for i in rota if i != deposito]
    plt.scatter([coordenadas[i][0] for i in clientes],
                [coordenadas[i][1] for i in clientes],
                c='blue', label='Clientes', zorder=2)

    for i in clientes:
        plt.text(coordenadas[i][0], coordenadas[i][1], str(i),
                 color='black', fontsize=8, fontweight='bold',
                 verticalalignment='bottom', horizontalalignment='right', zorder=4)

    # Depósito
    plt.scatter(coordenadas[deposito][0], coordenadas[deposito][1],
                c='red', s=50, label='Depósito', zorder=3)
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
