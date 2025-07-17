import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

# ============================
# Leitura de arquivo EVRP
# ============================
def ler_arquivo_evrp(caminho_arquivo):
    """
    Lê um arquivo .evrp no formato padrão e extrai:
    - Coordenadas dos nós (clientes e depósito)
    - ID do depósito
    
    Parâmetros:
        caminho_arquivo (str): Caminho para o arquivo .evrp
    
    Retorna:
        coordenadas (dict): Dicionário {id: (x, y)}
        deposito (int): ID do nó do depósito
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

# ============================
# Cálculo de distância Euclidiana
# ============================
def distancia(ponto1, ponto2):
    """
    Calcula a distância Euclidiana entre dois pontos 2D.
    """
    return np.linalg.norm(np.array(ponto1) - np.array(ponto2))

# ============================
# Avaliação da aptidão (distância total da rota)
# ============================
def distancia_total(rota, coordenadas, deposito):
    """
    Calcula a distância total da rota considerando ida e volta ao depósito.

    Parâmetros:
        rota (list): Lista de IDs dos clientes visitados
        coordenadas (dict): Dicionário com as posições dos nós
        deposito (int): ID do depósito

    Retorna:
        float: distância total da rota
    """
    dist = distancia(coordenadas[deposito], coordenadas[rota[0]])  # ida
    for i in range(len(rota) - 1):
        dist += distancia(coordenadas[rota[i]], coordenadas[rota[i+1]])
    dist += distancia(coordenadas[rota[-1]], coordenadas[deposito])  # volta
    return dist

# ============================
# Inicialização da população
# ============================
def inicializar_populacao(n_nos, parametros):
    """
    Inicializa a população com permutações aleatórias dos IDs dos clientes,
    excluindo o depósito (assumido como ID 1).

    Parâmetros:
        n_nos (int): Número total de nós (incluindo depósito)
        parametros (dict): Dicionário com parâmetros do AG

    Retorna:
        list: Lista de cromossomos (listas de permutação de clientes)
    """
    clientes_ids = list(range(2, n_nos + 1))  # Exclui o depósito que tem id=1
    populacao = []
    for _ in range(parametros['TAMANHO_POPULACAO']):
        individuo = clientes_ids[:]
        random.shuffle(individuo)
        populacao.append(individuo)
    return populacao

# ============================
# Crossover em ciclo (Cycle Crossover - CX)
# ============================
def cycle_crossover(p1, p2):
    """
    Aplica o crossover em ciclo (CX) entre dois pais.

    Parâmetros:
        p1 (list), p2 (list): Indivíduos pais (listas de permutação)

    Retorna:
        tuple: Dois filhos resultantes do cruzamento
    """
    size = len(p1)
    filho = [None] * size
    ciclos = 0
    i = 0

    while None in filho:
        if filho[i] is None:
            ciclos += 1
            idx = i
            while True:
                filho[idx] = p1[idx] if ciclos % 2 else p2[idx]
                idx = p1.index(p2[idx])
                if idx == i:
                    break
        i = filho.index(None) if None in filho else 0

    filho2 = [p2[i] if x == p1[i] else p1[i] for i, x in enumerate(filho)]
    return filho, filho2

# ============================
# Mutações
# ============================

def mutacao_swap(individuo):
    """
    Mutação simples por troca de duas posições aleatórias.
    """
    a, b = random.sample(range(len(individuo)), 2)
    individuo[a], individuo[b] = individuo[b], individuo[a]
    return individuo

def mutacao_inversao(individuo):
    """
    Mutação por inversão de um trecho aleatório da rota.
    Isso geralmente ajuda a melhorar a busca local.
    """
    a, b = sorted(random.sample(range(len(individuo)), 2))
    individuo[a:b+1] = individuo[a:b+1][::-1]
    return individuo

def mutacao(individuo, taxa_mutacao):
    """
    Aplica mutação baseada na taxa: troca simples ou inversão.
    Pode aplicar múltiplas mutações para maior diversidade.
    """
    if random.random() < taxa_mutacao:
        # Escolhe aleatoriamente o tipo de mutação
        if random.random() < 0.5:
            individuo = mutacao_swap(individuo)
        else:
            individuo = mutacao_inversao(individuo)
    return individuo



# ============================
# Seleção 
# ============================

def selecao_roleta(pop, aptidoes):
    """
    Seleção por roleta (probabilidade proporcional ao inverso da distância).
    """
    aptidoes_inv = [1.0 / apt for apt in aptidoes]
    soma = sum(aptidoes_inv)
    probs = [apt/soma for apt in aptidoes_inv]
    escolhido = np.random.choice(len(pop), p=probs)
    return pop[escolhido]

def selecao_torneio(pop, aptidoes, k):
    """
    Seleção por torneio: escolhe o melhor entre k indivíduos aleatórios.

    Parâmetros:
        pop (list): População atual
        aptidoes (list): Lista de distâncias (aptidão)
        k (int): Tamanho do torneio

    Retorna:
        list: Indivíduo selecionado
    """
    return min(random.sample(list(zip(pop, aptidoes)), k), key=lambda x: x[1])[0]


# =======================================================================
# Função principal de evolução com critério de parada por estagnação 
#========================================================================

def evoluir(coordenadas, deposito, parametros, verbose=False):
    """
    Algoritmo genético para EVRP com mutação melhorada e opção de seleção.
    Remove parada por estagnação para execução até o máximo de avaliações.
    
    Parâmetros:
        - coordenadas: dict {id: (x,y)}
        - deposito: int
        - parametros: dict com chaves:
            * 'TAMANHO_POPULACAO'
            * 'TORNEIO_K'
            * 'NUM_ELITISTAS'
            * 'MAX_SEM_MELHORA' (não usado como parada aqui)
            * 'TAXA_MUTACAO'
            * 'SELECAO' ('torneio' ou 'roleta')
        - verbose: bool para imprimir progresso
    
    Retorna:
        - melhor_distância, melhor_rota
    """
    n_nos = len(coordenadas)
    populacao = inicializar_populacao(n_nos, parametros)

    max_aval = 25000 * n_nos
    avaliacoes = 0
    melhor_dist = float('inf')
    melhor_rota = None

    num_elitistas = parametros.get('NUM_ELITISTAS', 1)
    if num_elitistas == 0:
        num_elitistas = 1

    taxa_mutacao = parametros.get('TAXA_MUTACAO', 0.02)
    selecao_tipo = parametros.get('SELECAO', 'torneio')

    while avaliacoes < max_aval:
        aptidoes = [distancia_total(c, coordenadas, deposito) for c in populacao]
        avaliacoes += len(populacao)

        # Ordena população
        pop_ordenada = [ind for _, ind in sorted(zip(aptidoes, populacao), key=lambda x: x[0])]
        aptidoes_ordenadas = sorted(aptidoes)

        # Atualiza melhor solução
        if aptidoes_ordenadas[0] < melhor_dist:
            melhor_dist = aptidoes_ordenadas[0]
            melhor_rota = pop_ordenada[0]
            if verbose:
                print(f"Avaliações {avaliacoes} - Nova melhor distância: {melhor_dist:.2f}")

        nova_pop = []
        nova_pop.extend(pop_ordenada[:num_elitistas])  # elitismo garantido

        clientes_ids = [i for i in range(1, n_nos + 1) if i != deposito]

        # Geração dos novos indivíduos
        while len(nova_pop) < parametros['TAMANHO_POPULACAO']:
            # Seleção pais
            if selecao_tipo == 'torneio':
                pai1 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])
                pai2 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])
            elif selecao_tipo == 'roleta':
                pai1 = selecao_roleta(pop_ordenada, aptidoes_ordenadas)
                pai2 = selecao_roleta(pop_ordenada, aptidoes_ordenadas)
            else:
                pai1 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])
                pai2 = selecao_torneio(pop_ordenada, aptidoes_ordenadas, parametros['TORNEIO_K'])

            filho1, filho2 = cycle_crossover(pai1, pai2)

            filho1 = mutacao(filho1, taxa_mutacao)
            filho2 = mutacao(filho2, taxa_mutacao)

            if len(nova_pop) + 2 <= parametros['TAMANHO_POPULACAO']:
                nova_pop.extend([filho1, filho2])
            else:
                nova_pop.append(filho1)

        populacao = nova_pop

    return melhor_dist, melhor_rota



# ============================
# Visualização da Rota
# ============================
def plotar_rota(rota, coordenadas, deposito, caminho_salvar):
    """
    Gera um gráfico da rota e salva como imagem PNG.
    Adiciona o depósito no início e no fim para plotar a rota completa.

    Parâmetros:
        rota (list): Rota dos clientes (sem depósito)
        coordenadas (dict): Coordenadas dos nós
        deposito (int): ID do depósito
        caminho_salvar (str): Caminho do arquivo PNG
    """
    plt.figure(figsize=(8, 6))

    rota_completa = [deposito] + rota + [deposito]

    xs = [coordenadas[i][0] for i in rota_completa]
    ys = [coordenadas[i][1] for i in rota_completa]
    plt.plot(xs, ys, color='green', linewidth=2, zorder=1)

    # Clientes
    clientes = rota
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
