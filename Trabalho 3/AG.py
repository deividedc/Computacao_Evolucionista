import numpy as np
import random
import math
import itertools
import matplotlib.pyplot as plt

# =========================================
# Função para ler arquivo no formato EVRP
# =========================================
def ler_arquivo_evrp(caminho_arquivo):
    """
    Lê arquivo de instância EVRP e retorna um dicionário com:
    - coordenadas dos nós (clientes, depósito, estações)
    - demandas dos clientes
    - depósito
    - estações de recarga
    - capacidade do veículo
    - energia máxima e consumo
    - número de veículos
    """
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()

    coordenadas = {}
    demandas = {}
    estacoes = set()
    deposito = None
    capacidade = None
    energia_max = None
    consumo = None
    numero_veiculos = None

    em_coord = em_demanda = em_estacoes = False  # flags para seções do arquivo

    for i, linha in enumerate(linhas):
        linha = linha.strip()
        # Captura parâmetros básicos
        if linha.startswith("CAPACITY"):
            capacidade = int(linha.split(":")[1].strip())
        elif linha.startswith("ENERGY_CAPACITY"):
            energia_max = float(linha.split(":")[1].strip())
        elif linha.startswith("ENERGY_CONSUMPTION"):
            consumo = float(linha.split(":")[1].strip())
        elif linha.startswith("VEHICLES"):
            numero_veiculos = int(linha.split(":")[1].strip())
        # Flags para ler as seções de coordenadas, demandas e estações
        elif linha.startswith("NODE_COORD_SECTION"):
            em_coord = True
            em_demanda = em_estacoes = False
            continue
        elif linha.startswith("DEMAND_SECTION"):
            em_coord = False
            em_demanda = True
            em_estacoes = False
            continue
        elif linha.startswith("STATIONS_COORD_SECTION"):
            em_coord = False
            em_demanda = False
            em_estacoes = True
            continue
        elif linha.startswith("DEPOT_SECTION"):
            em_coord = em_demanda = em_estacoes = False
            deposito = int(linhas[i+1].strip())  # Assume que depósito está na linha seguinte
            continue

        # Leitura das coordenadas dos nós
        if em_coord:
            partes = linha.split()
            if len(partes) == 3:
                id_no, x, y = map(float, partes)
                coordenadas[int(id_no)] = (x, y)
        # Leitura das demandas dos clientes
        elif em_demanda:
            partes = linha.split()
            if len(partes) == 2:
                id_no, demanda = map(int, partes)
                demandas[id_no] = demanda
        # Leitura das estações de recarga
        elif em_estacoes and linha != "" and linha != "EOF":
            estacoes.add(int(linha))

    # Define clientes excluindo depósito e estações
    clientes = set(coordenadas.keys()) - {deposito} - estacoes

    # Retorna dicionário com todos os dados lidos
    return {
        'coordenadas': coordenadas,
        'deposito': deposito,
        'demandas': demandas,
        'estacoes': estacoes,
        'clientes': clientes,
        'capacidade': capacidade,
        'energia_max': energia_max,
        'consumo': consumo,
        'numero_veiculos': numero_veiculos
    }

# =========================================
# Função para calcular distância Euclidiana
# =========================================
def distancia(a, b):
    """
    Recebe duas coordenadas (tuplas) e calcula a distância euclidiana.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

# =========================================
# Avalia o custo (distância total) de uma rota, considerando restrições
# =========================================
def avaliar(ro, dados):
    """
    Avalia uma solução (rota) somando distâncias e penalizando violações:
    - Capacidade máxima do veículo
    - Energia máxima e consumo
    - Número mínimo de veículos permitidos
    - Penalização severa se o veículo ficar sem energia viável
    Retorna a distância total (fitness) da solução.
    """
    coords = dados['coordenadas']
    demandas = dados['demandas']
    capacidade = dados['capacidade']
    energia_max = dados['energia_max']
    consumo = dados['consumo']
    deposito = dados['deposito']
    estacoes = dados['estacoes']
    num_veiculos = dados['numero_veiculos']

    total_dist = 0
    carga = 0
    energia = energia_max
    atual = deposito
    rotas = 1

    for cliente in ro:
        demanda = demandas[cliente]
        dist = distancia(coords[atual], coords[cliente])
        energia_necessaria = dist * consumo

        if carga + demanda > capacidade:
            dist_volta = distancia(coords[atual], coords[deposito])
            energia_volta = dist_volta * consumo
            if energia < energia_volta:
                return 1e10
            total_dist += dist_volta
            atual = deposito
            carga = 0
            energia = energia_max
            rotas += 1
            dist = distancia(coords[atual], coords[cliente])
            energia_necessaria = dist * consumo

        if energia < energia_necessaria:
            estacao_escolhida = escolher_estacao_melhorada(atual, cliente, energia, dados)
            if estacao_escolhida is not None:
                total_dist += distancia(coords[atual], coords[estacao_escolhida])
                energia = energia_max
                atual = estacao_escolhida
                dist = distancia(coords[atual], coords[cliente])
                energia_necessaria = dist * consumo
            else:
                dist_volta = distancia(coords[atual], coords[deposito])
                energia_volta = dist_volta * consumo
                if energia < energia_volta:
                    return 1e10
                total_dist += dist_volta
                atual = deposito
                carga = 0
                energia = energia_max
                rotas += 1
                dist = distancia(coords[atual], coords[cliente])
                energia_necessaria = dist * consumo

        energia -= energia_necessaria
        carga += demanda
        total_dist += dist
        atual = cliente

    dist_volta = distancia(coords[atual], coords[deposito])
    energia_volta = dist_volta * consumo
    if energia < energia_volta:
        return 1e10
    total_dist += dist_volta

    if rotas < num_veiculos:
        total_dist += 1e6 * (num_veiculos - rotas)

    return total_dist

    # Penalização se número de veículos usados for menor que o mínimo exigido
    if rotas < num_veiculos:
        total_dist += 1e6 * (num_veiculos - rotas)

    return total_dist


# =========================================
# Gera um indivíduo (rota)
# =========================================

# =========================================
# ACO
# =========================================

def gerar_individuo_aco(dados, num_iteracoes=10, alpha=1.0, beta=3.0, rho=0.1, Q=1.0):
    coords = dados['coordenadas']
    demandas = dados['demandas']
    clientes = list(dados['clientes'])
    estacoes = dados['estacoes']
    deposito = dados['deposito']
    capacidade = dados['capacidade']
    energia_max = dados['energia_max']
    consumo = dados['consumo']

    n = len(clientes)
    nodes = [deposito] + clientes + list(estacoes)
    node_indices = {node: i for i, node in enumerate(nodes)}

    dist_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            dist_matrix[i][j] = distancia(coords[nodes[i]], coords[nodes[j]])

    pheromones = np.ones((len(nodes), len(nodes)))

    melhor_rota = None
    melhor_custo = float('inf')

    for _ in range(num_iteracoes):
        nao_visitados = set(clientes)
        rota = []
        carga = 0
        energia = energia_max
        atual = deposito

        while nao_visitados:
            i_atual = node_indices[atual]
            candidatos = []
            probs = []

            for c in nao_visitados:
                i_c = node_indices[c]
                demanda_c = demandas[c]
                dist = dist_matrix[i_atual][i_c]
                energia_necessaria = dist * consumo

                if carga + demanda_c <= capacidade and energia >= energia_necessaria:
                    tau = pheromones[i_atual][i_c] ** alpha
                    eta = (1.0 / (dist + 1e-6)) ** beta
                    candidatos.append(c)
                    probs.append(tau * eta)

            if not candidatos:
                carga = 0
                energia = energia_max
                atual = deposito
                continue

            total = sum(probs)
            probs = [p / total for p in probs]
            escolhido = random.choices(candidatos, weights=probs, k=1)[0]

            rota.append(escolhido)
            dist_percorrido = dist_matrix[node_indices[atual]][node_indices[escolhido]]
            energia -= dist_percorrido * consumo
            carga += demandas[escolhido]
            atual = escolhido
            nao_visitados.remove(escolhido)

        custo = avaliar(rota, dados)  # só float
        rota_corrigida = reconstruir_rota(rota, dados)  # corrige rota incluindo depósito e estações

        # Atualiza feromônios na rota corrigida
        for i in range(len(rota_corrigida)-1):
            a = node_indices[rota_corrigida[i]]
            b = node_indices[rota_corrigida[i+1]]
            pheromones[a][b] = (1 - rho) * pheromones[a][b] + rho * Q / (dist_matrix[a][b] + 1e-6)
            pheromones[b][a] = pheromones[a][b]

        if custo < melhor_custo:
            melhor_custo = custo
            melhor_rota = rota

    return melhor_rota





# ========================================
# KNN
# ========================================

def gerar_individuo_guloso(dados):
    coords = dados['coordenadas']
    demandas = dados['demandas']
    clientes_disponiveis = set(dados['clientes'])
    deposito = dados['deposito']
    capacidade = dados['capacidade']
    energia_max = dados['energia_max']
    consumo = dados['consumo']

    atual = deposito
    energia = energia_max
    carga = 0
    rota = []

    while clientes_disponiveis:
        proximos = sorted(
            clientes_disponiveis,
            key=lambda c: distancia(coords[atual], coords[c])
        )

        for cliente in proximos:
            demanda = demandas[cliente]
            dist = distancia(coords[atual], coords[cliente])
            energia_necessaria = dist * consumo

            if carga + demanda <= capacidade and energia >= energia_necessaria:
                rota.append(cliente)
                carga += demanda
                energia -= energia_necessaria
                atual = cliente
                clientes_disponiveis.remove(cliente)
                break
        else:
            atual = deposito
            energia = energia_max
            carga = 0

    return rota


def gerar_individuo(dados):
    """
    Gera um indivíduo embaralhando aleatoriamente a lista de clientes.
    """
    clientes = list(dados['clientes'])
    random.shuffle(clientes)
    return clientes

# ============================================================
# Seleção por torneio
# ============================================================
def torneio(populacao, aptidoes, k):
    """
    Realiza seleção por torneio:
    - seleciona k indivíduos aleatoriamente
    - retorna o melhor (menor aptidão)
    """
    selecionados = random.sample(range(len(populacao)), k)
    melhor = min(selecionados, key=lambda i: aptidoes[i])
    return populacao[melhor][:]  # retorna cópia do indivíduo

# ============================================================
# Operadores de Crossover
# ============================================================

def crossover_PMX(pai1, pai2):
    n = len(pai1)
    i, j = sorted(random.sample(range(n), 2))
    filho = [None] * n

    # Copia segmento do pai1
    filho[i:j] = pai1[i:j]

    # Cria mapeamentos bidirecionais
    map1 = {}
    map2 = {}
    for a, b in zip(pai1[i:j], pai2[i:j]):
        map1[b] = a
        map2[a] = b

    def resolve_conflito(gene):
        # Resolve conflito navegando no mapeamento até achar gene não conflituoso
        while gene in filho[i:j]:
            gene = map1.get(gene, gene)
        return gene

    # Preenche posições antes e depois do segmento
    for idx in list(range(0, i)) + list(range(j, n)):
        gene = pai2[idx]
        if gene in filho[i:j]:
            gene = resolve_conflito(gene)
        filho[idx] = gene

    return filho



def crossover_OX(pai1, pai2):
    """
    Crossover Order Crossover (OX) para permutações:
    - escolhe um trecho contínuo de pai1
    - preenche o restante pela ordem de pai2 sem repetição
    """
    n = len(pai1)
    i, j = sorted(random.sample(range(n), 2))
    filho = [None] * n
    filho[i:j] = pai1[i:j]
    pos = j
    for g in pai2:
        if g not in filho:
            if pos >= n: pos = 0
            filho[pos] = g
            pos += 1
    return filho

def crossover_CX(pai1, pai2):
    """
    Crossover Cycle Crossover (CX):
    - preserva ciclos entre pais para garantir posição relativa
    """
    n = len(pai1)
    filho = [None] * n
    ciclos = set()
    idx = 0
    while idx not in ciclos:
        ciclos.add(idx)
        gene = pai2[idx]
        idx = pai1.index(gene)
    for i in ciclos:
        filho[i] = pai1[i]
    for i in range(n):
        if filho[i] is None:
            filho[i] = pai2[i]
    return filho

def crossover(pai1, pai2, tipo="OX"):
    """
    Wrapper para escolher o tipo de crossover entre OX,CX e PMX.
    """
    if tipo == "CX":
        return crossover_CX(pai1, pai2)
    elif tipo == "PMX":
        return crossover_PMX(pai1, pai2)
    else:
        return crossover_OX(pai1, pai2)
    
# ============================
# População inicial
# ============================

def gerar_populacao_inicial(dados, tamanho_pop, perc_guloso =0.5, perc_aco=0.5):
    """
    Gera população inicial combinando:
      - perc_guloso  fraction usando gerar_individuo_guloso 
      - perc_aco  fraction usando gerar_individuo_aco (ACO)
      - restante usando gerar_individuo (aleatório)
    perc_guloso  and perc_aco are floats entre 0 e 1. Se a soma > 1.0, normaliza-se.
    Retorna lista de indivíduos (cada indivíduo = lista/permutação de clientes).
    """
    # Normaliza percentuais se necessário
    if perc_guloso  < 0: perc_guloso  = 0.0
    if perc_aco < 0: perc_aco = 0.0
    s = perc_guloso  + perc_aco
    if s > 1e-12 and s != 1.0:
        if s > 1.0:
            perc_guloso  = perc_guloso  / s
            perc_aco = perc_aco / s
        # se soma < 1.0: sobra para aleatórios, mantemos perc_guloso  e perc_aco como estão

    num_knn = int(round(tamanho_pop * perc_guloso ))
    num_aco = int(round(tamanho_pop * perc_aco))
    # Ajuste final para garantir exatamente tamanho_pop elementos
    total = num_knn + num_aco
    num_random = max(0, tamanho_pop - total)

    populacao = []

    # Gera indivíduos KNN (guloso)
    for _ in range(num_knn):
        ind = gerar_individuo_guloso(dados)
        # garante formato
        if ind is None:
            ind = gerar_individuo(dados)
        populacao.append(ind)

    # Gera indivíduos ACO
    for _ in range(num_aco):
        # ACO pode ocasionalmente retornar None; tenta algumas vezes
        tentativa = 0
        ind = None
        while tentativa < 5 and ind is None:
            ind = gerar_individuo_aco(dados)
            tentativa += 1
        if ind is None:
            ind = gerar_individuo(dados)
        populacao.append(ind)

    # Gera indivíduos aleatórios para completar
    for _ in range(num_random):
        populacao.append(gerar_individuo(dados))

    # Se por arredondamento ficar maior que tamanho_pop, corta
    if len(populacao) > tamanho_pop:
        populacao = populacao[:tamanho_pop]

    # Embaralha a população para misturar origens (opcional)
    random.shuffle(populacao)
    return populacao


# ============================
# Mutações
# ============================

def mutacao_swap(individuo):
    """
    Mutação que troca dois genes aleatórios de posição.
    """
    a, b = random.sample(range(len(individuo)), 2)
    individuo[a], individuo[b] = individuo[b], individuo[a]
    return individuo

def mutacao_inversao(individuo):
    """
    Mutação que inverte uma subsequência aleatória da rota.
    """
    a, b = sorted(random.sample(range(len(individuo)), 2))
    individuo[a:b+1] = individuo[a:b+1][::-1]
    return individuo

def mutacao(individuo, taxa_mutacao):
    """
    Aplica mutação probabilisticamente segundo taxa_mutacao.
    Escolhe aleatoriamente entre swap e inversão para diversificar busca.
    """
    if random.random() < taxa_mutacao:
        if random.random() < 0.5:
            individuo = mutacao_swap(individuo)
        else:
            individuo = mutacao_inversao(individuo)
    return individuo

# =========================================
# Reconstruir rota incluindo depósito e estações
# =========================================
def reconstruir_rota(ro, dados):
    """
    Insere automaticamente depósitos e estações de recarga na rota para
    respeitar restrições de capacidade e energia.
    Retorna a rota completa com esses nós inseridos.
    """
    coords = dados['coordenadas']
    demandas = dados['demandas']
    capacidade = dados['capacidade']
    energia_max = dados['energia_max']
    consumo = dados['consumo']
    deposito = dados['deposito']

    rota_completa = [deposito]
    carga = 0
    energia = energia_max
    atual = deposito

    for cliente in ro:
        demanda = demandas[cliente]
        dist = distancia(coords[atual], coords[cliente])
        energia_necessaria = dist * consumo

        if carga + demanda > capacidade:
            dist_volta = distancia(coords[atual], coords[deposito])
            energia_volta = dist_volta * consumo
            if energia < energia_volta:
                return [deposito]
            rota_completa.append(deposito)
            atual = deposito
            carga = 0
            energia = energia_max
            dist = distancia(coords[atual], coords[cliente])
            energia_necessaria = dist * consumo

        if energia < energia_necessaria:
            estacao_escolhida = escolher_estacao_melhorada(atual, cliente, energia, dados)
            if estacao_escolhida is not None:
                rota_completa.append(estacao_escolhida)
                energia = energia_max
                atual = estacao_escolhida
                dist = distancia(coords[atual], coords[cliente])
                energia_necessaria = dist * consumo
            else:
                dist_volta = distancia(coords[atual], coords[deposito])
                energia_volta = dist_volta * consumo
                if energia < energia_volta:
                    return [deposito]
                rota_completa.append(deposito)
                atual = deposito
                carga = 0
                energia = energia_max
                dist = distancia(coords[atual], coords[cliente])
                energia_necessaria = dist * consumo

        rota_completa.append(cliente)
        carga += demanda
        energia -= energia_necessaria
        atual = cliente

    dist_final = distancia(coords[atual], coords[deposito])
    energia_final = dist_final * consumo

    if energia < energia_final:
        return [deposito]
    rota_completa.append(deposito)

    return rota_completa



def plotar_rota(rota, dados, nome_arquivo):
    """
    Gera um gráfico das rotas, incluindo clientes, depósitos e estações.
    Salva a figura em arquivo.
    """
    coords = dados['coordenadas']
    deposito = dados['deposito']
    estacoes = dados['estacoes']

    plt.figure(figsize=(8, 6))
    cores = itertools.cycle(['blue', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'yellow', 'black'])

    # Índices no vetor da rota onde está o depósito
    indices_deposito = [i for i, no in enumerate(rota) if no == deposito]

    # Filtra índices para ignorar depósitos consecutivos
    indices_filtrados = [indices_deposito[0]]
    for idx in indices_deposito[1:]:
        if idx != indices_filtrados[-1] + 1:
            indices_filtrados.append(idx)

    # Para cada segmento entre depósitos, plota uma rota com cor diferente
    for idx in range(len(indices_filtrados) - 1):
        cor = next(cores)
        inicio = indices_filtrados[idx]
        fim = indices_filtrados[idx + 1]

        xs = [coords[rota[i]][0] for i in range(inicio, fim + 1)]
        ys = [coords[rota[i]][1] for i in range(inicio, fim + 1)]

        plt.plot(xs, ys, '-o', color=cor, label=f'Rota {idx + 1}')

    # Marca depósito com estrela vermelha
    x_dep, y_dep = coords[deposito]
    plt.plot(x_dep, y_dep, 'r*', markersize=15, label='Depósito')

    # Marca estações com quadrados verdes
    for est in estacoes:
        x_est, y_est = coords[est]
        plt.plot(x_est, y_est, 'gs', markersize=5, label='Estação de Recarga')

    # Remove legendas repetidas
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict()
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h

    # Coloca a legenda fora do gráfico, à direita
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.title("Rotas do Veículo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()  # Ajusta layout para dar espaço para a legenda
    plt.savefig(nome_arquivo, bbox_inches='tight')  # salva imagem com margem ajustada
    plt.close()  # fecha figura para liberar memória


# =========================================
# Calcula a carga transportada em cada rota
# =========================================
def calcular_cargas_por_rota(rota, dados):
    """
    Para cada rota entre depósitos, soma as demandas dos clientes (excluindo estações).
    Retorna lista de cargas por rota.
    """
    demandas = dados['demandas']
    deposito = dados['deposito']
    estacoes = dados['estacoes']

    # Índices onde está o depósito
    indices_deposito = [i for i, no in enumerate(rota) if no == deposito]

    cargas = []
    # Para cada trecho entre depósitos, calcula carga somada
    for idx in range(len(indices_deposito) - 1):
        inicio = indices_deposito[idx]
        fim = indices_deposito[idx + 1]
        carga_rota = 0
        for no in rota[inicio + 1:fim]:
            if no not in estacoes and no != deposito:
                carga_rota += demandas.get(no, 0)
        cargas.append(carga_rota)

    return cargas

# =========================================
# Função principal de evolução do Algoritmo Genético
# =========================================
def evoluir(dados, parametros):
    """
    Executa o AG com os parâmetros definidos:
    - inicializa população
    - calcula aptidões
    - realiza seleção, crossover e mutação
    - mantém elite (melhores)
    - repete até atingir limite máximo de avaliações
    Retorna melhor solução e distância avaliada.
    """
    TAM = parametros['TAMANHO_POPULACAO']
    MUT = parametros['TAXA_MUTACAO']
    K = parametros['TORNEIO_K']
    ELITE = parametros['NUM_ELITISTAS']
    tipo_cross = parametros.get('TIPO_CROSSOVER', 'OX')

    n_nos = len(dados['coordenadas'])
    max_aval = 25000 * n_nos  # limite máximo de avaliações para parar

    # Cria população inicial aleatória
    # populacao = [gerar_individuo_guloso(dados) for _ in range(TAM)]
    # populacao = [gerar_individuo_aco(dados) for _ in range(TAM)]
    populacao = gerar_populacao_inicial(dados, TAM, parametros.get('PERC_', 0.5), parametros.get('PERC_ACO', 0.5))


    
    aptidoes = [avaliar(ind, dados) for ind in populacao]
    num_aval = TAM  # número de avaliações já feitas

    melhor_aptidao = min(aptidoes)  # melhor valor de aptidão na população
    melhor_individuo = populacao[np.argmin(aptidoes)]
    contador_estagnacao = 0

    while num_aval < max_aval:
        nova_pop = []

        # Copia elite para a próxima geração
        elites = sorted(zip(populacao, aptidoes), key=lambda x: x[1])[:ELITE]
        nova_pop.extend([e[0] for e in elites])

        # Preenche o restante da população com filhos gerados por seleção, crossover e mutação
        while len(nova_pop) < TAM:
            pai1 = torneio(populacao, aptidoes, K)
            pai2 = torneio(populacao, aptidoes, K)
            filho = crossover(pai1, pai2, tipo_cross)
            filho = mutacao(filho, MUT)
            nova_pop.append(filho)

        populacao = nova_pop
        aptidoes = [avaliar(ind, dados) for ind in populacao]
        num_aval += TAM

        min_apt = min(aptidoes)
        idx_min = np.argmin(aptidoes)

        # Atualiza melhor solução se encontrou melhor aptidão
        if min_apt < melhor_aptidao - 1e-6:
            melhor_aptidao = min_apt
            melhor_individuo = populacao[idx_min]
            contador_estagnacao = 0
        else:
            contador_estagnacao += 1

    # Retorna a melhor distância e a rota reconstruída com depósitos e estações
    return melhor_aptidao, reconstruir_rota(melhor_individuo, dados)










#==============================================================================
# Escolher estaçao
#==============================================================================
def escolher_estacao_melhorada(atual, cliente, energia, dados):
    """
    Escolhe a melhor estação de recarga com base em múltiplos critérios:
    - Energia viável
    - Custo total: atual → estação → cliente → depósito (lookahead)
    """
    coords = dados['coordenadas']
    estacoes = dados['estacoes']
    energia_max = dados['energia_max']
    consumo = dados['consumo']
    deposito = dados['deposito']

    melhor_estacao = None
    menor_custo = float('inf')

    for est in estacoes:
        dist_ate_est = distancia(coords[atual], coords[est])
        energia_ate_est = dist_ate_est * consumo

        dist_est_para_cli = distancia(coords[est], coords[cliente])
        energia_est_para_cli = dist_est_para_cli * consumo

        if energia >= energia_ate_est and energia_max >= energia_est_para_cli:
            dist_cli_para_dep = distancia(coords[cliente], coords[deposito])

            custo_total = dist_ate_est + dist_est_para_cli + 0.5 * dist_cli_para_dep  # pesos ajustáveis
            if custo_total < menor_custo:
                menor_custo = custo_total
                melhor_estacao = est

    return melhor_estacao
