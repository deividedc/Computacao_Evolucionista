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
    - capacidade máxima do veículo
    - energia máxima e consumo
    - número máximo de veículos permitidos
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

    total_dist = 0  # acumulador da distância total
    carga = 0  # carga atual do veículo
    energia = energia_max  # energia disponível inicialmente
    atual = deposito  # nó atual, inicia no depósito
    rotas = 1  # conta número de rotas usadas (veículos)

    for cliente in ro:
        demanda = demandas[cliente]
        dist = distancia(coords[atual], coords[cliente])
        energia_necessaria = dist * consumo  # energia consumida para ir até o cliente

        # Se a carga excede a capacidade, volta para depósito e reinicia
        if carga + demanda > capacidade:
            total_dist += distancia(coords[atual], coords[deposito])
            atual = deposito
            carga = 0
            energia = energia_max
            rotas += 1

        # Se não tem energia suficiente para o cliente, tenta usar estação de recarga
        elif energia < energia_necessaria:
            estacao_escolhida = None
            menor_dist = float('inf')
            for est in estacoes:
                dist_para_est = distancia(coords[atual], coords[est])
                dist_est_para_cli = distancia(coords[est], coords[cliente])
                energia_necessaria_para_est = dist_para_est * consumo
                energia_necessaria_para_cli = dist_est_para_cli * consumo

                # Verifica se é possível ir até estação e depois ao cliente
                if energia >= energia_necessaria_para_est and energia_max >= energia_necessaria_para_cli:
                    rota_com_estacao = dist_para_est + dist_est_para_cli
                    if rota_com_estacao < menor_dist:
                        menor_dist = rota_com_estacao
                        estacao_escolhida = est

            # Se encontrou estação viável, vai até ela antes do cliente
            if estacao_escolhida is not None:
                total_dist += distancia(coords[atual], coords[estacao_escolhida])
                energia = energia_max
                atual = estacao_escolhida
                dist = distancia(coords[atual], coords[cliente])
                energia_necessaria = dist * consumo
                energia -= energia_necessaria
                carga += demanda
                total_dist += dist
                atual = cliente
            else:
                # Se não, retorna ao depósito para recarregar e reinicia rota
                total_dist += distancia(coords[atual], coords[deposito])
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
        else:
            # Caso normal: segue para o cliente normalmente
            carga += demanda
            energia -= energia_necessaria
            total_dist += dist
            atual = cliente

    # Volta final ao depósito
    total_dist += distancia(coords[atual], coords[deposito])

    # Penaliza se número de rotas (veículos) usadas for menor que o permitido
    if rotas < num_veiculos:
        total_dist += 1e6 * (num_veiculos - rotas)

    return total_dist

# =========================================
# Gera um indivíduo (rota) aleatória inicial
# =========================================
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
    Wrapper para escolher o tipo de crossover entre OX e CX.
    """
    if tipo == "CX":
        return crossover_CX(pai1, pai2)
    else:
        return crossover_OX(pai1, pai2)

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
    estacoes = dados['estacoes']

    rota_completa = [deposito]  # inicia em depósito
    carga = 0
    energia = energia_max
    atual = deposito

    for cliente in ro:
        demanda = demandas[cliente]
        dist = distancia(coords[atual], coords[cliente])
        energia_necessaria = dist * consumo

        # Se extrapola capacidade, volta ao depósito
        if carga + demanda > capacidade:
            rota_completa.append(deposito)
            atual = deposito
            carga = 0
            energia = energia_max

        # Se energia insuficiente para ir direto, tenta estação
        elif energia < energia_necessaria:
            estacao_escolhida = None
            menor_dist = float('inf')
            for est in estacoes:
                dist_para_est = distancia(coords[atual], coords[est])
                dist_est_para_cli = distancia(coords[est], coords[cliente])
                energia_necessaria_para_est = dist_para_est * consumo
                energia_necessaria_para_cli = dist_est_para_cli * consumo

                # Verifica se pode ir à estação e depois ao cliente
                if energia >= energia_necessaria_para_est and energia_max >= energia_necessaria_para_cli:
                    rota_com_estacao = dist_para_est + dist_est_para_cli
                    if rota_com_estacao < menor_dist:
                        menor_dist = rota_com_estacao
                        estacao_escolhida = est

            if estacao_escolhida is not None:
                rota_completa.append(estacao_escolhida)
                energia = energia_max
                atual = estacao_escolhida

        rota_completa.append(cliente)
        carga += demanda
        energia -= energia_necessaria
        atual = cliente

    rota_completa.append(deposito)  # finaliza em depósito
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
    populacao = [gerar_individuo(dados) for _ in range(TAM)]
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
