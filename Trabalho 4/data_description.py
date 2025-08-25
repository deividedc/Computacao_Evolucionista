#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 12:28:30 2025

@author: deivid
"""

import numpy      as np
import pandas     as pd
import requests

from io import BytesIO


key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
# Acesso via link
link = 'https://docs.google.com/spreadsheet/ccc?key=' + key + '&output=csv'
r    = requests.get(link)
data = r.content

# Lendo o CSV com ajustes para evitar warnings
df = pd.read_csv(BytesIO(data), header=0, low_memory=False)

# Seleção e transformação de dados
cols = ['siteid', 'sampledate', 'itemengabbreviation', 'itemvalue']
data = df[cols]

# Pivotando os dados
data = data.pivot(index=['siteid', 'sampledate'], columns='itemengabbreviation', values='itemvalue')

# Adicionando a coluna 'site'
data['site'] = [data.index[i][0] for i in range(len(data))]

# Filtrando os dados
data = data[data['site'] < 1008]

# Selecionando as colunas de interesse
cols = ['EC', 'RPI', 'SS', 'WT', 'pH']
X    = data[cols].copy()  # Criação de cópia explícita para evitar SettingWithCopyError

# Convertendo as colunas para numérico e lidando com valores inválidos
for c in cols:
    X[c] = pd.to_numeric(X[c], errors='coerce')

# Removendo valores nulos
X = X.dropna()

# Obter a descrição detalhada (estatísticas padrão)
descricao = X.describe()

# Converter para LaTeX
tabela_latex = descricao.to_latex(float_format="%.2f")

print(tabela_latex)


