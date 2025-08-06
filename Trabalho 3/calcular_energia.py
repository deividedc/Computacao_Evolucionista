import math

# Coordenadas dos nós (id: (x, y))
coordenadas = {
    1: (266, 235),
    2: (295, 272),
    3: (301, 258),
    4: (309, 260),
    5: (217, 274),
    6: (218, 278),
    7: (282, 267),
    8: (242, 249),
    9: (230, 262),
    10: (249, 268),
    11: (256, 267),
    12: (265, 257),
    13: (267, 242),
    14: (259, 265),
    15: (315, 233),
    16: (329, 252),
    17: (318, 252),
    18: (329, 224),
    19: (267, 213),
    20: (275, 192),
    21: (303, 201),
    22: (208, 217),
    23: (326, 181),
    24: (229, 198),
    25: (229, 230),
    26: (229, 262),
    27: (269, 198),
    28: (269, 230),
    29: (269, 262),
    30: (309, 198),
    31: (309, 230),
    32: (309, 262),
}

# Sequência da rota
rota = [1, 8, 22, 5, 6, 9, 26, 10, 12, 13, 1, 11, 14, 1, 7, 2, 3, 4, 17, 16, 18, 15, 31, 23, 21, 20, 19, 1]# 26,9, 10, 12, 13, 1]

# Função para calcular distância euclidiana entre dois pontos
def distancia_euclidiana(p1, p2):
    x1, y1 = coordenadas[p1]
    x2, y2 = coordenadas[p2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Separar rota em subrotas
subrotas = []
rota_atual = [rota[0]]

distancia = 0

for i in range(len(rota)-1):
    distancia = distancia + distancia_euclidiana(rota[i], rota[i+1])
    
print(f"distancia: {distancia}")

print(f"energia: {distancia*1.2}")
    


