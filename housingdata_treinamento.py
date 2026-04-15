from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import pandas as pd
import math
import pickle
import numpy as np

dados = pd.read_csv('HousingData.csv', sep=',')

for coluna in dados.columns:
    moda = dados[coluna].mode()[0]
    dados[coluna] = dados[coluna].fillna(moda)

scaler = MinMaxScaler()

normalizador = scaler.fit(dados)
pickle.dump(normalizador, open('normalizador_housingdata.pkl', 'wb'))

dados_normalizados = normalizador.fit_transform(dados)
dados_normalizados = pd.DataFrame(dados_normalizados, columns=dados.columns)

distorcoes = []

K = range(1, 101)

for i in K:
    cluster_housingdata = KMeans(n_clusters=i, random_state=42).fit(dados_normalizados)

    distorcoes.append(
        sum(
            np.min(
                cdist(
                    dados_normalizados, cluster_housingdata.cluster_centers_, 'euclidean'
                ), axis=1
            ) / dados_normalizados.shape[0]
        )
    )

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []

for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)

numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('Numero otimo de clusters =', numero_clusters_otimo)

cluster_housingdata = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(dados_normalizados)
pickle.dump(cluster_housingdata, open('cluster_housingdata.pkl', 'wb'))