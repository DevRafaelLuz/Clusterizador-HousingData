import pandas as pd
import pickle

dados = pd.read_csv('HousingData.csv', sep=',')

nomes_colunas = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

cluster_housingdata = pickle.load(open('cluster_housingdata.pkl', 'rb'))

centroides = pd.DataFrame(cluster_housingdata.cluster_centers_, columns=nomes_colunas)

dados_normalizados = centroides

normalizador = pickle.load(open('normalizador_housingdata.pkl', 'rb'))
dados = normalizador.inverse_transform(dados_normalizados)
dados = pd.DataFrame(dados, columns=dados_normalizados.columns)

dados = dados.rename(columns={
    'CRIM': 'TAXA DE CRIMINALIDADE PER CAPITA POR CIDADE',
    'ZN': 'PROPORCAO TERRENOS ZONIFICADOS PARA LOTES COM MAIS DE 25.000 sq.ft.',
    'INDUS': 'HECTARES DE NEGOCIOS NAO VAREJISTAS POR CIDADE',
    'CHAS': 'VARIAVEL FICTICIA DE CHARLES RIVER',
    'NOX': 'CONCENTRACAO DE OXIDOS NITRICOS',
    'RM': 'NUMERO MEDIO DE QUARTOS POR MORADIA',
    'AGE': 'PROPORCAO DE UNIDADES OCUPADAS ANTES DE 1940',
    'DIS': 'DISTANCIAS PONDERADAS PARA CINCO CENTROS DE EMPREGO',
    'RAD': 'INDICE DE ACESSIBILIDADE AS RODOVIAS RADIAIS',
    'TAX': 'ALIQUOTA DO IMPOSTO SOBRE PROPRIEDADE DE VALOR TOTAL POR $10.000',
    'PTRATIO': 'RELACAO ALUNO-PROFESSOR POR CIDADE',
    'B': 'PROPORCAO DE NEGROS POR CIDADE',
    'LSTAT': '% MENOR STATUS DA POPULACAO',
    'MEDV': 'VALOR MEDIANO DE CASAS OCUPADAS EM MILHARES DE COLARES'
})

print(dados)