import pandas as pd
import pickle

imovel_normalizado = pd.DataFrame(columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])

novo_imovel = pd.DataFrame([[0.13545, 18, 6.48, 0, 0.399, 4.679, 87.6, 4.6165, 3, 271, 16.3, 391.5, 5.05, 30]], columns=['CRIM', 
                                          'ZN', 
                                          'INDUS', 
                                          'CHAS', 
                                          'NOX', 
                                          'RM', 
                                          'AGE', 
                                          'DIS', 
                                          'RAD', 
                                          'TAX', 
                                          'PTRATIO', 
                                          'B', 
                                          'LSTAT', 
                                          'MEDV'])

normalizador = pickle.load(open('normalizador_housingdata.pkl', 'rb'))
novo_imovel = normalizador.transform(novo_imovel)
novo_imovel = pd.DataFrame(novo_imovel, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
novo_imovel_normalizado = pd.concat([novo_imovel, imovel_normalizado]).fillna(0)

cluster_housingdata = pickle.load(open('cluster_housingdata.pkl', 'rb'))
cluster_novo_imovel = cluster_housingdata.predict(novo_imovel_normalizado)
print('Cluster do novo imovel: ', cluster_novo_imovel)