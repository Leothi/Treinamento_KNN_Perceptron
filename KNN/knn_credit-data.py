# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:23:45 2019

@author: Leonardo
"""
import pandas as pd

base = pd.read_csv('credit-data.csv')

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values # 0 conseguiu pagar, 1 não conseguiu

# Pré-processamento
base.loc[base['age']<0, 'age'] = 40.92

# Valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 0:3])

previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalonamento - deixando tudo na mesma escala pra não ter problemas com importância de atributos - deixa mais rápido
# Padronização ou Normalização
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # Padronização é mais recomendado

previsores = scaler.fit_transform(previsores)

# Divisão entre base de testes e treinamento
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

# Criação (para kNN é muito importante fazer escalonamento)
from sklearn.neighbors import KNeighborsClassifier
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # K= 3 ou 5

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Comparando a classe_teste e as previsões
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes) # % de acurácia - 
matriz = confusion_matrix(classe_teste, previsoes) # Matriz de confusão - importante saber ler
