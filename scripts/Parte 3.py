import statsmodels.api as sm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import auc

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

from src.carga_datos import CargaDatos
ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

cargador.cargar_datos()
cargador.mostrar_dimensiones()

jugadores=cargador.datos
print(jugadores.head())

n=len(jugadores["Gls"])
n_train= int(n*0.8)
n_test= n-n_train
random.seed(40)
indices_train=random.sample(range(n), n_train)
jugadores_train=jugadores.iloc[indices_train]
jugadores_test=jugadores.drop(indices_train)

#Variables train
gls = 1*(jugadores_train["Gls"] > 0)
edad = jugadores_train["Age"] 
partidos_jugados = jugadores_train["MP"] 
min=jugadores_train["Min"] 
tiros=jugadores_train["Sh"] 
tiros_al_arco=jugadores_train["SoT"] 
xG=jugadores_train["xG"] 
bloques_90min=jugadores_train["90s"] 
tiros_90min=jugadores_train["Sh/90"]  
acciones_de_gol=jugadores_train["SCA"] 
conducciones_progresivas=jugadores_train["PrgC"]


#Variables test
gls_test = 1*(jugadores_test["Gls"] > 0)
edad_test = jugadores_test["Age"] 
partidos_jugados_test = jugadores_test["MP"] 
min_test=jugadores_test["Min"] 
tiros_test=jugadores_test["Sh"] 
tiros_al_arco_test=jugadores_test["SoT"] 
xG_test=jugadores_test["xG"] 
bloques_90min_test=jugadores_test["90s"] 
tiros_90min_test=jugadores_test["Sh/90"]  
acciones_de_gol_test=jugadores_test["SCA"] 
conducciones_progresivas_test=jugadores_test["PrgC"]

#Planteamos modelo 1
print(jugadores_train.isnull().sum())  
#Sacamas los nulos de edad:
jugadores_sin_nulos_train = jugadores_train.dropna(subset=["Age"])
tiros_90min_sin_nulos = jugadores_sin_nulos_train["Sh/90"]
conducciones_progresivas_sin_nulos = jugadores_sin_nulos_train["PrgC"]
min_sin_nulos = jugadores_sin_nulos_train["Min"]
edad_sin_nulos = jugadores_sin_nulos_train["Age"]
gls_sin_nulos = 1*(jugadores_sin_nulos_train["Gls"] > 0)

jugadores_sin_nulos_test = jugadores_test.dropna(subset=["Age"])
tiros_90min_sin_nulos_test = jugadores_sin_nulos_test["Sh/90"]
conducciones_progresivas_sin_nulos_test = jugadores_sin_nulos_test["PrgC"]
min_sin_nulos_test = jugadores_sin_nulos_test["Min"]
edad_sin_nulos_test = jugadores_sin_nulos_test["Age"]
gls_sin_nulos_test = 1*(jugadores_sin_nulos_test["Gls"] > 0)

#Modelo 1: Edad, Minutos, Tiros cada 90 minutos y Conducciones Progresivas
x=np.column_stack((edad_sin_nulos,min_sin_nulos,tiros_90min_sin_nulos,conducciones_progresivas_sin_nulos))
X=sm.add_constant(x)
modelo_train = sm.Logit(gls_sin_nulos, X)
resultado_train = modelo_train.fit()
print(resultado_train.summary())

#Armamos la tabla de predicciones para el set de test
x_test=np.column_stack((edad_sin_nulos_test,min_sin_nulos_test,tiros_90min_sin_nulos_test,conducciones_progresivas_sin_nulos_test))
X_test=sm.add_constant(x_test)
predicciones_test = np.dot(X_test, resultado_train.params)
prob_pred=np.exp(predicciones_test) / (1 + np.exp(predicciones_test))

#Calculamos el mejor punto de corte
# Generar valores de p
p_values = np.linspace(0, 1, 100)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred = 1 * (prob_pred >= p)

    a = np.sum((y_pred == 1) & (gls_sin_nulos_test == 1))  # Verdaderos positivos
    b = np.sum((y_pred==1) & (gls_sin_nulos_test==0))  # Falsos positivos
    c = np.sum((y_pred== 0) & (gls_sin_nulos_test==1))  # Falsos negativos
    d = np.sum((y_pred==0)&(gls_sin_nulos_test==0))  # Verdaderos negativos
    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (b+d)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

#Calculamos el índice de Youden y encontramos el mejor punto de corte
j = np.array(sensibilidad) + np.array(especificidad) - 1
print("Mejor punto de corte:", p_values[j==max(j)])
print("Sensibilidad:", np.array(sensibilidad)[j==max(j)])
print("Especificidad:", np.array(especificidad)[j==max(j)])

#Graficamos la curva ROC
p1=1-np.array(especificidad)[j==max(j)]
p2=np.array(sensibilidad)[j==max(j)]
plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad), np.array(sensibilidad))

plt.xlabel('especificidad')
plt.ylabel('sensibilidad')
plt.scatter(p1,p2, color='red')
plt.title('curva ROC')
plt.grid(True)
plt.show()
#Calculamos el AUC
auc1 = auc(1-np.array(especificidad), np.array(sensibilidad))
print("AUC:", auc1)

#Planteamos modelo 2
#Modelo 2: Partidos jugados, Tiros totales, acciones de gol
x=np.column_stack((partidos_jugados,tiros,acciones_de_gol))
modelo2 = sm.Logit(gls, sm.add_constant(x))
resultado2 = modelo2.fit()
print(resultado2.summary())

#Armamos la tabla de predicciones para el set de test
x_test=np.column_stack((partidos_jugados_test,tiros_test,acciones_de_gol_test))
X_test=sm.add_constant(x_test)
predicciones_test = np.dot(X_test, resultado2.params)
prob_pred=np.exp(predicciones_test) / (1 + np.exp(predicciones_test))

#Calculamos el mejor punto de corte
# Generar valores de p
p_values = np.linspace(0, 1, 100)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred = 1 * (prob_pred >= p)

    a = np.sum((y_pred == 1) & (gls_test == 1))  # Verdaderos positivos
    b = np.sum((y_pred==1) & (gls_test==0))  # Falsos positivos
    c = np.sum((y_pred== 0) & (gls_test==1))  # Falsos negativos
    d = np.sum((y_pred==0)&(gls_test==0))  # Verdaderos negativos
    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (b+d)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

#Calculamos el índice de Youden y encontramos el mejor punto de corte
j = np.array(sensibilidad) + np.array(especificidad) - 1
print("Mejor punto de corte:", p_values[j==max(j)])
print("Sensibilidad:", np.array(sensibilidad)[j==max(j)])
print("Especificidad:", np.array(especificidad)[j==max(j)])

#Graficamos la curva ROC
p1=1-np.array(especificidad)[j==max(j)]
p2=np.array(sensibilidad)[j==max(j)]
plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad), np.array(sensibilidad))

plt.xlabel('especificidad')
plt.ylabel('sensibilidad')
plt.scatter(p1,p2, color='red')
plt.title('curva ROC')
plt.grid(True)
plt.show()
#Calculamos el AUC
auc2 = auc(1-np.array(especificidad), np.array(sensibilidad))
print("AUC:", auc2)

#Planteamos modelo 3
#Modelo 3: Tiros al arco, xG, Bloques cada 90 minutos
x=np.column_stack((tiros_al_arco,xG,bloques_90min))
modelo3 = sm.Logit(gls, sm.add_constant(x))
resultado3 = modelo3.fit()
print(resultado3.summary())

#Armamos la tabla de predicciones para el set de test
x_test=np.column_stack((tiros_al_arco_test,xG_test,bloques_90min_test))
X_test=sm.add_constant(x_test)
predicciones_test = np.dot(X_test, resultado3.params)
prob_pred=np.exp(predicciones_test) / (1 + np.exp(predicciones_test))

#Calculamos el mejor punto de corte
# Generar valores de p
p_values = np.linspace(0, 1, 100)

# Inicializar listas para almacenar sensibilidad y especificidad
sensibilidad = []
especificidad = []

for p in p_values:
    y_pred = 1 * (prob_pred >= p)

    a = np.sum((y_pred == 1) & (gls_test == 1))  # Verdaderos positivos
    b = np.sum((y_pred==1) & (gls_test==0))  # Falsos positivos
    c = np.sum((y_pred== 0) & (gls_test==1))  # Falsos negativos
    d = np.sum((y_pred==0)&(gls_test==0))  # Verdaderos negativos
    # Calcular sensibilidad y especificidad
    sens = a / (a+c)
    espec = d / (b+d)

    # Agregar a las listas
    sensibilidad.append(sens)
    especificidad.append(espec)

#Calculamos el índice de Youden y encontramos el mejor punto de corte
j = np.array(sensibilidad) + np.array(especificidad) - 1
print("Mejor punto de corte:", p_values[j==max(j)])
print("Sensibilidad:", np.array(sensibilidad)[j==max(j)])
print("Especificidad:", np.array(especificidad)[j==max(j)])

#Graficamos la curva ROC
p1=1-np.array(especificidad)[j==max(j)]
p2=np.array(sensibilidad)[j==max(j)]
plt.figure(figsize=(10, 5))
plt.plot(1 - np.array(especificidad), np.array(sensibilidad))

plt.xlabel('especificidad')
plt.ylabel('sensibilidad')
plt.scatter(p1,p2, color='red')
plt.title('curva ROC')
plt.grid(True)
plt.show()
#Calculamos el AUC
auc3 = auc(1 - np.array(especificidad), np.array(sensibilidad))
print("AUC:", auc3)