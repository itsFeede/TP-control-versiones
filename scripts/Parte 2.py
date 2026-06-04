import statsmodels.api as sm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import sys 

sys.path.append(r"C:\Users\sanch\OneDrive\Desktop\Tp Jugadores\TP-control-versiones\src")


RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))
from src.ClaseRegresionLineal import RegresionLineal
from src.ClaseRegresionLineal import RegresionLinealMultiple
from src.carga_datos import CargaDatos
ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)

cargador.cargar_datos()
cargador.mostrar_dimensiones()

jugadores=cargador.datos
print(jugadores.head())

gls = jugadores["Gls"]
edad = jugadores["Age"] 
partidos_jugados = jugadores["MP"] 
min=jugadores["Min"] 
tiros=jugadores["Sh"] 
tiros_al_arco=jugadores["SoT"] 
xG=jugadores["xG"] 
bloques_90min=jugadores["90s"] 
tiros_90min=jugadores["Sh/90"]  
acciones_de_gol=jugadores["SCA"] 
conducciones_progresivas=jugadores["PrgC"] 

#Planteamos modelo 1
print(jugadores.isnull().sum())
#Sacamas los nulos de edad:
jugadores_sin_nulos = jugadores.dropna(subset=["Age"])
tiros_90min_sin_nulos = jugadores_sin_nulos["Sh/90"]
conducciones_progresivas_sin_nulos = jugadores_sin_nulos["PrgC"]
min_sin_nulos = jugadores_sin_nulos["Min"]
edad_sin_nulos = jugadores_sin_nulos["Age"]
gls_sin_nulos = jugadores_sin_nulos["Gls"]

#Modelo 1: Edad, Minutos, Tiros cada 90 minutos y Conducciones Progresivas
x=np.column_stack((edad_sin_nulos,min_sin_nulos,tiros_90min_sin_nulos,conducciones_progresivas_sin_nulos))
modelo1 = RegresionLinealMultiple(x, gls_sin_nulos)
modelo1.ajustar_modelo()
print(modelo1.resultado.summary())

modelo1.plot_residuos_vs_predichos()
modelo1.graficar_qqplot()

#Planteamos modelo 2
#Modelo 2: Partidos jugados, Tiros totales, acciones de gol

x=np.column_stack((partidos_jugados,tiros,acciones_de_gol))
modelo2 = RegresionLinealMultiple(x, gls)
modelo2.ajustar_modelo()
print(modelo2.resultado.summary())

modelo2.plot_residuos_vs_predichos()
modelo2.graficar_qqplot()

#Planteamos modelo 3
#Modelo 3: Tiros al arco, xG, Bloques cada 90 minutos

x=np.column_stack((tiros_al_arco,xG,bloques_90min))
modelo3 = RegresionLinealMultiple(x, gls)
modelo3.ajustar_modelo()
print(modelo3.resultado.summary())

modelo3.plot_residuos_vs_predichos()
modelo3.graficar_qqplot()

