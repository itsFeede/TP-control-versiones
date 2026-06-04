import statsmodels.api as sm
from pathlib import Path
import sys
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#---#
RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

CargaDatos = import_module("src.carga_datos").CargaDatos

class regresion_lineal:
    def __init__(self, base_datos):
        # Permite recibir directamente el DataFrame o un objeto cargador.
        if hasattr(base_datos, "base_datos"):
            self.base_datos = base_datos.base_datos
        elif hasattr(base_datos, "datos"):
            self.base_datos = base_datos.datos
        else:
            self.base_datos = base_datos

    def regresion_lineal_simple(self):
        #vamos a tomaar como variable predictora a goles
        #elegimos la variable tiros al arco.
        self.y_1 = self.base_datos["Gls"] 
        self.x_1 = self.base_datos["SoT"]
        self.X_1 = sm.add_constant(self.x_1)
        self.modelo_1 = sm.OLS(self.y_1,self.X_1).fit()
        return self.modelo_1.summary()
    
    def regresion_lineal_simple2(self):
        self.x_2=self.base_datos["xG"]
        self.y_2=self.base_datos["Gls"]
        self.X_2=sm.add_constant(self.x_2)
        self.modelo_2=sm.OLS(self.y_2,self.X_2).fit()
        return self.modelo_2.summary()

    def qqplot_residuos_modelo_1(self):
        if not hasattr(self, "modelo_1"):
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple().")

        residuos = self.modelo_1.resid
        residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)
        res_ordenados = np.sort(residuos_std)

        n = len(res_ordenados)
        proporciones = np.linspace(0.5 / n, 1 - 0.5 / n, n)
        cuantiles_teoricos = stats.norm.ppf(proporciones)

        plt.figure(figsize=(8, 6))
        plt.scatter(cuantiles_teoricos, res_ordenados)
        mn = min(cuantiles_teoricos)
        mx = max(cuantiles_teoricos)
        plt.plot([mn, mx], [mn, mx], color="red")
        plt.title("Q-Q Plot de Residuos (Modelo 1)")
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Residuos estandarizados (ordenados)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def grafico_ajuste_modelo_1(self):
        if not hasattr(self, "modelo_1"):
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple().")

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.x_1,
            self.y_1,
            color="steelblue",
            s=40,
            alpha=0.7,
            edgecolor="white",
            label="Datos observados",
        )

        x_ordenado = np.sort(self.x_1)
        X_ordenado_const = sm.add_constant(x_ordenado)
        y_pred = self.modelo_1.predict(X_ordenado_const)

        plt.plot(
            x_ordenado,
            y_pred,
            color="firebrick",
            linewidth=2.5,
            label="Recta ajustada (Modelo)",
        )

        plt.title("Ajuste de Regresión Lineal Simple (Modelo 1)", fontsize=12)
        plt.xlabel("Tiros al arco (SoT)")
        plt.ylabel("Goles (Gls)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()

    def qqplot_residuos_modelo_2(self):
        if not hasattr(self, "modelo_2"):
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple2().")

        residuos = self.modelo_2.resid
        residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)
        res_ordenados = np.sort(residuos_std)

        n = len(res_ordenados)
        proporciones = np.linspace(0.5 / n, 1 - 0.5 / n, n)
        cuantiles_teoricos = stats.norm.ppf(proporciones)

        plt.figure(figsize=(8, 6))
        plt.scatter(cuantiles_teoricos, res_ordenados)
        mn = min(cuantiles_teoricos)
        mx = max(cuantiles_teoricos)
        plt.plot([mn, mx], [mn, mx], color="red")
        plt.title("Q-Q Plot de Residuos (Modelo 2)")
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Residuos estandarizados (ordenados)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def grafico_ajuste_modelo_2(self):
        if not hasattr(self, "modelo_2"):
            raise ValueError("Primero debe ajustar el modelo con regresion_lineal_simple2().")

        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.x_2,
            self.y_2,
            color="steelblue",
            s=40,
            alpha=0.7,
            edgecolor="white",
            label="Datos observados",
        )

        x_ordenado = np.sort(self.x_2)
        X_ordenado_const = sm.add_constant(x_ordenado)
        y_pred = self.modelo_2.predict(X_ordenado_const)

        plt.plot(
            x_ordenado,
            y_pred,
            color="firebrick",
            linewidth=2.5,
            label="Recta ajustada (Modelo)",
        )

        plt.title("Ajuste de Regresión Lineal Simple (Modelo 2)", fontsize=12)
        plt.xlabel("Goles esperados (xG)")
        plt.ylabel("Goles (Gls)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()
    
    def grafico_scatter1 (self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.x_1, self.y_1, color="blue", alpha=0.6, label="Datos reales")

        # Línea de regresión
        x_ordenado = np.sort(self.x_1)
        X_ordenado_const = sm.add_constant(x_ordenado)
        plt.plot(x_ordenado, self.modelo_1.predict(X_ordenado_const), color="red", label="Recta ajustada")

        plt.xlabel("Tiros al arco (SoT)")
        plt.ylabel("Goles (Gls)")
        plt.title("Regresión lineal simple: Gls vs SoT")
        plt.legend()
        plt.show()

    def grafico_scatter2 (self):
        plt.figure(figsize=(8,6))
        plt.scatter(self.x_2, self.y_2, color="blue", alpha=0.6, label="Datos reales")

        # Línea de regresión
        plt.plot(self.x_2, self.modelo_2.predict(self.X_2), color="red", label="Recta ajustada")

        plt.xlabel("Goles esperados (xG)")
        plt.ylabel("Goles (Gls)")
        plt.title("Regresión lineal simple: Gls vs xG")
        plt.legend()
        plt.show()

        



ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
cargador = CargaDatos(ruta_datos)
cargador.cargar_datos()
modelado = regresion_lineal(cargador)
print(modelado.regresion_lineal_simple())
modelado.grafico_ajuste_modelo_1()
modelado.qqplot_residuos_modelo_1()

print(modelado.regresion_lineal_simple2())
modelado.grafico_ajuste_modelo_2()
modelado.qqplot_residuos_modelo_2()