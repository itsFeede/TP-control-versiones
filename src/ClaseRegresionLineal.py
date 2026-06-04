import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.anova import anova_lm
class RegresionLineal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.betas = None
        self.resultado = None # Aquí guardaremos el objeto 'Results' de statsmodels



    def ajustar_modelo(self):
        """
        Agrega la columna de unos a la matriz X y ajusta el modelo 
        usando Mínimos Cuadrados Ordinarios (OLS). Guarda y devuelve las betas.
        """
        X_const = sm.add_constant(self.x)
        modelo = sm.OLS(self.y, X_const)
        self.resultado = modelo.fit() # Ejecuta el entrenamiento
        
        self.betas = self.resultado.params.tolist() # Convierte los coeficientes a lista
        return self.betas

    def test_normalidad_shapiro(self):
        """ 
        Realiza el test de Shapiro-Wilk sobre los residuos del modelo.
        Hipótesis Nula (H0): Los residuos provienen de una distribución normal.
        Si el p-valor es > 0.05, no hay evidencia para rechazar la normalidad (¡el supuesto se cumple!).
        """
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
            
        residuos = self.resultado.resid
        
        # El test devuelve dos valores: el estadístico W y el p-valor
        estadistico, p_valor = stats.shapiro(residuos)
        
        return {
            "Estadistico_W": estadistico,
            "p_valor": p_valor,
            "Cumple_Normalidad": p_valor > 0.05 # True si cumple, False si no
        }

    def test_homocedasticidad_breusch_pagan(self):
        """
        Realiza el test de Breusch-Pagan sobre los residuos del modelo.
        Hipótesis Nula (H0): La varianza de los errores es constante (Homocedasticidad).
        Si el p-valor es > 0.05, no hay evidencia para rechazar H0 (¡el supuesto se cumple!).
        """
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
            
        residuos = self.resultado.resid
        
        # El test necesita la matriz X con la constante incluida
        X_const = sm.add_constant(self.x)
        
        # El test devuelve 4 valores: LM statistic, LM p-value, F-statistic, F p-value.
        # Por convención, solemos mirar los valores del estadístico LM (Lagrange Multiplier).
        lm_stat, lm_pval, f_stat, f_pval = smd.het_breuschpagan(residuos, X_const)
        
        return {
            "Estadistico_LM": lm_stat,
            "p_valor": lm_pval,
            "Cumple_Homocedasticidad": lm_pval > 0.05 # True si cumple, False si no
        }


    def plot_residuos_vs_predichos(self):
        """
        Realiza un gráfico de los residuos en función de los valores predichos.
        
        """
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")

        
        y_hat = self.resultado.fittedvalues
        residuos = self.resultado.resid

        plt.figure(figsize=(8, 5))
        plt.scatter(y_hat, residuos, color="purple", alpha=0.7, edgecolor='white')

        
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)

        plt.title("Residuos vs. Valores Predichos")
        plt.xlabel("Valores Predichos hat_Y")
        plt.ylabel("Residuos e_i")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()




    def graficar_qqplot(self):
        """
        Genera un gráfico Q-Q manual para evaluar si los residuos del modelo 
        siguen una distribución normal (comparando cuantiles reales vs teóricos).
        """
        if self.resultado is None:
            raise ValueError("Error: El modelo debe ajustarse antes de graficar.")
      
        
        # Obtenemos residuos y los normalizamos
        residuos = self.resultado.resid
        residuos_std = (residuos - np.mean(residuos)) / np.std(residuos)
        res_ordenados = np.sort(residuos_std)
        
        # Calculamos cuantiles teóricos 
        n = len(res_ordenados)
        proporciones = np.linspace(0.5/n, 1 - 0.5/n, n)
        cuantiles_teoricos = stats.norm.ppf(proporciones)
        
        plt.scatter(cuantiles_teoricos, res_ordenados)
        plt.plot([min(cuantiles_teoricos), max(cuantiles_teoricos)], 
                 [min(cuantiles_teoricos), max(cuantiles_teoricos)], color='red')
        plt.title("Q-Q Plot de Residuos")
        plt.show()




    def obtener_estadisticas(self):
        """
        Extrae del modelo ajustado los p-valores, errores estándar e 
        intervalos de confianza. Devuelve todo organizado en un diccionario.
        """
        if self.resultado is None:
            raise ValueError( "Ajustá el modelo primero.")
        
        ic = self.resultado.conf_int()
        return {
            "p_valor": self.resultado.pvalues.tolist(),
            "error_estandar": self.resultado.bse.tolist(),
            "IC_inf": ic[0].tolist(),
            "IC_sup": ic[1].tolist()
        }




    def estimar_varianza_error(self):
 
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
        
        # statsmodels calcula esta varianza automáticamente y la guarda 
        # en el atributo mse_resid (Mean Squared Error of Residuals)
        varianza_estimada = self.resultado.mse_resid
        
        return varianza_estimada


    def obtener_residuos(self):
       
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
            
        # statsmodels guarda los residuos en el atributo 'resid'
        return self.resultado.resid



    def predecir(self, new_x):
        """
        Toma nuevos datos de X, les agrega la constante y usa las betas 
        calculadas para predecir nuevos valores de Y.
        """
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
        new_x_const = sm.add_constant(new_x, has_constant='add')
        return self.resultado.predict(new_x_const)




    def calcular_intervalos(self,x_new,nivel):
        """
        Calcula los intervalos de confianza para la media (IC) y de predicción (IP)
        utilizando get_prediction y summary_frame de statsmodels.
        """
        if self.resultado is None:
            raise ValueError("Error: El modelo debe ajustarse antes de graficar.")

        #Preparamos x_new agregando la constante 
        x_new_const = sm.add_constant(x_new, has_constant='add')

        # Obtenemos el objeto de predicción
        # alpha es el nivel de significancia (1 - confianza)
        alpha = 1 - nivel
        prediccion_obj = self.resultado.get_prediction(x_new_const)

        # Generamos el summary_frame con los intervalos calculados
        # Este DataFrame contiene columnas fijas: 'mean', 'mean_se', 
        # 'mean_ci_lower', 'mean_ci_upper', 'obs_ci_lower', 'obs_ci_upper'
        cuadro_resumen = prediccion_obj.summary_frame(alpha=alpha)

        # 5. Extraemos y devolvemos el diccionario con las listas
        return {
            "IC_media_inf": cuadro_resumen['mean_ci_lower'].values,
            "IC_media_sup": cuadro_resumen['mean_ci_upper'].values,
            "IP_inf": cuadro_resumen['obs_ci_lower'].values,
            "IP_sup": cuadro_resumen['obs_ci_upper'].values
        }




    def obtener_R2(self):
      return self.resultado.rsquared


    def comparar_modelos_anova(self, otro_modelo):
        """
        Compara el modelo actual con otro modelo usando un Test F para modelos anidados.
        Ambos deben ser instancias de las clases RegresionLineal (Simple o Multiple).
        """
        if self.resultado is None:
            raise ValueError("El modelo actual no está ajustado. Usá ajustar_modelo() primero.")
        if otro_modelo.resultado is None:
            raise ValueError("El modelo a comparar no está ajustado. Usá ajustar_modelo() primero.")

        # Realizamos la comparación usando anova_lm de statsmodels
        tabla_comparacion = anova_lm(self.resultado, otro_modelo.resultado)
        
        return tabla_comparacion
    
    
class RegresionLinealSimple(RegresionLineal):
    def __init__(self, x, y):
        """
        Hereda el constructor del padre. Se usa específicamente cuando hay 
        una sola variable predictora.
        """
        super().__init__(x, y)

    def predecir(self, new_x):
        """
        Llama al método predecir de la clase base usando super().
        """
        return super().predecir(new_x)



    def graficar_recta_ajustada(self):
        """
        Dibuja el diagrama de dispersión y la recta de regresión
        usando exclusivamente matplotlib de forma manual.
        
        """
        
        if self.resultado is None:
            raise ValueError("Ajustá el modelo primero.")
            
        # 2. Crear la figura
        plt.figure(figsize=(8, 6))
        
        #Graficar los puntos 
        plt.scatter(self.x, self.y, color='steelblue', s=40, alpha=0.7, 
                    edgecolor='white', label="Datos observados")
        
        
        x_ordenado = np.sort(self.x)
        X_ordenado_const = sm.add_constant(x_ordenado)
        
        #Calcular la predicción sobre esos puntos ordenados
        y_pred = self.resultado.predict(X_ordenado_const)
        
        #Graficar la línea de regresión
        plt.plot(x_ordenado, y_pred, color='firebrick', linewidth=2.5, 
                 label="Recta ajustada (Modelo)")
        
        
        plt.title("Ajuste de Regresión Lineal Simple", fontsize=12)
        plt.xlabel("Variable Predictora (X)")
        plt.ylabel("Variable Respuesta (Y)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.show()




class RegresionLinealMultiple(RegresionLineal):
    def __init__(self, x, y):
        """
        Hereda el constructor. Se usa cuando X es una matriz (varias variables).
        """
        super().__init__(x, y)



    def predecir(self, new_x):
        """
        Al igual que en la simple, delega la lógica de predicción a la 
        clase padre.
        """
        return super().predecir(new_x)



    def obtener_R2_ajustado(self):
        return self.resultado.rsquared_adj
    
    
    