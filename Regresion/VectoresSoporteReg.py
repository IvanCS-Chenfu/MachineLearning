import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#########################
### CREAMOS LOS DATOS ###
#########################

# (muestras, características de las que depende el target, ruido, semilla de datos, ordenada en el origen)
X, Y = make_regression(n_samples=500, n_features=1, noise=50, random_state=42, bias = 100)

# Dibujamos la relación
plt.scatter(X, Y)
plt.show()


########################################
### VECTORES DE SOPORTE DE REGRESIÓN ###
########################################
# Es igual que la Regresión Lineal/Polinomial pero al entrenar obviamos los datos que están muy alejados (errores)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


from sklearn.svm import SVR

# Creamos el Objeto de los Vectores de Soporte de Regresión
# (Polinomio, Cuánto afectan los errores a nuestra ecuación, Máximo error que va a afectar)
svr = SVR(kernel = 'linear', C = 1.0, epsilon=0.2)

# Entrenamos la regresión lineal
svr.fit(X_train,Y_train)

# Una vez tenida la regresión lineal entrenada, realizamos la predicción de "Y_test" utilizando "X_test"
Y_pred = svr.predict(X_test)

# Comparamos lo predicho con los datos reales.
plt.scatter(X_test, Y_test)
plt.plot(X_test,Y_pred, color = 'red')
plt.show()

# Valor de la pendiente
print(svr.coef_)

# Valor de la ordenada en el origen
print(svr.intercept_)

# Precisión del Modelo según R^2
print(svr.score(X_train,Y_train))