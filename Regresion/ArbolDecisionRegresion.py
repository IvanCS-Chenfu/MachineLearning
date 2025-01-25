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


from sklearn.tree import DecisionTreeRegressor

# Creamos el Objeto de los Vectores de Soporte de Regresión
# (numero de clases/ramificaciones) evita sobreajustes
tree = DecisionTreeRegressor(max_depth = 5)

# Entrenamos la regresión lineal
tree.fit(X_train,Y_train)

# Una vez tenida la regresión lineal entrenada, realizamos la predicción de "Y_test" utilizando "X_test"
Y_pred = tree.predict(X_test)

# Comparamos lo predicho con los datos reales.
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_test, Y_test)
plt.plot(X_grid,tree.predict(X_grid), color = 'red')
plt.show()

# Precisión del Modelo según R^2
print(tree.score(X_train,Y_train))