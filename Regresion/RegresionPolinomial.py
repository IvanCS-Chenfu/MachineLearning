import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

########################
### CARGAR LOS DATOS ###
########################

boston = datasets.load_diabetes()

# Da toda la infomración en bruto (no nos sirve)
#print(boston)
print()

# Da todos los grupos en los que se agrupa la información
#print(boston.keys())
print()

# Este grupo describe qué es la variable "target" y de cuántas cosas depende
# En este caso "target" es la capacidad de tener diabetes y depende del sexo, la edad, la masa...
#print(boston.DESCR)
print()

# Dice la cantidad de datos que tenemos
#print(boston.data.shape)
print()

# Los nombres de las variables de las que depende el "target"
#print(boston.feature_names)
print()



###################################
### REGRESIÓN POLINOMIAL SIMPLE ###
###################################

# Elegimos la columna "2" de las que depende el "target". En este caso, el índice de masa.
X = boston.data[:, np.newaxis, 6]
Y = boston.target

# Dibujamos la relación
plt.scatter(X, Y)
plt.xlabel('Índice de Masa')
plt.ylabel('Medida de la Enfermedad')
plt.show()


from sklearn.model_selection import train_test_split

# De todos los datos de X, cogemos de forma aleatoria un 20% y se lo damos a la variable "X_test".
# El 80% restante se le asigna a "X_train". Lo mismo se realiza con "Y"
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


from sklearn.preprocessing import PolynomialFeatures

# Ajustamos el grado de nuestra regresión polinomial
poli_reg = PolynomialFeatures(degree = 2) # Un grado demasiado grande puede ocasionar un sobreajuste

# Transformamos las variables de entrenamiento a un mayor grado. (X pasan a X, X^2 ; X1 y X2 pasarían a X1, X2, X1^2, X2^2, X1X2)
X_train_poli = poli_reg.fit_transform(X_train)
X_test_poli = poli_reg.fit_transform(X_test)

# Creamos el Objeto de la Regresión Polinomial
pr = linear_model.LinearRegression()

# Entrenamos la regresión polinomial
pr.fit(X_train_poli,Y_train)

# Una vez tenida la regresión lineal entrenada, realizamos la predicción de "Y_test" utilizando "X_test"
Y_pred = pr.predict(X_test_poli)

# Comparamos lo predicho con los datos reales.
plt.scatter(X_test, Y_test)
plt.plot(X_test,Y_pred, color = 'red')
plt.title('Regresión Polinomial Simple')
plt.xlabel('Índice de Masa')
plt.ylabel('Medida de la Enfermedad')
plt.show()

# Valor de la pendiente
print(pr.coef_)

# Valor de la ordenada en el origen
print(pr.intercept_)

# Precisión del Modelo según R^2
print(pr.score(X_train_poli,Y_train))
