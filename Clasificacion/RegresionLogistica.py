import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

########################
### CARGAR LOS DATOS ###
########################

datos = datasets.load_breast_cancer()

# Da toda la infomración en bruto (no nos sirve)
#print(datos)
print()

# Da todos los grupos en los que se agrupa la información
#print(datos.keys())
print()

# Este grupo describe qué es la variable "target" y de cuántas cosas depende
# En este caso "target" es la capacidad de tener diabetes y depende del sexo, la edad, la masa...
#print(datos.DESCR)
print()

# Dice la cantidad de datos que tenemos
print(datos.data.shape)
print()

# Los nombres de las variables de las que depende el "target"
#print(datos.feature_names)
print()



###########################
### REGRESIÓN LOGISTICA ###
###########################

# Elegimos la columna "2" de las que depende el "target". En este caso, el índice de masa.
X = datos.data
Y = datos.target

from sklearn.model_selection import train_test_split

# De todos los datos de X, cogemos de forma aleatoria un 20% y se lo damos a la variable "X_test".
# El 80% restante se le asigna a "X_train". Lo mismo se realiza con "Y"
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

# Escalamos los datos ya que las magnitudes de cada característica son diferentes 
# Por ejemplo: (la glucosa en sangre tiene una magnitud de 1 mientras que el peso una magnitud de 1000)

from sklearn.preprocessing import StandardScaler

escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)


from sklearn.linear_model import LogisticRegression

algoritmo = LogisticRegression()
algoritmo.fit(X_train, Y_train)

Y_pred = algoritmo.predict(X_test)

# Realizo la matriz de confusión para comprobar los resultados (cuantos falsos positivos y negativos hay)
from sklearn.metrics import confusion_matrix

# VP FN
# FP VN
matriz = confusion_matrix(Y_test, Y_pred)
print(matriz)


# Observamos lo bueno que es nuestro sistema
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

precision = precision_score(Y_test, Y_pred)
print(precision)

exactitud = accuracy_score(Y_test, Y_pred)
print(exactitud)

sensibilidad = recall_score(Y_test, Y_pred)
print(sensibilidad)

Puntuaje_F1 = f1_score(Y_test, Y_pred)
print(Puntuaje_F1)

roc_aux = roc_auc_score(Y_test, Y_pred)
print(roc_aux)