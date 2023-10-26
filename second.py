import numpy as np
from sympy import symbols, diff, sympify, lambdify

# Funcion para recorrer los puntos que no son puntos de frontera de la malla para las velocidades en X.
# function : La ecuacion que se evaluara en cada punto de la malla.
# nx : Numero de puntos de la malla en x (Se resta 1 para no tomar los valores de frontera de los bordes).
# ny : Numero de puntos de la malla en y (Se resta 1 para no tomar los valores de frontera de los bordes).
# h : Distancia entre puntos de la malla.
# npoints : Numero de puntos de la malla que no son de frontera.
def travelMatrixVx(function, nx, ny, h, npoints):
    i = 0
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (Vx[y, x] == -1):
                i = i + 1
                row = np.full(9, 0, dtype=object)
                function(x, y, h, row, i)
                Avx.append(row)

# Funcion para recorrer los puntos que no son puntos de frontera de la malla para las velocidades en Y.
# function : La ecuacion que se evaluara en cada punto de la malla.
# nx : Numero de puntos de la malla en x (Se resta 1 para no tomar los valores de frontera de los bordes).
# ny : Numero de puntos de la malla en y (Se resta 1 para no tomar los valores de frontera de los bordes).
# h : Distancia entre puntos de la malla.
# npoints : Numero de puntos de la malla que no son de frontera.
def travelMatrixVy(function, nx, ny, h, npoints):
    i = 0
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (Vx[y, x] == -1):
                i = i + 1
                row = np.full(npoints, 0, dtype=object)
                function(x, y, h, row, i)
                Avy.append(row)

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# el string que concatena el coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de
# frontera es diferente de 0, se agrega al vector de resultados de la ecuacion.
# se agrega al vector de resultados de la ecuacion.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
# pos : Posicion en la fila de la matriz de coeficientes de la ecuacion.
def discretizarVx(coeficiente, x, y, row, i, pos):
    if Vx[y, x] == -1:
        row[pos] = str(coeficiente) + "*" + "Vx" + str(IDvx[y - 1, x - 1]);
    else:
        if Vx[y, x] != 0:
            bvx[i - 1] = bvx[i - 1] + (-1 * (coeficiente * Vx[y, x]))

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# el string que concatena el coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de
# frontera es diferente de 0, se agrega al vector de resultados de la ecuacion. Se diferencia de la funcion
# discretizarVx en que tambien evalua un segundo punto centrado.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# xNoLineal : Coordenada x del punto no lineal.
# yNoLineal : Coordenada y del punto no lineal.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
# pos : Posicion en la fila de la matriz de coeficientes de la ecuacion.
def discretizarVxNoLineal(coeficiente, xNoLineal, yNoLineal, x, y, row, i, pos):
    variableNoLineal = None;

    if Vx[yNoLineal, xNoLineal] == -1:
        variableNoLineal = "Vx" + str(IDvx[yNoLineal - 1, xNoLineal - 1]);
    else:
        if Vx[yNoLineal, xNoLineal] != 0:
            variableNoLineal = Vx[yNoLineal, xNoLineal] * coeficiente;
        else:
            variableNoLineal = 0;

    if Vx[y, x] == -1:
        if variableNoLineal.isnumeric() and variableNoLineal != 0:
            row[pos] = str(variableNoLineal) + "*" + "Vx" + str(IDvx[y - 1, x - 1]);
        elif variableNoLineal == 0:
            row[pos] = 0;
        else:
            row[pos] = str(coeficiente) + "*" + str(variableNoLineal) + "*" + "Vx" + str(IDvx[y - 1, x - 1]);
    else:
        if Vx[y, x] != 0:
            if variableNoLineal.isnumeric() and variableNoLineal != 0:
                bvx[i - 1] = bvx[i - 1] + (-1 * (variableNoLineal * Vx[y, x]))
            elif variableNoLineal == 0:
                row[pos] = 0;
            else:
                row[pos] = str(Vx[y, x]) + "*" + variableNoLineal;
        else:
            row[pos] = 0;

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de frontera es diferente de 0,
# se agrega al vector de resultados de la ecuacion.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
# pos : Posicion en la fila de la matriz de coeficientes de la ecuacion.
# Funcion para discretizar los puntos de la malla.
def discretizarVy(coeficiente, x, y, row, i, pos):
    if Vy[y, x] == -1:
        row[pos] = str(coeficiente) + "*" + "Vy" + str(IDvy[y - 1, x - 1]);
    else:
        if Vy[y, x] != 0:
            bvy[i - 1] = bvy[i - 1] + (-1 * (coeficiente * Vy[y, x]))

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# el string que concatena el coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de
# frontera es diferente de 0, se agrega al vector de resultados de la ecuacion. Se diferencia de la funcion
# discretizarVy en que tambien evalua un segundo punto centrado.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# xNoLineal : Coordenada x del punto no lineal.
# yNoLineal : Coordenada y del punto no lineal.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
# pos : Posicion en la fila de la matriz de coeficientes de la ecuacion.
def discretizarVyNoLineal(coeficiente, xNoLineal, yNoLineal, x, y, row, i, pos):
    variableNoLineal = None;

    if Vy[yNoLineal, xNoLineal] == -1:
        variableNoLineal = "Vy" + str(IDvy[yNoLineal - 1, xNoLineal - 1]);
    else:
        if Vy[yNoLineal, xNoLineal] != 0:
            variableNoLineal = Vy[yNoLineal, xNoLineal] * coeficiente;
        else:
            variableNoLineal = 0;

    if Vy[y, x] == -1:
        if variableNoLineal.isnumeric() and variableNoLineal != 0:
            row[pos] = str(variableNoLineal) + "*" + "Vy" + str(IDvy[y - 1, x - 1]);
        elif variableNoLineal == 0:
            row[pos] = 0;
        else:
            row[pos] = str(coeficiente) + "*" + str(variableNoLineal) + "*" + "Vy" + str(IDvy[y - 1, x - 1]);
    else:
        if Vy[y, x] != 0:
            if variableNoLineal.isnumeric() and variableNoLineal != 0:
                bvy[i - 1] = bvy[i - 1] + (-1 * (variableNoLineal * Vy[y, x]))
            elif variableNoLineal == 0:
                row[pos] = 0;
            else:
                row[pos] = str(Vy[y, x]) + "*" + variableNoLineal;
        else:
            row[pos] = 0;

# Funcion que representa la ecuacion de Navier Stokes para la velocidad en X previamente simplificada.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# h : Distancia entre puntos de la malla.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
def navierStokesSimplifyVx(x, y, h, row, i):
    discretizarVx(2, x - h, y, row, i, 0)
    discretizarVx(2, x + h, y, row, i, 1)
    discretizarVx(2, x, y - h, row, i, 2)
    discretizarVx(2, x, y + h, row, i, 3)
    discretizarVx(-8, x, y, row, i, 4)
    discretizarVxNoLineal(-1, x, y, x + h, y, row, i, 5)
    discretizarVxNoLineal(1, x, y, x - h, y, row, i, 6)
    discretizarVxNoLineal(-1, x, y, x, y + h, row, i, 7)
    discretizarVxNoLineal(1, x, y, x, y - h, row, i, 8)

# Funcion que representa la ecuacion de Navier Stokes para la velocidad en Y previamente simplificada.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# h : Distancia entre puntos de la malla.
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
def navierStokesSimplifyVy(x, y, h, row, i):
    discretizarVy(2, x - h, y, row, i,0)
    discretizarVy(2, x + h, y, row, i,1)
    discretizarVy(2, x, y - h, row, i,2)
    discretizarVy(2, x, y + h, row, i,3)
    discretizarVy(-8, x, y, row, i,4)
    discretizarVyNoLineal(-1, x, y, x + h, y, row, i,5)
    discretizarVyNoLineal(1, x, y, x - h, y, row, i,6)
    discretizarVyNoLineal(-1, x, y, x, y + h, row, i,7)
    discretizarVyNoLineal(1, x, y, x, y - h, row, i,9)

# MEDIDAS Y RESTRICCIONES DES LAS MALLES.
# Ancho de la malla.
anchoMalla = 20
# Altura de la malla.
alturaMalla = 10
# Numero de puntos de la malla en x.
nx = 21
# Numero de puntos de la malla en y.
ny = 11
# Numero de puntos de la columnas en x.
nxc = 5
# Numero de puntos de la columnas en y.
nyc = 3

# Matriz de la velocidades en X.
Vx = np.full((ny, nx), -1)
# Restriccion de frontera pared derecha.
Vx[0, :] = 0
# Restriccion de frontera pared superior.
Vx[-1, :] = 0
# Restriccion de frontera pared inferior.
Vx[:, -1] = 0
# Restriccion de frontera pared izquierda.
Vx[:, 0] = 5

# Matriz de la velocidades en y.
Vy = Vx.copy()
# Restriccion de frontera pared izquierda.
Vy[:, 0] = 0

# Restriccion columna 1.
for i in range(1, nyc + 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

# Restriccion columna 2
for i in range(Vx.shape[0] - (nyc + 1), Vx.shape[0] - 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

# MATRICES PARA EL SISTEMA DE ECUACIONES.
## Matrices para Vx
# Matriz de los coeficientes de la ecuacion.
Avx = []
# Vector de los resultados de la ecuacion.
bvx = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Matriz de los identificadores de los puntos de la malla.
IDvx = np.empty((ny - 2, nx - 2), dtype=int)
# Vector de las variables de la ecuacion.
Xvx = ["Vx" + str(i) for i in range(1, ((ny - 2) * (nx - 2)) - 2 * (nxc * nyc) + 1)]
# Se asigna un valor a cada punto de la malla que no es de frontera.
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

## Matrices para Vy
# Matriz de los coeficientes de la ecuacion.
Avy = []
# Vector de los resultados de la ecuacion.
bvy = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvy = ["Vy" + str(i) for i in range(1, ((ny - 2) * (nx - 2)) - 2 * (nxc * nyc) + 1)]
# Matriz de los identificadores de los puntos de la malla.
IDvy = IDvx

# RESULTADOS MALLA.
travelMatrixVx(navierStokesSimplifyVx, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))
travelMatrixVy(navierStokesSimplifyVy, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

# RESOLUCION DEL SISTEMA DE ECUACIONES.

# Funcion para generar las ecuaciones simbolicas.
# ecuaciones : Lista de ecuaciones simbolicas.
def generarEcuaciones(ecuaciones):

    tamaño = len(Avx)
    for i in range(0, tamaño):
        ecuacion = ""
        for j in range(0, 9):
            if Avx[i][j] != 0:
                ecuacion = ecuacion + str(Avx[i][j]) + " + "
        if bvx[i] != 0:
            ecuacion = ecuacion + str(-bvx[i])
        else:
            ecuacion = ecuacion + "0"

        ecuacion_simbolica = sympify(ecuacion)
        ecuaciones.append(ecuacion_simbolica)

# Funcion para aplicar el metodo de Newton Rapshon.
# initialValue : Valor inicial de las variables.
# tolerance : Tolerancia del metodo.
# maxIterations : Numero maximo de iteraciones.
# ecuaciones : Lista de ecuaciones simbolicas.
def newtonRapshon(initialValue, tolerance, maxIterations, ecuaciones):

    x = initialValue
    jacobiano = evaluarJacobianoNp(ecuaciones, variablesvX)

    for _ in range(maxIterations):
        deltaX = evaluarMatrizInversaNp(jacobiano, variablesvX, x) @ evaluarEcuaciones(ecuaciones, variablesvX, x)
        xNueva = x - deltaX
        if abs(np.linalg.norm(xNueva) - np.linalg.norm(x)) < tolerance:
            return x
        else:
            x = xNueva
    return x

# Funcion para evaluar las ecuaciones simbolicas.
# ecuaciones : Lista de ecuaciones simbolicas.
# variables : Lista de variables simbolicas.
# valoresIniciales : Lista de valores iniciales de las variables.
def evaluarEcuaciones(ecuaciones, variables, valoresIniciales):

    tamaño = len(variables)
    vectorEvaluado = np.zeros(tamaño)

    for i in range(0, tamaño):
        vectorEvaluado[i] = lambdify(variables, ecuaciones[i])(*valoresIniciales)

    return vectorEvaluado

# Funcion para evaluar el jacobiano de las ecuaciones simbolicas.
# ecuaciones : Lista de ecuaciones simbolicas.
# variables : Lista de variables simbolicas.
# valoresIniciales : Lista de valores iniciales de las variables.
def evaluarJacobianoNp(ecuaciones, variables):

    tamaño = len(variables)
    jacobiano = np.zeros((tamaño, tamaño), dtype=object)

    for i in range(0, tamaño):
        for j in range(0, tamaño):
            jacobiano[i][j] = diff(ecuaciones[i], variables[j])

    return jacobiano

# Funcion para evaluar la matriz inversa del jacobiano de las ecuaciones simbolicas.
# jacobiano : Matriz jacobiana de las ecuaciones simbolicas.
# variables : Lista de variables simbolicas.
# valoresIniciales : Lista de valores iniciales de las variables.
def evaluarMatrizInversaNp(jacobiano, variables, valoresIniciales):

    tamaño = len(variables)
    matriz = np.zeros((tamaño, tamaño))

    for i in range(0, tamaño):
        for j in range(0, tamaño):
            matriz[i][j] = lambdify(variables, jacobiano[i][j])(*valoresIniciales)
    matrizInversa = np.linalg.inv(matriz)

    return matrizInversa

# ECUACIONES SIMBOLICAS.
variablesvX = symbols(Xvx)
variablesVy = symbols(Xvy)

# Lista de ecuaciones simbolicas.
ecuacionesVx = []
ecuacionesVy = []

# Valor inicial de las variables.
initialValue = np.full(len(variablesvX), 1, dtype=int)

# Se generan las ecuaciones simbolicas.
generarEcuaciones(ecuacionesVx)
generarEcuaciones(ecuacionesVy)

# Se resuelve el sistema de ecuaciones.
print(newtonRapshon(initialValue, 0.1, 1, ecuacionesVx))
print(newtonRapshon(initialValue, 0.1, 1, ecuacionesVy))

