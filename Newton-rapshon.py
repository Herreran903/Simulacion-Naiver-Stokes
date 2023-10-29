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
    discretizarVyNoLineal(1, x, y, x, y - h, row, i,8)

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
def generarEcuaciones(A, b, ecuaciones):
    tamaño = len(A)
    for i in range(0, tamaño):
        ecuacion = ""
        for j in range(0, 9):
            if A[i][j] != 0:
                ecuacion = ecuacion + str(A[i][j]) + " + "
        if b[i] != 0:
            ecuacion = ecuacion + str(-b[i])
        else:
            ecuacion = ecuacion + "0"
        ecuacion_simbolica = sympify(ecuacion)
        ecuaciones.append(ecuacion_simbolica)

# Funcion para aplicar el metodo de Newton Rapshon.
# initialValue : Valor inicial de las variables.
# tolerance : Tolerancia del metodo.
# maxIterations : Numero maximo de iteraciones.
# ecuaciones : Lista de ecuaciones simbolicas.
def newtonRapshon(initialValue, tolerance, maxIterations, ecuaciones, variables):
    x = initialValue
    jacobiano = evaluarJacobianoNp(ecuaciones, variables)
    num = 0
    for _ in range(maxIterations):
        num += 1
        deltaX = evaluarMatrizInversaNp(jacobiano, variables, x) @ evaluarEcuaciones(ecuaciones, variables, x)
        xNueva = x - deltaX
        if abs(np.linalg.norm(xNueva) - np.linalg.norm(x)) < tolerance:
            return [num, xNueva]
        else:
            x = xNueva
    return [num, xNueva]

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
variablesVx = symbols(Xvx)
variablesVy = symbols(Xvy)

# Lista de ecuaciones simbolicas.
ecuacionesVx = []
ecuacionesVy = []

# Valor inicial de las variables.
initialValue = np.full(len(variablesVx), 0, dtype=int)
initialValueVx = [2.49998758e+00, 1.34669132e+00, 7.73744912e-01, 4.68535777e-01,
                2.95095021e-01, 1.88755238e-01, 1.09382311e-01, 1.37785071e-04,
                2.41919248e-04, 2.70391003e-04, 2.49607178e-04, 1.95860999e-04,
                1.41590332e-04, 8.23670857e-05, 3.65322341e+00, 2.49984851e+00,
                1.68145155e+00, 1.13206534e+00, 7.66705648e-01, 5.15610734e-01,
                3.09187764e-01, 9.03258685e-04, 1.31263006e-03, 1.32012930e-03,
                1.09267967e-03, 8.27375353e-04, 5.54827115e-04, 3.25266577e-04,
                4.22600228e+00, 3.31773913e+00, 2.49895657e+00, 1.84047923e+00,
                1.33713200e+00, 9.50681674e-01, 5.99287030e-01, 5.63680890e-03,
                6.15763086e-03, 5.07659920e-03, 3.79510446e-03, 2.58509546e-03,
                1.68690974e-03, 9.22804944e-04, 4.53071607e+00, 3.86560175e+00,
                3.15437749e+00, 2.49448255e+00, 1.92720302e+00, 1.44984754e+00,
                1.01803879e+00, 5.39211628e-01, 2.99098211e-01, 1.70612947e-01,
                9.92213139e-02, 5.84108264e-02, 3.71926556e-02, 2.42894216e-02,
                1.61040081e-02, 1.04200614e-02, 6.70587635e-03, 4.05270848e-03,
                2.21916832e-03, 4.70224389e+00, 4.22575015e+00, 3.64776299e+00,
                3.04700165e+00, 2.47574354e+00, 1.95390532e+00, 1.46243187e+00,
                9.67424262e-01, 6.08255990e-01, 3.73985592e-01, 2.26561835e-01,
                1.36961275e-01, 8.39588079e-02, 5.28099073e-02, 3.32554998e-02,
                2.11719460e-02, 1.30500460e-02, 7.95847450e-03, 4.18266317e-03,
                4.80042719e+00, 4.45541124e+00, 3.99743823e+00, 3.47659430e+00,
                2.93793771e+00, 2.39935049e+00, 1.83129714e+00, 1.13538545e+00,
                7.07091234e-01, 4.36839599e-01, 2.68080678e-01, 1.63527904e-01,
                1.06630017e-01, 7.06258110e-02, 4.72210801e-02, 3.06975871e-02,
                1.98118954e-02, 1.19897665e-02, 6.57035465e-03, 4.84163302e+00,
                4.56922749e+00, 4.19599449e+00, 3.74969309e+00, 3.25528096e+00,
                2.70725101e+00, 1.94345481e+00, 4.84899687e-02, 5.33795794e-02,
                4.42402655e-02, 3.32012742e-02, 2.26639141e-02, 1.48102969e-02,
                8.10430891e-03, 4.76340585e+00, 4.46835249e+00, 4.12443119e+00,
                3.73161226e+00, 3.29110652e+00, 2.76677082e+00, 1.95904740e+00,
                2.33065041e-02, 3.40096628e-02, 3.43270269e-02, 2.84730878e-02,
                2.15949852e-02, 1.44861478e-02, 8.49372574e-03, 4.11463828e+00,
                3.62782164e+00, 3.27545456e+00, 2.95361837e+00, 2.61563432e+00,
                2.21133081e+00, 1.56059376e+00, 1.06533787e-02, 1.87590433e-02,
                2.10060563e-02, 1.94248181e-02, 1.52489872e-02, 1.10263559e-02,
                6.40993185e-03]

# initialValueVy = [0.02425,  0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.0495,   0.02425,
#                   0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.0495,   0.062125, 0.1,
#                   0.1,      0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,
#                   0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,
#                   0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,
#                   0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,
#                   0.1,      0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.1,      0.1,
#                   0.1,      0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,
#                   0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,
#                   0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
#                   0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
#                   0.087375, 0.087375, 0.087375, 0.087375, 0.1,      0.1,      0.1,      0.1,
#                   0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,
#                   0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,
#                   0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
#                   0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.087375, 0.0495,
#                   0.087375, 0.087375, 0.087375, 0.087375, 0.087375, 0.07475,  0.0495,   0.087375,
#                   0.087375, 0.087375, 0.087375, 0.087375, 0.07475]


initialValueVy = [-1.250e-16,  4.375e-16,  4.375e-16,  4.375e-16,  4.375e-16,
                 4.375e-16,  2.500e-16, -1.250e-16,  4.375e-16,  4.375e-16,
                 4.375e-16,  4.375e-16,  4.375e-16,  2.500e-16,  4.375e-16,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 8.125e-16,  4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  8.125e-16,  4.375e-16,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  8.125e-16,
                 4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  8.125e-16,  4.375e-16,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  4.375e-16,
                 4.375e-16,  4.375e-16,  4.375e-16,  4.375e-16,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 8.125e-16,  4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  8.125e-16,
                 4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  8.125e-16,  8.125e-16,  8.125e-16,
                 8.125e-16,  8.125e-16,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  8.125e-16,  4.375e-16,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 8.125e-16,  4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  1.000e-15,  8.125e-16,  4.375e-16,  1.000e-15,
                 1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,  8.125e-16,
                 4.375e-16,  1.000e-15,  1.000e-15,  1.000e-15,  1.000e-15,
                 1.000e-15,  8.125e-16,  2.500e-16,  8.125e-16,  8.125e-16,
                 8.125e-16,  8.125e-16,  8.125e-16,  6.250e-16,  2.500e-16,
                 8.125e-16,  8.125e-16,  8.125e-16,  8.125e-16,  8.125e-16,
                 6.250e-16]

# Se generan las ecuaciones simbolicas.

generarEcuaciones(Avx, bvx, ecuacionesVx)
generarEcuaciones(Avy, bvy, ecuacionesVy)

# Se resuelve el sistema de ecuaciones.
#print(newtonRapshon(initialValueVx, 0.001, 10, ecuacionesVx, variablesVx))
print(newtonRapshon(initialValueVy, 0.000000000000000001, 10, ecuacionesVy, variablesVy))

