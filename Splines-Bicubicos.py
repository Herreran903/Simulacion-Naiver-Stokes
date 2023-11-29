import numpy as np
from sympy import symbols, diff, sympify, lambdify
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Recibe la matriz con los puntos desconocidos y un vector de soluciones para generar la nueva matriz con los 
# puntos desconocidos evaluados en el vector solucion.
def reemplazarValoresDefault(matriz, vector):
    # Crear una copia de la matriz para evitar cambios en la original
    nueva_matriz = matriz.astype(float)

    # Obtener las posiciones de los valores -1 en la matriz
    posiciones = np.where(matriz == -1)

    # Iterar sobre las posiciones y reemplazar los valores con los del vector
    for i in range(len(posiciones[0])):
        fila = posiciones[0][i]
        columna = posiciones[1][i]
        nueva_matriz[fila, columna] = vector[i]

    return nueva_matriz


np.set_printoptions(linewidth=np.inf)
# VALORES INICIALES.
nx = 21
ny = 11
nxc = 5
nyc = 3

Vx = np.full((ny, nx), -1)
Vx[0, :] = 0
Vx[-1, :] = 0
Vx[:, -1] = 0
Vx[:, 0] = 5

Vy = Vx.copy()
Vy[:, 0] = 0

for i in range(1, nyc + 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

for i in range(Vx.shape[0] - (nyc + 1), Vx.shape[0] - 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

IDvx = np.empty((ny - 2, nx - 2), dtype=int)
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

# Vector solucion generado por el metodo de Jacobi para el eje X
resultsInX = [2.49999494e+00, 1.34671408e+00, 7.73798462e-01, 4.68633309e-01,
              2.95258513e-01, 1.88968667e-01, 1.09593910e-01, 1.73284655e-04,
              3.18332053e-04, 4.00371709e-04, 3.92951411e-04, 3.90731037e-04,
              2.84817780e-04, 2.38989358e-04, 3.65324664e+00, 2.49992565e+00,
              1.68162075e+00, 1.13239643e+00, 7.67226317e-01, 5.16343914e-01,
              3.09880550e-01, 1.06035334e-03, 1.65671735e-03, 1.77578694e-03,
              1.71530554e-03, 1.38482328e-03, 1.24680229e-03, 6.94801020e-04,
              4.22605889e+00, 3.31791461e+00, 2.49936939e+00, 1.84123694e+00,
              1.33841444e+00, 9.52407873e-01, 6.01112770e-01, 6.32313784e-03,
              7.25628507e-03, 6.56684902e-03, 5.30699225e-03, 4.41347302e-03,
              3.02916992e-03, 2.28477751e-03, 4.53082569e+00, 3.86596501e+00,
              3.15517923e+00, 2.49605700e+00, 1.92973091e+00, 1.45358076e+00,
              1.02234655e+00, 5.43091771e-01, 3.02538575e-01, 1.73846705e-01,
              1.02133415e-01, 6.12108336e-02, 4.01603720e-02, 2.76898580e-02,
              1.96023874e-02, 1.43973184e-02, 1.00997272e-02, 7.88489895e-03,
              4.30415263e-03, 4.70244583e+00, 4.22637905e+00, 3.64924003e+00,
              3.04973756e+00, 2.48042223e+00, 1.96051687e+00, 1.47085049e+00,
              9.75834595e-01, 6.16567528e-01, 3.81611973e-01, 2.33786690e-01,
              1.43587921e-01, 9.06950696e-02, 5.95640001e-02, 4.06026867e-02,
              2.80140631e-02, 2.06740715e-02, 1.35129800e-02, 9.56204826e-03,
              4.80074345e+00, 4.45645511e+00, 3.99975243e+00, 3.48112352e+00,
              2.94524757e+00, 2.41012993e+00, 1.84383962e+00, 1.14678840e+00,
              7.17297743e-01, 4.46469507e-01, 2.76782490e-01, 1.71905840e-01,
              1.15520275e-01, 8.08126151e-02, 5.77077399e-02, 4.26093071e-02,
              2.99896656e-02, 2.34589978e-02, 1.28269267e-02, 4.84210227e+00,
              4.57069393e+00, 4.19941944e+00, 3.75603517e+00, 3.26595310e+00,
              2.72176753e+00, 1.95878003e+00, 5.46527703e-02, 6.32524252e-02,
              5.76150319e-02, 4.67941108e-02, 3.90536834e-02, 2.68908812e-02,
              2.02978039e-02, 4.76397191e+00, 4.47021240e+00, 4.12855820e+00,
              3.73961122e+00, 3.30384394e+00, 2.78457193e+00, 1.97610011e+00,
              2.75395412e-02, 4.32668955e-02, 4.66090850e-02, 4.52016809e-02,
              3.66363743e-02, 3.30431671e-02, 1.84823040e-02, 4.11516054e+00,
              3.62945635e+00, 3.27925397e+00, 2.96063484e+00, 2.62727046e+00,
              2.22674655e+00, 1.57573427e+00, 1.35170589e-02, 2.49351510e-02,
              3.14794984e-02, 3.10194529e-02, 3.09201856e-02, 2.26334477e-02,
              1.89807749e-02]

plt.imshow(reemplazarValoresDefault(Vx, resultsInX), cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()



# FUNCIONES PARA LA INTERPOLACION BICUBICA.

# variables simbolicas
variables = symbols("x y")

# función para las ecuaciones de la spline cubica
def biCubicSplineEquation():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + " * x^" + str(i) + " * y^" + str(j) + " + "
    
    #borra el ultimo +
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubica de la derivada de x
def biCubicSplineEquationX():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(i) + " * x^" + str(i - 1) + " * y^" + str(j) + " + "

    #borra el ultimo +
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubicade la derivada de y
def biCubicSplineEquationY():

    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(j) + " * x^" + str(i) + " * y^" + str(j - 1) + " + "

    #borra el ultimo +
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubica de xy
def biCubicSplineEquationXY():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(i) + "*" + str(j) + " * x^" + str(i - 1) + " * y^" + str(
                j - 1) + " + "
    
    #borra el ultimo +
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para generar y agrupar todas las 16 ecuaciones de la spline cubica.
def generateSystemEquations(variables):
    ecuations = []
    ecuationBiCubic = biCubicSplineEquation()
    ecuationBiCubicX = biCubicSplineEquationX()
    ecuationBiCubicY = biCubicSplineEquationY()
    ecuationBiCubicXY = biCubicSplineEquationXY()
    for i in range(0, 2):
        for j in range(0, 2):
            ecuations.append(lambdify(variables, ecuationBiCubic)(*[j, i]))
    for i in range(0, 2):
        for j in range(0, 2):
            ecuations.append(lambdify(variables, ecuationBiCubicX)(*[j, i]))
    for i in range(0, 2):
        for j in range(0, 2):
            ecuations.append(lambdify(variables, ecuationBiCubicY)(*[j, i]))
    for i in range(0, 2):
        for j in range(0, 2):
            ecuations.append(lambdify(variables, ecuationBiCubicXY)(*[j, i]))

    return ecuations

# función para generar la matriz de coeficientes
def generateCoefficientMatrix(variables):
    coefs = []
    ecuations = generateSystemEquations(variables)
    for ecuation in ecuations:
        coefficients = [ecuation.coeff(symbols(f'a{j}{i}')) for i in range(4) for j in range(4)]
        coefs.extend(coefficients)
    coefficientMatrix = [coefs[i:i + 16] for i in range(0, len(coefs), 16)]

    return np.matrix(coefficientMatrix).astype(float)


# FUNCIONES PARA LAS DERIVADAS Y MALLA. Recibe un X,Y con esto calcula el valor en el punto. Se usa para las derivadas.
def valueOfMesh(x, y):
    if x != 0 and y != 0:
        try:
            id = IDvx[y - 1, x - 1]
            if id != 0:
                return resultsInX[id - 1]
            else:
                return 0
        except IndexError:
            try:
                return Vx[y, x]
            except IndexError:
                return 0
    else:
        try:
            return Vx[y, x]
        except IndexError:
            return 0

# función para la derivada de x
def valueOfFirstDerivateX(x, y):
    firstDerivateX = (valueOfMesh(x + 1, y) - valueOfMesh(x - 1, y)) / 2
    return firstDerivateX

# función para la derivada de y
def valueOfFirstDerivateY(x, y):
    firstDerivateY = (valueOfMesh(x, y + 1) - valueOfMesh(x, y - 1)) / 2
    return firstDerivateY

# función para la derivada de xy
def valueOfFirstDerivateXY(x, y):
    firstDerivateXY = (valueOfMesh(x - 1, y - 1) + valueOfMesh(x + 1, y + 1) - valueOfMesh(x + 1, y - 1) + valueOfMesh(
        x - 1, y + 1)) / 4
    return firstDerivateXY

# función para calcular el vector de funciones
def calculateVectorFunctions(x, y):
    functions = [valueOfMesh, valueOfFirstDerivateX, valueOfFirstDerivateY, valueOfFirstDerivateXY]
    vector = []
    for n in range(0, 4):
        function = functions[n]
        for i in range(y, y + 2):
            for j in range(x, x + 2):
                vector.append(function(j, i))
    return vector

# función para calcular el vector de constantes
def calculateVectorConstants(x, y):
    vector = []
    for i in range(y, y + 4):
        for j in range(x, x + 4):
            vector.append("a" + str(j) + str(i))

    return vector

# calcula la matriz inversa
inverseMatrix = np.linalg.inv(generateCoefficientMatrix(variables))

# calcula los coeficientes de la ecuacion de la spline cubica
def calculateConstants(x, y, inverseMatrix):
    constants = inverseMatrix @ calculateVectorFunctions(x, y)

    return np.array(constants)[0]

# genera la matriz de constantes
def tourMatrix(nx, ny, matrix, inverseMatrix):
    constants = []
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (matrix[y, x] == -1):
                constants.append(calculateConstants(x, y, inverseMatrix))
            else:
                if x == 25:
                    constants.append(calculateConstants(x, y, inverseMatrix))

    return np.array(constants)


constants = tourMatrix(nx, ny, Vx, inverseMatrix)

# calcula los 3 nuevos puntos por cada punto hallado anteriormente. el punto a su derecha, el diagonal y el de abajo.
def calculateNewPoints(i, x, y, variables, constants):
    ecuacion_spline = biCubicSplineEquation()
    constantes = calculateVectorConstants(0, 0)

    # Sustituir los valores de las constantes en la ecuación de la spline
    ecuacion_sustituida = ecuacion_spline.subs(
        {symbols(f'a{j}{k}'): constants[i][constantes.index(f'a{j}{k}')] for k in range(4) for j in range(4)})
    
    diagonal = ecuacion_sustituida.subs({symbols('x'): 0.5, symbols('y'): 0.5})
    abajo = ecuacion_sustituida.subs({symbols('x'): 0, symbols('y'): 0.5})
    derecha = ecuacion_sustituida.subs({symbols('x'): 0.5, symbols('y'): 0})

    return (derecha, abajo, diagonal)


## Definiciones para los diferentes puntos nuevos

# Derecha
i = 0
Ader = Vx.copy().astype(float)
Bder = np.zeros(len(resultsInX)).astype(float)

# Abajo
Abaj = Vx.copy().astype(float)
Babaj = np.zeros(len(resultsInX)).astype(float)

# Diagonal
Adiag = Vx.copy().astype(float)
Bdiag = np.zeros(len(resultsInX)).astype(float)

# Se rellenan los vectores por cada posición nueva iterando con la función calculateNewPoints.
for y in range(0, ny - 1):
    for x in range(0, nx - 1):
        if (Vx[y, x] == -1):
            nuevos_valores = calculateNewPoints(i, x, y, variables, constants)

            Bder[i] = nuevos_valores[0]
            Ader[y, x] = nuevos_valores[0]

            Babaj[i] = nuevos_valores[1]
            Abaj[y, x] = nuevos_valores[1]

            Bdiag[i] = nuevos_valores[2]
            Adiag[y, x] = nuevos_valores[2]
            i += 1

def reemplazar_valores(matriz, vector):
    # Crear una copia de la matriz para evitar cambios en la original
    nueva_matriz = matriz.astype(float)

    # Obtener las posiciones de los valores -1 en la matriz
    posiciones = np.where(matriz == -1)

    # Iterar sobre las posiciones y reemplazar los valores con los del vector
    i = 1

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (Vx[y, x] == -1):
                fila = posiciones[0][i]
                columna = posiciones[1][i]

                # punto x,y
                nueva_matriz[fila, columna] = vector[i]
                # punto x+1,y derecha
                nueva_matriz[fila, columna + 1] = vector[i+1]
                # punto x,y+1 abajo
                nueva_matriz[fila + 1, columna] = vector[i+2]
                # punto x+1,y+1 diagonal
                nueva_matriz[fila + 1, columna + 1] = vector[i+3]

                i += 4


    return nueva_matriz

# intercala los vectores del puntos de la soulción con sus vecionos, es decir
# Punto [0,0], Punto [1,0], Punto [0,1], Punto [1,1] y asi sucevivamente.
def intercalar_vectores(lista1, lista2, lista3, lista4):
    # Verificar que todas las listas tengan la misma longitud
    if len(lista1) != len(lista2) != len(lista3) != len(lista4):
        raise ValueError('Todas las listas deben tener la misma longitud')

    # Inicializar la lista resultado
    resultado = []

    # Iterar sobre los elementos de todas las listas e intercalarlos
    for elem1, elem2, elem3, elem4 in zip(lista1, lista2, lista3, lista4):
        resultado.extend([elem1, elem2, elem3, elem4])

    return resultado

# Vector con los puntos de la solución intercalados con sus vecinos
vectorTotal = intercalar_vectores(resultsInX, Bder, Babaj, Bdiag)

# valores para la nueva matriz con las filas y columnas duplicadas. Tambien se duplican los bloques.
nx =  40 
ny =  20 
nxc = 10 
nyc = 6 

# Generacion de la nueva matriz
Vx = np.full((ny, nx), -1)
Vx[0, :] = 0
Vx[-1, :] = 0
Vx[:, -1] = 0
Vx[:, 0] = 5

Vy = Vx.copy()
Vy[:, 0] = 0
for i in range(1, nyc + 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2))):
        Vx[i, j] = 0
        Vy[i, j] = 0

for i in range(Vx.shape[0] - (nyc + 1), Vx.shape[0] - 1):
    for j in range((Vx.shape[1] // 2) - (nxc // 2), ((Vx.shape[1] // 2) + (nxc // 2))):
        Vx[i, j] = 0
        Vy[i, j] = 0

IDvx = np.empty((ny - 2, nx - 2), dtype=int)
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

IDvx = np.empty((ny - 2, nx - 2), dtype=int)
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

# Función para generar la matriz final con los puntos calculados anteriormente
def buildMatriz(vector, matriz):
    j = 0
    for y in range(1, matriz.shape[0] - 1, 1):
        for x in range(1, matriz.shape[1] - 1, 1):
            if matriz[y, x] == -1:
                matriz[y:y+2, x:x+2] = np.array([[vector[0 + j], vector[1 + j]],
                                                  [vector[2 + j], vector[3 + j]]], dtype=float)
                j += 4

    return matriz

Vx2 = Vx.copy().astype(float)
MatrizFinal = buildMatriz(vectorTotal, Vx2)

plt.imshow(MatrizFinal, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()
