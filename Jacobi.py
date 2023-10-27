import numpy as np
import matplotlib.pyplot as plt
import openpyxl

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
            if(Vx[y, x] == -1):
                i = i + 1
                row = np.full(npoints, 0, dtype=object)
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
            if(Vx[y, x] == -1):
                i = i + 1
                row = np.full(npoints, 0, dtype=object)
                function(x, y, h, row, i)
                Avy.append(row)

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de frontera es diferente de 0,
# se agrega al vector de resultados de la ecuacion.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la System de coeficientes de la ecuacion.
# i : Numero de la fila.
def discretizarVx(coeficiente, x, y, row, i):
    if Vx[y, x] == -1:
        row[(IDvx[y - 1, x - 1]) - 1] = coeficiente
    else:
        if Vx[y, x] != 0:
            bvx[i - 1] = bvx[i - 1] + (-1 * (coeficiente * Vx[y, x]))

# Funcion para determinar si un punto es condicion de frontera o no. Segun lo obtenido, se retornara el valor del
# coeficiente que acompana al punto desconocido o el valor de frontera. Si el valor es de frontera es diferente de 0,
# se agrega al vector de resultados de la ecuacion.
# coeficiente : Valor del coeficiente que acompana al punto desconocido.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# row : Fila de la System de coeficientes de la ecuacion.
# i : Numero de la fila.
# Funcion para discretizar los puntos de la malla.
def discretizarVy(coeficiente, x, y, row, i):
    if Vy[y, x] == -1:
        row[(IDvy[y - 1, x - 1]) - 1] = coeficiente
    else:
        if Vy[y, x] != 0:
            bvy[i - 1] = bvy[i - 1] + (-1 * (coeficiente * Vy[y, x]))

# Funcion que representa la ecuacion de Navier Stokes para la velocidad en X previamente simplificada.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# h : Distancia entre puntos de la malla.
# row : Fila de la System de coeficientes de la ecuacion.
# i : Numero de la fila.
def navierStokesSimplifyVx(x, y, h, row, i):
    discretizarVx(3, x - h, y, row, i)
    discretizarVx(1, x + h, y, row, i)
    discretizarVx(3, x, y - h, row, i)
    discretizarVx(1, x, y + h, row, i)
    discretizarVx(-8, x, y, row, i)

# Funcion que representa la ecuacion de Navier Stokes para la velocidad en Y previamente simplificada.
# x : Coordenada x del punto.
# y : Coordenada y del punto.
# h : Distancia entre puntos de la malla.
# row : Fila de la System de coeficientes de la ecuacion.
# i : Numero de la fila.
def navierStokesSimplifyVy(x, y, h, row, i):
    discretizarVy(3, x - h, y, row, i)
    discretizarVy(1, x + h, y, row, i)
    discretizarVy(3, x, y - h, row, i)
    discretizarVy(1, x, y + h, row, i)
    discretizarVy(-8, x, y, row, i)

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

# System de la velocidades en X.
Vx = np.full((ny, nx), -1)
# Restriccion de frontera pared derecha.
Vx[0, :] = 0
# Restriccion de frontera pared superior.
Vx[-1, :] = 0
# Restriccion de frontera pared inferior.
Vx[:, -1] = 0
# Restriccion de frontera pared izquierda.
Vx[:, 0] = 5

# System de la velocidades en y.
Vy = Vx.copy()
# Restriccion de frontera pared izquierda.
Vy[:, 0] = 0

# Restriccion columna 1.
for i in range(1, nyc + 1):
    for j in range((Vx.shape[1]//2) - (nxc // 2), ((Vx.shape[1]//2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

# Restriccion columna 2
for i in range(Vx.shape[0] - (nyc + 1), Vx.shape[0] - 1):
    for j in range((Vx.shape[1]//2) - (nxc // 2), ((Vx.shape[1]//2) + (nxc // 2)) + 1):
        Vx[i, j] = 0
        Vy[i, j] = 0

# MATRICES PARA EL SISTEMA DE ECUACIONES.

## Matrices para Vx
# System de los coeficientes de la ecuacion.
Avx = []
# Vector de los resultados de la ecuacion.
bvx = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvx = ["Vx" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# System de los identificadores de los puntos de la malla.
IDvx = np.empty((ny - 2, nx - 2), dtype=int)
# Se asigna un valor a cada punto de la malla que no es de frontera.
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

print(IDvx)

## Matrices para Vy
# System de los coeficientes de la ecuacion.
Avy = []
# Vector de los resultados de la ecuacion.
bvy = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvy = ["Vy" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# System de los identificadores de los puntos de la malla.
IDvy = IDvx


travelMatrixVx(navierStokesSimplifyVx, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

travelMatrixVy(navierStokesSimplifyVy, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

## Entrega 2 Jacobi-SobreRelajado

## Determina si la System es diagonal dominante.
def dominante(System):
    for i in range(len(System)):
        suma = 0
        for j in range(len(System)):
            if i != j:
                suma += abs(System[i][j])
            if abs(System[i][i]) < suma:
                return False
    return True

print(dominante(Avx))
print(dominante(Avy))

def calcToleranciaNorInf(arrayact, arrayant):
    x = abs(max((arrayact) - (arrayant))) / abs(max(arrayact))
    return x


def aplicarRelajación(A, B, w):
    n = len(A)
    X = np.zeros((n, n+1))

    for i in range(n):
        V = np.array(A[i])
        div = V[i]
        V = V  * -1
        V = np.append(V, B[i])
        V = (V * w) / div
        V[i] = 1-w
        X[i, :] = V

    return X


def jacobi(matriz, array, tolerancia,c):
    ArrayAct = np.zeros(len(array))
    for i in range(len(array)):
        x = 0
        for j in range(len(array)):
                x += matriz[i][j] * array[j]

        ArrayAct[i] = x + matriz[i][len(array)] 

    if(calcToleranciaNorInf(ArrayAct, array) < tolerancia):
        #return [ArrayAct, calcToleranciaNorInf(ArrayAct, array), c+1]
        return ArrayAct
    else:
        #print(ArrayAct, calcToleranciaNorInf(ArrayAct, array), c+1)
        return jacobi(matriz, ArrayAct, tolerancia, c+1)

def aplicarJacobi(A, B, solInicial, w, tolerancia):
    sistema = aplicarRelajación(A, B, w)
    
    return jacobi(sistema, solInicial, tolerancia, 0)

## Solución para x
cero = np.zeros(len(Avx))
print("Jacobi Relajado para Vx")
solVx = aplicarJacobi(Avx, bvx, cero, 1.01, 0.001)
print(solVx)

## Solución para y
solIniy = np.full(len(Avy), 0.1)
print("Jacobi Relajado para Vy")
solVy = aplicarJacobi(Avy, bvy, solIniy, 1.01, 0.001)
print(solVy)

"""

# GRAFICA DE LA MALLA.
x = np.linspace(0, anchoMalla, nx)
y = np.linspace(0, alturaMalla, ny)
X, Y = np.meshgrid(x, y)

wall_x = np.array([((nx - 1) // 2) - ((nxc -1 )// 2) , ((nx - 1) // 2) - ((nxc -1 )// 2) , ((nx - 1) // 2) + ((nxc -1 )// 2), ((nx - 1) // 2) + ((nxc -1 )// 2)])  # Coordenadas en X de las paredes
wall_y = np.array([0, nyc, nyc, 0])
wall_y1 = np.array([ny-1, (ny-1) - nyc,  (ny-1) - nyc, ny-1])
plt.figure(figsize=(10, 5))
plt.plot(wall_x, wall_y, 'r-')  # Dibuja las paredes en rojo
plt.plot(wall_x, wall_y1, 'r-')  # Dibuja las paredes en rojo
plt.plot(X, Y, 'bo', markersize=1)  # 'bo' representa puntos azules para la malla
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Malla con Paredes para el Dominio 2D')
plt.grid(True)
plt.show()

"""