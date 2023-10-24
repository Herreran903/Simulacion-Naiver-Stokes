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
# row : Fila de la matriz de coeficientes de la ecuacion.
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
# row : Fila de la matriz de coeficientes de la ecuacion.
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
# row : Fila de la matriz de coeficientes de la ecuacion.
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
# row : Fila de la matriz de coeficientes de la ecuacion.
# i : Numero de la fila.
def navierStokesSimplifyVy(x, y, h, row, i):
    discretizarVy(3, x - h, y, row, i)
    discretizarVy(1, x + h, y, row, i)
    discretizarVy(3, x, y - h, row, i)
    discretizarVy(1, x, y + h, row, i)
    discretizarVy(-8, x, y, row, i)

# MEDIDAS Y RESTRICCIONES DE LA MALLA.
# Ancho de la malla.
anchoMalla = 16
# Altura de la malla.
alturaMalla = 8
# Numero de puntos de la malla en x.
nx = 17
# Numero de puntos de la malla en y.
ny = 9
# Numero de puntos de las columnas en x.
nxc = 5
# Numero de puntos de las columnas en y.
nyc = 2

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
# Matriz de los coeficientes de la ecuacion.
Avx = []
# Vector de los resultados de la ecuacion.
bvx = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvx = ["Vx" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# Matriz de los identificadores de los puntos de la malla.
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
# Matriz de los coeficientes de la ecuacion.
Avy = []
# Vector de los resultados de la ecuacion.
bvy = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvy = ["Vy" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# Matriz de los identificadores de los puntos de la malla.
IDvy = IDvx

# RESULTADOS MALLA.
## Discretizacion malla para Vx.
print("Malla Navier Stokes para Vx")
print(Vx)
print("ID puntos malla Navier Stokes")
print(IDvx)
print("Ecuaciones Navier Stokes")
travelMatrixVx(navierStokesSimplifyVx, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

## Discretizacion malla para Vy.
print("Malla Navier Stokes para Vy")
print(Vy)
print("ID puntos malla Navier Stokes")
print(IDvy)
print("Ecuaciones Navier Stokes")
travelMatrixVy(navierStokesSimplifyVy, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

# RESULTADOS SISTEMA DE ECUACIONES.
workbook = openpyxl.Workbook()
## Hoja de excel para Vx.
ArrayExcelVx = [list(row) for row in Avx]

# Se agrega el vector de variables a la matriz de coeficientes.
for i in range(len(ArrayExcelVx)):
    ArrayExcelVx[i].append(Xvx[i])
# Se agrega el vector de resultados a la matriz de coeficientes.
for i in range(len(ArrayExcelVx)):
    ArrayExcelVx[i].append(bvx[i])

sheet1 = workbook.active
sheet1.title = "Vx"

# Llenar el archivo de Excel con los datos de la matriz.
for row in ArrayExcelVx:
    sheet1.append(row)

## Hoja de excel para Vy.
ArrayExcelVy = [list(row2) for row2 in Avy]

# Se agrega el vector de variables a la matriz de coeficientes.
for i in range(len(ArrayExcelVy)):
    ArrayExcelVy[i].append(Xvy[i])
# Se agrega el vector de resultados a la matriz de coeficientes.
for i in range(len(ArrayExcelVy)):
    ArrayExcelVy[i].append(bvy[i])

sheet2 = workbook.create_sheet(title= "Vy")

# Llenar el archivo de Excel con los datos de la matriz.
for row2 in ArrayExcelVy:
    sheet2.append(row2)

# Guardar el archivo de Excel.
workbook.save('equationsNavierStokes.xlsx')

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















