import numpy as np
import matplotlib.pyplot as plt
import openpyxl


# Funcion para recorrer la malla.
def travelMatrix(function, nx, ny, h, npoints):
    i = 0
    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if(Vx[y, x] == -1):
                i = i + 1
                row = np.full(npoints, 0, dtype=object)
                function(x, y, h, row, i)
                A.append(row)

# Funcion para discretizar los puntos de la malla.
def discretizar(coeficiente, x, y, row, i):
    if Vx[y, x] == -1:
        row[(ID[y - 1, x - 1]) - 1] = coeficiente
    else:
        if Vx[y, x] != 0:
            b[i - 1] = b[i - 1] + (-1 * (coeficiente * Vx[y, x]))


# EJEMPLO NAVIER STOKES MODFICADO
heightMesh = 5
widthMesh = 10

nx = 11
ny = 6

x = np.linspace(0, widthMesh, nx)
y = np.linspace(0, heightMesh, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 5))
plt.plot(X, Y, 'bo', markersize=1)  # 'bo' representa puntos azules para la malla
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Malla con Paredes para el Dominio 2D')
plt.grid(True)
#plt.show()

# Matriz de la velocidades en X.
Vx = np.full((ny, nx), -1)
# Restriccion de frontera pared izquierda.
Vx[:, 0] = 2
# Restriccion de frontera pared derecha.
Vx[0, :] = 0
# Restriccion de frontera pared superior.
Vx[-1, :] = 0
# Restriccion de frontera pared inferior.
Vx[:, -1] = 0
# Restriccion columna 1.
for i in range(1, 2):
    for j in range(4, 7):
        Vx[i, j] = 0
# Restriccion columna 2
for i in range(4, 5):
    for j in range(4, 7):
        Vx[i, j] = 0
# Matriz de los coeficientes de la ecuacion.
A = []
# Vector de los resultados de la ecuacion.
b = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
x = ["W" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# Matriz de los identificadores de los puntos de la malla.
ID = np.empty((ny - 2, nx - 2), dtype=int)

valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        ID[i, j] = valor
        valor += 1

ID = [[1, 2, 3, 0, 0, 0, 4, 5, 6],
      [7,  8,  9, 10, 11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20, 21, 22, 23, 24],
      [25, 26, 27,  0,  0,  0, 28, 29, 30]]

ID = np.array(ID)

print("Malla Navier Stokes")
print(Vx)
print("ID puntos malla Navier Stokes")
print(ID)


def navierStokesSimplify(x, y, h, row, i):
    discretizar(3, x - h, y, row, i)
    discretizar(1, x + h, y, row, i)
    discretizar(3, x, y - h, row, i)
    discretizar(1, x, y + h, row, i)
    discretizar(-8, x, y, row, i)


print("Ecuaciones Navier Stokes")
travelMatrix(navierStokesSimplify, nx, ny, 1, (ny - 2) * (nx - 2))

ArrayExcel = [list(row) for row in A]

# Se agrega el vector de variables a la matriz de coeficientes.
for i in range(len(ArrayExcel)):
    ArrayExcel[i].append(x[i])
# Se agrega el vector de resultados a la matriz de coeficientes.
for i in range(len(ArrayExcel)):
    ArrayExcel[i].append(b[i])

workbook = openpyxl.Workbook()
sheet = workbook.active

# Llenar el archivo de Excel con los datos de la matriz
for row in ArrayExcel:
    sheet.append(row)

# Guardar el archivo de Excel
workbook.save('equationsNavierStokes3.xlsx')

# EJEMPLO LAPLACE PROFESORA
heightMesh = 10
widthMesh = 20

nx = 5
ny = 3

x = np.linspace(0, widthMesh, nx)
y = np.linspace(0, heightMesh, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 5))
plt.plot(X, Y, 'bo', markersize=1)  # 'bo' representa puntos azules para la malla
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Malla con Paredes para el Dominio 2D')
plt.grid(True)
#plt.show()

# Matriz de la temperatura.
Vx = np.full((ny, nx), -1)
# Restriccion de frontera pared izquierda.
Vx[:, 0] = 0
# Restriccion de frontera pared superior.
Vx[0, :] = 0
# Restriccion de frontera pared inferior.
Vx[-1, :] = 0
# Restriccion de frontera pared derecha.
Vx[:, -1] = 100
# Matriz de los coeficientes de la ecuacion.
A = []
# Vector de los resultados de la ecuacion.
b = np.full((ny - 2) * (nx - 2), 0, dtype=object)
# Vector de las variables de la ecuacion.
x = ["W" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# Matriz de los identificadores de los puntos de la malla.
ID = np.empty((ny - 2, nx - 2), dtype=int)

valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        ID[i, j] = valor
        valor += 1



print("Malla Laplace")
print(Vx)
print("ID puntos malla Laplace")
print(ID)


def laplaceSimplify(x, y, h, d, i):
    discretizar(1/25, x - h, y, d, i)
    discretizar(1/25, x + h, y, d, i)
    discretizar(1/25, x, y - h, d, i)
    discretizar(1/25, x, y + h, d, i)
    discretizar(-4/25, x, y, d, i)


print("Ecuaciones Laplace")
travelMatrix(laplaceSimplify, nx, ny, 1, (ny - 2) * (nx - 2))

ArrayExcel = [list(row) for row in A]

# Se agrega el vector de variables a la matriz de coeficientes.
for i in range(len(ArrayExcel)):
    ArrayExcel[i].append(x[i])
# Se agrega el vector de resultados a la matriz de coeficientes.
for i in range(len(ArrayExcel)):
    ArrayExcel[i].append(b[i])

workbook = openpyxl.Workbook()
sheet = workbook.active

# Llenar el archivo de Excel con los datos de la matriz
for row in ArrayExcel:
    sheet.append(row)

# Guardar el archivo de Excel
#workbook.save('equationsLaplace.xlsx')
