

# Funcion para recorrer la malla.
def travelMatrix(function, nx, ny, h, d):
    for y in range(1, ny-1):
        for x in range(1, nx-1):
            print(function(x, y, h, d))
            ecuationsSystem[y, x] = function(x, y, h, d)
    print(ecuationsSystem)

# Funcion para discretizar los puntos de la malla.
def discretizar(x, y):
    if(Vh[y, x] == -1):
        return "W" + str(ID[y-1, x-1])
    else:
        return str(Vh[y, x])

""" EJEMPLO NAVIER STOKES MODFICADO"""
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
plt.show()

Vh = np.full((ny, nx), -1)  # Matriz para la temperatura
Vh[:, 0] = 2
Vh[0, :] = 0
Vh[-1, :] = 0
Vh[:, -1] = 0

ID = np.empty((ny-2, nx-2), dtype=int)

valor = 1
for i in range(0, ny-2):
    for j in range(0, nx-2):
        ID[i, j] = valor
        valor += 1

ecuationsSystem = np.full((ny, nx), "c")

print("Malla Navier Stokes")
print(Vh)
print("ID puntos malla Navier Stokes")
print(ID)

def navierStokesSimplify(x, y, h, d):
    return "3 * " + discretizar(x - h, y) + " + " + discretizar(x + h, y) + " + 3 * " + discretizar(x, y - h) + " + " + discretizar(x, y + h) +" - 8 * " + discretizar(x, y)

print("Ecuaciones Navier Stokes")
travelMatrix(navierStokesSimplify, nx, ny, 1, 0)

""" EJEMPLO LAPLACE PROFESORA"""
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
plt.show()

Vh = np.full((ny, nx), -1)  # Matriz para la temperatura
Vh[:, 0] = 0
Vh[0, :] = 0
Vh[-1, :] = 0
Vh[:, -1] = 100

ID = np.empty((ny-2, nx-2), dtype=int)

valor = 1
for i in range(0, ny-2):
    for j in range(0, nx-2):
        ID[i, j] = valor
        valor += 1

ecuationsSystem = np.full((ny, nx), "c")

print("Malla Laplace")
print(Vh)
print("ID puntos malla Laplace")
print(ID)

def laplaceSimplify(x, y, h, d):
    return "1/25 (" + discretizar(x - h, y) + " + " + discretizar(x + h, y) + " + " + discretizar(x, y - h) + " + " + discretizar(x, y + h) + " - 4" + discretizar(x, y) + ")"

print("Ecuaciones Laplace")
travelMatrix(laplaceSimplify, nx, ny, 1, 0)
