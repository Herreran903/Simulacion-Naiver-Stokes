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
bvx = np.full(141, 0, dtype=object)
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
bvy = np.full(141, 0, dtype=object)
# Vector de las variables de la ecuacion.
Xvy = ["Vy" + str(i) for i in range(1, (ny - 2) * (nx - 2) + 1)]
# System de los identificadores de los puntos de la malla.
IDvy = IDvx


travelMatrixVx(navierStokesSimplifyVx, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

travelMatrixVy(navierStokesSimplifyVy, nx, ny, 1, ((ny - 2) * (nx - 2)) - (2 * (nxc * nyc)))

## Entrega 2 Gradiente Conjugado

def igualdad(A, Msol):
    n = len(A)
    sol = np.zeros(n)
    for i in range(n):
        x = 0
        for j in range(n):
            x += A[i][j] * Msol[j]
        sol[i] = x

    return sol
def gradianteConjugado(matriz, vector, puntoInicial, tol):
    x = puntoInicial
    gradiente = vector - (matriz @ x)
    direccion = gradiente

    i = 0
    while np.linalg.norm(gradiente) > tol:
        alpha = np.dot(gradiente.T, gradiente) / np.dot(direccion.T, np.dot(matriz, direccion))
        x = x + (alpha * direccion)
        gradienteNuevo = gradiente - alpha * np.dot(matriz, direccion)
        beta = np.dot(gradienteNuevo.T, gradienteNuevo) / np.dot(gradiente.T, gradiente)
        direccionNueva = gradienteNuevo + beta * direccion
        gradiente = gradienteNuevo
        direccion = direccionNueva

        print(np.linalg.norm(gradiente))
        i += 1

    print(f'Convergencia alcanzada después de {i} iteraciones.')
    return x

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

print("Gradiente Conjugado para Vx")
solX = gradianteConjugado(Avx, bvx, np.array(initialValueVx), 1)
print(solX)

initialValueVy = [0.02425,  0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.0495,   0.02425,
                  0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.0495,   0.062125, 0.1,
                  0.1,      0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,
                  0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,
                  0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,
                  0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,
                  0.1,      0.062125, 0.062125, 0.062125, 0.062125, 0.062125, 0.1,      0.1,
                  0.1,      0.1,      0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,
                  0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,
                  0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
                  0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
                  0.087375, 0.087375, 0.087375, 0.087375, 0.1,      0.1,      0.1,      0.1,
                  0.1,      0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,
                  0.1,      0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,
                  0.087375, 0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.087375,
                  0.062125, 0.1,      0.1,      0.1,      0.1,      0.1,      0.087375, 0.0495,
                  0.087375, 0.087375, 0.087375, 0.087375, 0.087375, 0.07475,  0.0495,   0.087375,
                  0.087375, 0.087375, 0.087375, 0.087375, 0.07475]

print("Gradiente Conjugado para Vy")
solY = gradianteConjugado(Avy, bvy, np.array(initialValueVy), 10)
print(solY)



