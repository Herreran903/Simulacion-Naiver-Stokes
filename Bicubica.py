import numpy as np
from sympy import symbols, diff, sympify, lambdify
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

def reemplazar_valores1(matriz, vector):
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
Vx[:, 0] = 1000

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

plt.imshow(reemplazar_valores1(Vx, resultsInX), cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()
resultsInX = [499.9989712183634, 269.34287709302356, 154.75956153648295, 93.72692869838637, 59.05129159587471, 37.79433330579631, 21.918237111618772, 0.034062329851489596, 0.0653470572982893, 0.0766243223826586, 0.08429278789089881, 0.06949411804143557, 0.067428916890518, 0.037990202458478825, 730.6493911015791, 499.98494093108616, 336.32461175899056, 226.47845636838937, 153.44672437786613, 103.26688972718057, 61.97810344531662, 0.2142725103768897, 0.3253191154887862, 0.36674681829875055, 0.3233534388102537, 0.3049742336346947, 0.21339884930670883, 0.17030982673331851, 845.2116390746651, 663.5834024766963, 499.8728399368059, 368.2495173042229, 267.67956376547806, 190.4865767368272, 120.21766938094996, 1.257869397272798, 1.4673110177795126, 1.2835331704011468, 1.1087092107960834, 0.8127362985144129, 0.6896200424135375, 0.37895147999386564, 906.1654432990545, 773.1920799818689, 631.0381126100076, 499.2072966140864, 385.95354383568264, 290.7060802861571, 204.48230711747067, 108.60733966801281, 60.51882902327496, 34.759105769922684, 20.437900201216802, 12.229311192450865, 8.054097726792534, 5.499741108571821, 3.984271035329513, 2.7780684316751807, 2.1604540658385574, 1.3992834562966032, 1.0149979552550732, 940.4886441607111, 845.2776066346846, 729.8441236070852, 609.955524640304, 496.07175918651296, 392.12335564442503, 294.14644170918916, 195.19337112909037, 123.2886703893572, 76.34872323219219, 46.73148413452714, 28.749367334182196, 18.094043656444892, 11.986595540050432, 8.000121014404986, 5.785522886188896, 3.8705961110027385, 3.016484893955473, 1.6216008544142655, 960.1496078189622, 891.2882444894312, 799.9573134897558, 696.2123458704509, 589.0716819565961, 481.99566718630706, 368.8070275072002, 229.3245766778426, 143.49292439925347, 89.26317835651587, 55.39015647921407, 34.3426152736944, 23.170025859730057, 16.0481347299502, 11.732270573126284, 8.218932856084733, 6.417504128603806, 4.161362489515188, 3.025466298085745, 968.4191885617996, 914.1431499619036, 839.8744791948113, 751.2263426045819, 653.1604914092375, 544.3988233116265, 391.7117706976287, 10.869997102525849, 12.794152264555933, 11.256362772759067, 9.781186634993638, 7.1867280796493525, 6.125004430396264, 3.3644823173374965, 952.7960818250586, 894.0373398154454, 825.7242371624243, 747.8996161300473, 660.8086189983524, 556.8627745172275, 395.2743305275266, 5.566914315697598, 8.492169242832155, 9.631551955568945, 8.514321436567634, 8.074050825488358, 5.650538134775167, 4.531149826500408, 823.0306581579887, 725.8962604012888, 655.8400665792208, 592.1488461641229, 525.4204088767749, 445.39845250139285, 315.10221511489834, 2.6557340435351033, 5.121654728965543, 6.019935508848876, 6.659436917753968, 5.493624908104855, 5.360952392387557, 3.014631466201017]
#
# resultsInX = [2.49998758e+00, 1.34669132e+00, 7.73744912e-01, 4.68535777e-01,
#                2.95095021e-01, 1.88755238e-01, 1.09382311e-01, 1.37785071e-04,
#                2.41919248e-04, 2.70391003e-04, 2.49607178e-04, 1.95860999e-04,
#                1.41590332e-04, 8.23670857e-05, 3.65322341e+00, 2.49984851e+00,
#                1.68145155e+00, 1.13206534e+00, 7.66705648e-01, 5.15610734e-01,
#                3.09187764e-01, 9.03258685e-04, 1.31263006e-03, 1.32012930e-03,
#                1.09267967e-03, 8.27375353e-04, 5.54827115e-04, 3.25266577e-04,
#                4.22600228e+00, 3.31773913e+00, 2.49895657e+00, 1.84047923e+00,
#                1.33713200e+00, 9.50681674e-01, 5.99287030e-01, 5.63680890e-03,
#                6.15763086e-03, 5.07659920e-03, 3.79510446e-03, 2.58509546e-03,
#                1.68690974e-03, 9.22804944e-04, 4.53071607e+00, 3.86560175e+00,
#                3.15437749e+00, 2.49448255e+00, 1.92720302e+00, 1.44984754e+00,
#                1.01803879e+00, 5.39211628e-01, 2.99098211e-01, 1.70612947e-01,
#                9.92213139e-02, 5.84108264e-02, 3.71926556e-02, 2.42894216e-02,
#                1.61040081e-02, 1.04200614e-02, 6.70587635e-03, 4.05270848e-03,
#                2.21916832e-03, 4.70224389e+00, 4.22575015e+00, 3.64776299e+00,
#                3.04700165e+00, 2.47574354e+00, 1.95390532e+00, 1.46243187e+00,
#                9.67424262e-01, 6.08255990e-01, 3.73985592e-01, 2.26561835e-01,
#                1.36961275e-01, 8.39588079e-02, 5.28099073e-02, 3.32554998e-02,
#                2.11719460e-02, 1.30500460e-02, 7.95847450e-03, 4.18266317e-03,
#                4.80042719e+00, 4.45541124e+00, 3.99743823e+00, 3.47659430e+00,
#                2.93793771e+00, 2.39935049e+00, 1.83129714e+00, 1.13538545e+00,
#                7.07091234e-01, 4.36839599e-01, 2.68080678e-01, 1.63527904e-01,
#                1.06630017e-01, 7.06258110e-02, 4.72210801e-02, 3.06975871e-02,
#                1.98118954e-02, 1.19897665e-02, 6.57035465e-03, 4.84163302e+00,
#                4.56922749e+00, 4.19599449e+00, 3.74969309e+00, 3.25528096e+00,
#                2.70725101e+00, 1.94345481e+00, 4.84899687e-02, 5.33795794e-02,
#                4.42402655e-02, 3.32012742e-02, 2.26639141e-02, 1.48102969e-02,
#                8.10430891e-03, 4.76340585e+00, 4.46835249e+00, 4.12443119e+00,
#                3.73161226e+00, 3.29110652e+00, 2.76677082e+00, 1.95904740e+00,
#                2.33065041e-02, 3.40096628e-02, 3.43270269e-02, 2.84730878e-02,
#                2.15949852e-02, 1.44861478e-02, 8.49372574e-03, 4.11463828e+00,
#                3.62782164e+00, 3.27545456e+00, 2.95361837e+00, 2.61563432e+00,
#                2.21133081e+00, 1.56059376e+00, 1.06533787e-02, 1.87590433e-02,
#                2.10060563e-02, 1.94248181e-02, 1.52489872e-02, 1.10263559e-02,
#                6.40993185e-03]


variables = symbols("x y")


# FUNCIONES PARA LA INTERPOLACION BICUBICA.

# función para las ecuaciones de la spline cubica
def biCubicSplineEquation():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + " * x^" + str(i) + " * y^" + str(j) + " + "
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubica de la derivada de x
def biCubicSplineEquationX():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(i) + " * x^" + str(i - 1) + " * y^" + str(j) + " + "
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubicade la derivada de y
def biCubicSplineEquationY():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(j) + " * x^" + str(i) + " * y^" + str(j - 1) + " + "
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para las ecuaciones de la spline cubica de xy
def biCubicSplineEquationXY():
    ecuacion = ""
    for i in range(0, 4):
        for j in range(0, 4):
            ecuacion += "a" + str(i) + str(j) + "*" + str(i) + "*" + str(j) + " * x^" + str(i - 1) + " * y^" + str(
                j - 1) + " + "
    ecuacion_simbolica = sympify(ecuacion[:-3])

    return ecuacion_simbolica

# función para generar el sistema de ecuaciones
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


# FUNCIONES PARA LAS DERIVADAS Y MALLA.
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


# print(calculateVectorConstants(0,0))

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
print(constants)

# calcula los nuevos puntos
def calculateNewPoints(i, x, y, variables, constants):
    ecuacion_spline = biCubicSplineEquation()
    constantes = calculateVectorConstants(0, 0)
    a = lambdify(symbols('a00 a10 a20 a30 a01 a11 a21 a31 a02 a12 a22 a32 a03 a13 a23 a33'), ecuacion_spline)(
        *constants[i])
    # ecuacion_spline = ecuacion_spline.subs({symbols(f'a{j}{i}'): constants[i][j] for i in range(4) for j in range(4)})
    nuevos_valores = lambdify(variables, a)(*[(2 * x + 1) / 2, (2 * y + 1) / 2])
    ecuacion_sustituida = ecuacion_spline.subs(
        {symbols(f'a{j}{k}'): constants[i][constantes.index(f'a{j}{k}')] for k in range(4) for j in range(4)})
    diagonal = ecuacion_sustituida.subs({symbols('x'): 0.5, symbols('y'): 0.5})
    abajo = ecuacion_sustituida.subs({symbols('x'): 0, symbols('y'): 0.5})
    derecha = ecuacion_sustituida.subs({symbols('x'): 0.5, symbols('y'): 0})

    return (derecha, abajo, diagonal)


print(calculateNewPoints(0, 1, 1, variables, constants))

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

"""
print(Vx)
print(Bder)

plt.imshow(Ader, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()

plt.imshow(Abaj, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()

plt.imshow(Adiag, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()
"""

print(len(Bder))
print(len(Babaj))
print(len(Bdiag))

"""
plt.imshow(Vx, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()
"""

def reemplazar_valores(matriz, vector):
    # Crear una copia de la matriz para evitar cambios en la original
    nueva_matriz = matriz.astype(float)

    # Obtener las posiciones de los valores -1 en la matriz
    posiciones = np.where(matriz == -1)

    print(posiciones)
    print(len(posiciones[0]))

    # Iterar sobre las posiciones y reemplazar los valores con los del vector
    i = 1

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (Vx[y, x] == -1):
                print(i)
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

"""
matriz_nueva = reemplazar_valores(Vx, resultsInX)
matriz_nueva2 = reemplazar_valores(Vx, Bder)

plt.imshow(matriz_nueva, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()

plt.imshow(matriz_nueva2, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()

print(np.array(calculateVectorConstants(1, 1)))
print(inverseMatrix)
print(np.array(calculateVectorFunctions(1, 1)))

print(IDvx)
print(inverseMatrix)

print(29)
print(np.array(calculateVectorFunctions(19, 9)))
print(np.array(calculateVectorConstants(1, 1)))
print(inverseMatrix @ calculateVectorFunctions(19, 9))
print(calculateConstants(19, 9, inverseMatrix))
print(calculateNewPoints(140, 19, 9, variables, constants))
print(np.array(calculateVectorConstants(0, 0)))
print(biCubicSplineEquation())
print(Vx)
"""
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

vectorTotal = intercalar_vectores(resultsInX, Bder, Babaj, Bdiag)
# print(vectorTotal)

print(len(vectorTotal))
# VALORES INICIALES.
# 141 * 4 = 564

nx =  40 #39
ny =  20 #18
# nx * ny = 702
nxc = 10 #11
nyc = 6 #6
# nxc * nyc * 2 = 132
# 702-132 = 570 
#  6 puntos 
print()
"""

[[  1   2   3   4   5   6   7   0   0   0   0   0   8   9  10  11  12  13 14]
[ 15  16  17  18  19  20  21   0   0   0   0   0  22  23  24  25  26  27 28]
[ 29  30  31  32  33  34  35   0   0   0   0   0  36  37  38  39  40  41 42]
[ 43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 61]
[ 62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79 80]
[ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98 99]
[100 101 102 103 104 105 106   0   0   0   0   0 107 108 109 110 111 112 113]
[114 115 116 117 118 119 120   0   0   0   0   0 121 122 123 124 125 126 127]
[128 129 130 131 132 133 134   0   0   0   0   0 135 136 137 138 139 140 141]]

"""
# 36 * 17 - (120
# nx*ny - (60 + nx*2 + (ny-2)*2) = 282
Vx = np.full((ny, nx), -1)
Vx[0, :] = 0
Vx[-1, :] = 0
Vx[:, -1] = 0
Vx[:, 0] = 1000

Vy = Vx.copy()
Vy[:, 0] = 0
print(Vx)
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

print(Vx)
#print(IDvx)
#print(reemplazar_valores(Vx, vectorTotal))



# plt.imshow(reemplazar_valores(Vx, vectorTotal), cmap='viridis', interpolation='nearest')
# plt.colorbar()

# plt.show()

# Tamaño de la matriz
# ny, nx = 9, 9
#
# # Matriz inicial
# matrix = np.zeros((ny, nx))
#
# # Vector de interpolación (cambiado para un solo punto por simplicidad)
# interpolation_vector = [5, -1, -1, -1]
#
# # Coordenadas del punto en la matriz
# x, y = 1, 1
#
# # Actualización de la matriz
# matrix[y:y+2, x:x+2] += np.array([[interpolation_vector[0], interpolation_vector[1]],
#                                   [interpolation_vector[2], interpolation_vector[3]]])
#
# print(matrix)
# print(vectorTotal)


# for y in range(0, matrix.shape[0] - 1, 2):
#     for x in range(0, matrix.shape[1] - 1, 2):
#         matrix[y:y+2, x:x+2] = np.array([[vectorTotal[0], vectorTotal[1]],
#                                           [vectorTotal[2], vectorTotal[3]]])
#
# print(matrix)

IDvx = np.empty((ny - 2, nx - 2), dtype=int)
valor = 1
for i in range(0, ny - 2):
    for j in range(0, nx - 2):
        if Vx[i + 1, j + 1] == -1:
            IDvx[i, j] = valor
            valor += 1
        else:
            IDvx[i, j] = 0

print(IDvx)

a = Vx.copy().astype(float)
j = 0
print(a)
for y in range(1, a.shape[0] - 1, 1):
    for x in range(1, a.shape[1] - 1, 1):
        if a[y, x] == -1:
            a[y:y+2, x:x+2] = np.array([[vectorTotal[0 + j], vectorTotal[1 + j]],
                                              [vectorTotal[2 + j], vectorTotal[3 + j]]], dtype=float)
            j += 4




plt.imshow(a, cmap='viridis', interpolation='nearest')
plt.colorbar()

plt.show()





print(a)
print(len(vectorTotal))
