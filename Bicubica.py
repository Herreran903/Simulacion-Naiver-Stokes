import numpy as np
from sympy import symbols, diff, sympify, lambdify
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
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

# resultsInX = [-14.999998830386438, -3.7079811021634157e-06, 8.609204231291301e-06,
#               -1.6144968261633252e-05, 2.680995323034452e-05, -3.597223832774166e-05,
#               3.526772140943768e-05, -7.591842818620818e-06, 3.028659589564861e-05,
#               -7.923916344560952e-05, 0.0001635404102375104, -0.0002773530042174425,
#               0.0003794425153303288, -0.0003626605059799202, -15.00000377858996,
#               1.235863134940729e-05, -2.785767590873789e-05, 5.3998069876382004e-05,
#               -8.736133999254214e-05, 0.0001220591378964242, -0.00011815462208208949,
#               1.6882400386919114e-05, -8.56105909562482e-05, 0.0002471262355370112,
#               -0.0005244137209555856, 0.0008997988288914332, -0.0012415242074238042,
#               0.0011837442904669613, -14.999990915723592, -2.8828338237030238e-05,
#               6.713338875874442e-05, -0.00012661385379120205, 0.0002130491666101353,
#               -0.00029440114802792294, 0.0003096629003673801, 1.261438808552312e-05,
#               0.00014599201110103086, -0.000529196434352508, 0.001201317087405944,
#               -0.002103440524198764, 0.0029171430692942445, -0.0028021546225583185,
#               -15.000018036271038, 5.909233141299097e-05, -0.0001335926191239345,
#               0.00026096841969014406, -0.0004299656268269558, 0.0006336938504534206,
#               -0.0007495631609826248, 0.0006786549983583967, -0.0006190536058240381,
#               0.0005874711590238557, -0.0005296793871631678, 0.0004618688400942761,
#               -0.00029613415648648, -0.00010250514552271467, 0.0009510264720193062,
#               -0.0023366175284189952, 0.004193527071755507, -0.00587536759779839,
#               0.005637856606814999, -14.99996702163197, -0.00010472210238177126,
#               0.0002445435576583499, -0.000463644384133044, 0.0007920373509109346,
#               -0.0011455929385086883, 0.0014605758877748176, -0.001494059380485524,
#               0.0014890045179598577, -0.0013968871800669747, 0.001313052603036713,
#               - 0.0011098858479659801, 0.0007685963647603711, 5.8083779318235385e-05,
#               - 0.0016005276207403385, 0.004170888707234095, -0.007548555723252741,
#               0.010594117299031735, -0.010218061377903185, -15.000052766596646,
#               0.00017297860102516438, -0.0003910314882862309, 0.000764573830610793,
#               - 0.001260333401578606, 0.0018610490256079082, -0.002207309277163283,
#               0.002013224117478618, -0.0018446648861259463, 0.0017555845357958288,
#               -0.0015854905387161222, 0.0013847876768067202, -0.0008940829573904768,
#               -0.0002879316534304399, 0.0028083254407975516, -0.006925271634789162,
#               0.01244549257241596, -0.017446237729398233, 0.01674552825192939,
#               -14.999922001779805, -0.0002474090995230327, 0.000576905141321582,
#               -0.0010882044775679667, 0.0018349347845787456, -0.0025391652400070797,
#               0.0026812859818194923, 0.00013062770532386997, 0.0012586824302476401,
#               - 0.00463656399490598, 0.010575494901362552, -0.018550882962021717,
#               0.02575196521174003, -0.024746703084205082, -15.000095508662774,
#               0.0003128282318565212, -0.0007048923352783731, 0.0013691664391171088,
#               - 0.0022161722125129124, 0.0031056202230432817, -0.0030107026165644424,
#               0.00042593553193513885, -0.0022150289218658935, 0.0064542817454888435,
#               - 0.01374803277551278, 0.02363845127946177, -0.03264781825993581,
#               0.031145255317260114, -14.9999122312879, -0.0002780214019710847,
#               0.0006467760258246358, -0.001212743588161036, 0.0020192069482263264,
#               - 0.0027109346970390913, 0.0026658014685647657, -0.0005826960848218689,
#               0.002350153232016275, -0.006183826469338932, 0.012808100183304964,
#               - 0.021760555412184128, 0.029803251278648853, -0.028498944316468017]
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

nx =  41 #39
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
Vx[:, 0] = 5

Vy = Vx.copy()
Vy[:, 0] = 0
print(Vx)
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

#print(Vx)
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
