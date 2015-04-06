import numpy as np

# This files contains some convenient functions to read ecologyd data from the .csv file
# It is so far not documented yet (research code), but it should be quite straight forward to guess what they do for now

# Read the data
ecologyData = np.genfromtxt('ecologyData.csv', delimiter = ',')

# Remove any NaN entries
ecologyData = ecologyData[~np.isnan(ecologyData).any(axis = 1)]

# Extract the relevant data columns
ID = ecologyData[:, 1]
timeDisturbance = ecologyData[:, 2]
B4 = ecologyData[:, 3]
trait = ecologyData[:, 4]
fitness = ecologyData[:, 5]

def getForestTraitFitness(IDvalue):

    x = trait[ID == IDvalue]
    y = fitness[ID == IDvalue]

    return(x, y)

def getEcologyData(logtimedisturbance = False):

    x1 = np.array([timeDisturbance]).T

    if logtimedisturbance == True:
        x1 = np.log(x1)

    x2 = np.array([B4]).T
    x3 = np.array([trait]).T

    X = np.concatenate((x1, x2, x3), axis = 1)
    y = np.array(fitness)

    return(X, y)

def getForests(forestIDs, logtimedisturbance = False):

    (X, y) = getEcologyData(logtimedisturbance = logtimedisturbance)

    forestID = forestIDs[0]

    idf = ID[ID == forestID].copy()
    Xf = X[ID == forestID, :].copy()
    yf = y[ID == forestID].copy()

    for forestID in forestIDs[1:]:

        idf = np.append(idf, ID[ID == forestID])
        Xf = np.concatenate((Xf, X[ID == forestID, :]), axis = 0)
        yf = np.append(yf, y[ID == forestID])

    return(Xf, yf, idf.astype(int))

def getForestCharacteristics(forestIDs, logtimedisturbance = False):

    (X, y) = getEcologyData(logtimedisturbance = logtimedisturbance)

    x1 = X[:, 0]
    x2 = X[:, 1]

    x1u = np.array([])
    x2u = np.array([])
    idu = np.array([])

    for forestID in forestIDs:

        try:
            x1u = np.append(x1u, x1[ID == forestID][0])
            x2u = np.append(x2u, x2[ID == forestID][0])
            idu = np.append(idu, forestID)
        except IndexError:
            pass

    forestCharacteristics = np.concatenate((x1u[:, np.newaxis], x2u[:, np.newaxis]), axis = 1)

    return(forestCharacteristics, idu.astype(int))

def generateQueryPoints(forestIDs, logtimedisturbance = False, points = 50):

    (X, y) = getEcologyData(logtimedisturbance = logtimedisturbance)

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    i = 0
    forestID = forestIDs[i]

    while forestID not in ID:

        i += 1
        forestID = forestIDs[i]


    idf = ID[ID == forestID].copy()
    x1f = x1[ID == forestID].copy()
    x2f = x2[ID == forestID].copy()
    x3f = x3[ID == forestID].copy()

    idv = idf[0]
    x1v = x1f[0]
    x2v = x2f[0]
    x3fmin = min(x3f)
    x3fmax = max(x3f)

    x3f = np.linspace(x3fmin, x3fmax, num = points)

    idf = idv * np.ones(points)
    x1f = x1v * np.ones(points)
    x2f = x2v * np.ones(points)

    x1f = np.array([x1f]).T
    x2f = np.array([x2f]).T
    x3f = np.array([x3f]).T

    idq = idf.copy()
    Xq = np.concatenate((x1f, x2f, x3f), axis = 1)

    for forestID in forestIDs[(i + 1):]:

        if forestID not in ID:

            continue

        idf = ID[ID == forestID].copy()
        x1f = x1[ID == forestID].copy()
        x2f = x2[ID == forestID].copy()
        x3f = x3[ID == forestID].copy()

        idv = idf[0]
        x1v = x1f[0]
        x2v = x2f[0]
        x3fmin = min(x3f)
        x3fmax = max(x3f)

        x3f = np.linspace(x3fmin, x3fmax, num = points)

        idf = idv * np.ones(points)
        x1f = x1v * np.ones(points)
        x2f = x2v * np.ones(points)

        idf = np.array([idf]).T
        x1f = np.array([x1f]).T
        x2f = np.array([x2f]).T
        x3f = np.array([x3f]).T

        idq = np.append(idq, idf)
        Xqf = np.concatenate((x1f, x2f, x3f), axis = 1)
        Xq = np.concatenate((Xq, Xqf), axis = 0)

    return(Xq, idq.astype(int))

