import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist

# We will use these strings to denote the names of the hyperparameters involved in each kernel
s = ['Sensitivity']
l = ['Length Scale']

# For now, there are only two types of (stationary) kernels implemented: Isotropic (Scaled Identity Length Scale Matrix) and Anisotropic (Diagonal Length Scale Matrix)
ISOTROPIC = 0
ANISOTROPIC = 1

# This is just to avoid a few cycles of computation in the Matern 3/2 Kernel
SQRT3 = np.sqrt(3)

### Description of Kernel Implementation
### Each Kernel Function takes two parameters in Data Matrix Form (i.e. rows corresponds to data points or observations, and columns corresponds to feature dimensions)
### They return the corresponding kernel matrix from the two input data matrices
### Each function have a few properties/attributes, and they are
### 	k			: The number of hyperparameters
###		theta		: The vector of hyperparameters
###		thetaInitial: The vector of initial hyperparameters suggested (This is only used if the GP model was not assigned initial hyperparameters to start with - it will use this one instead)
###		name		: The name of the kernel
###		thetaNames	: A list of the names of the hyperparameters (e.g. ['Sensitivity', 'Length Scale 1', 'Length Scale 2', 'Length Scale 3'])
###		type		: Indicates the type of the kernel according to the above (e.g. ISISOTROPIC = 0, ANISOTROPIC = 1)

def se(X1, X2):

	return( (se.theta[0])**2 * np.exp( -0.5 * cdist(X1 / se.theta[1:], X2 / se.theta[1:], 'sqeuclidean') ) )

se.k = 1
se.theta = np.ones(1 + se.k)
se.thetaInitial = np.ones(1 + se.k)
se.name = 'Squared Exponential Kernel'
se.thetaNames = s + l
se.type = ANISOTROPIC

def m12(X1, X2):

	return( m12.theta[0]**2 * np.exp(-cdist(X1 / m12.theta[1:], X2 / m12.theta[1:], 'euclidean')) )

m12.k = 1
m12.theta = np.ones(1 + m12.k)
m12.thetaInitial = np.ones(1 + m12.k)
m12.name = 'Matern 1/2 Kernel'
m12.thetaNames = s + l
m12.type = ANISOTROPIC

def m32(X1, X2):
	
	a = SQRT3 * cdist(X1 / m32.theta[1:], X2 / m32.theta[1:], 'euclidean')

	return( m32.theta[0]**2 * (1 + a) * np.exp(-a) )

m32.k = 1
m32.theta = np.ones(1 + m32.k)
m32.thetaInitial = np.ones(1 + m32.k)
m32.name = 'Matern 3/2 Kernel'
m32.thetaNames = s + l
m32.type = ANISOTROPIC

def m52(X1, X2):

	a2 = 5 * cdist(X1 / m52.theta[1:], X2 / m52.theta[1:], 'sqeuclidean')

	a = np.sqrt(a2)

	return( m52.theta[0]**2 * (1 + a + a2/3) * np.exp(-a) )

m52.k = 1
m52.theta = np.ones(1 + m52.k)
m52.thetaInitial = np.ones(1 + m52.k)
m52.name = 'Matern 5/2 Kernel'
m52.thetaNames = s + l
m52.type = ANISOTROPIC

def m12iso(X1, X2):

	return( m12.theta[0]**2 * np.exp(-cdist(X1, X2, 'euclidean')/m12.theta[1]) )

m12iso.theta = np.ones(2)
m12iso.thetaInitial = np.ones(2)
m12iso.name = 'Matern 1/2 Isotropic Kernel'
m12iso.thetaNames = s + l
m12iso.type = ISOTROPIC

def m32iso(X1, X2):

	a = SQRT3 * cdist(X1, X2, 'euclidean')/m32.theta[1]

	return( m32.theta[0]**2 * (1 + a) * np.exp(-a) )

m32iso.theta = np.ones(2)
m32iso.thetaInitial = np.ones(2)
m32iso.name = 'Matern 3/2 Isotropic Kernel'
m32iso.thetaNames = s + l
m32iso.type = ISOTROPIC

def m52iso(X1, X2):

	a2 = 5 * cdist(X1, X2, 'euclidean')/(m52.theta[1]**2)

	a = np.sqrt(a2)

	return( m52.theta[0]**2 * (1 + a + a2/3) * np.exp(-a) )

m52iso.theta = np.ones(2)
m52iso.thetaInitial = np.ones(2)
m52iso.name = 'Matern 5/2 Isotropic Kernel'
m52iso.thetaNames = s + l
m52iso.type = ISOTROPIC


# Before the kernel is to be used, the number of hyperparameters must be set correctly
# Simply call this function with the kernel function and the number of features of your data matrix (note: NOT the number of hyperparameters)
def setNumberOfParams(ker, k):

	if ker.type == ANISOTROPIC:

		ker.k = k

		ker.theta = np.ones(k + 1)
		ker.thetaInitial = np.ones(k + 1)

		if k == 1:

			ker.thetaNames = s + l

		else:

			ls = []

			for i in range(k):
				ls += [l[0] + ' ' + str(i + 1)]

			ker.thetaNames = s + ls		
