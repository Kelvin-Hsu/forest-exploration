import numpy as np
import nlopt
import scipy.linalg as la
import kernels
import time

# This takes a covariance matrix and adds a certain noise level to the diagonal
# This is normally used to obtain the covariance of the queried function from the covariance of the latent function
def addObsNoise(fqVar, sigma = 0):

	nq = fqVar.shape[0]
	yqVar = fqVar + sigma**2 * np.diag(np.ones(nq))
	return(yqVar)

# This finds the lower cholesky decomposition of a matrix and provides jittering if needed
def choleskyjitter(A):
	
    try:
        return(la.cholesky(A, lower = True))
    except Exception:
        pass

    n = len(A)
    maxscale = 10*np.sum(A.diagonal())
    minscale = min(1/64, maxscale/1024)
    scale = minscale
    print('\t', 'Jittering...')

    while scale < maxscale:

        try:
        	jitA = scale * np.diag(np.random.rand(n))
        	L = la.cholesky(A + jitA, lower = True)
        	return(L)
        except Exception as e:
        	scale += minscale

    raise ValueError("Jittering failed")

# Here are some useful constants to be used in the Gaussian Process Class
INITIALISED = 0
LEARNED = 1
MISSING_KERNEL_MSG = "Please specify a kernel to use from kernels.py with 'GaussianProcess.setKernelFunction(kernelfunction)'."
HAS_NOT_LEARNED_MSG = "This Gaussian Process has not learned from the data yet. See 'GaussianProcess.learn()'."
WRONG_PARAM_LENGTH_MSG = "Incorrect length of parameters specified."
POSITIVE_SIGMA_REQUIRED_MSG = "Sigma must be non-negative."

# This is the Gaussian Process Class
class GaussianProcess:

	# Initialise the Gaussian Process
	# Note that it is necessary to initialise the data
	# Previously, I had 'None' as the default value of 'kernel',
	# but I think it may make more sense to just default it to a commonly used kernel,
	# such as Squared Exponential Kernel or Matern 3/2 Kernel
	# Here, I defaulted it to the Matern 3/2 Kernel
	# Okay, I changed it back to defaulting at 'None' afterall... 
	def __init__(self, X, y, kernel = None):

		# Set the data 
		self.setData(X, y)

		# Set the kernel function if it is provided
		if kernel != None:
			self.setKernelFunction(kernel)

		# Initial hyperparameters and noise level to start off from during learning
		self.initialParams = None
		self.initialSigma = None

		# Actual hyperparameters and noise level to be used for prediction
		self.kernelparams = None
		self.sigma = None

		# The matrix S is defined to be
		# 		S := K + sigma^2 * I
		# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
		# I have chosen to store this instead of K because it offers (slight) computational advantages
		# But I could easily have chosen to store K instead
		self.S = None

	# This sets the data of the Gaussian Process Model
	# This is called once automatically during the class initialisation
	# So, if you call it again, by implementation this means that you are overwriting the original data
	# By default, the class state will go back to being INITIALISED, even if it has LEARNED before
	# If the model has learned already and you would like to use the previous learning result (if you expect the previous model will work well with this new data), you can just set 'learned' to 'True'
	def setData(self, X, y, learned = False):

		# Set the data matrix X and observed output data vector y
		# We do some basic whitening here by just making sure our output data has zero mean
		# We only add the mean back during the prediction stage
		# Otherwise, all analysis is done on the slightly whitened data vector y
		self.X = X.copy()
		self.yMean = np.mean(y)
		self.y = y.copy() - self.yMean

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

		# If the user wants to use the previous learning result, let them do it
		# Otherwise, reset the class state to INITIALISED
		if learned == True:
			self.usePreviousLearningResult()
		else:
			self.state = INITIALISED

	# This adds new data on top of the original data into the model 
	def addData(self, X, y):

		# If the Gaussian Process Model has already been trained, we will use our fast cholesky update method to update the Cholesky Decomposition Matrix
		if self.state == LEARNED:

			# This is the number of observations of the incoming data
			nNew = X.shape[0]

			# First, we would have to update our S matrix
			# Again, the matrix S is defined to be
			# 		S := K + sigma^2 * I
			# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
			S = self.S.copy()
			Sn = self._kernel(self.X, X)
			Snn = self._kernel(X, X) + self.sigma**2 * np.eye(nNew)
			top = np.concatenate((S, Sn), axis = 1)
			bottom = np.concatenate((Sn.T, Snn), axis = 1)
			self.S = np.concatenate((top, bottom), axis = 0)

			# Now, we can use our fast cholesky update algorithm to find our lower cholesky decomposition without recomputing everything
			L = self.L.copy()
			Ln = la.solve_triangular(L, Sn, lower = True).T
			On = np.zeros(Ln.shape).T
			Lnn = choleskyjitter(Snn - Ln.dot(Ln.T))
			top = np.concatenate((L, On), axis = 1)
			bottom = np.concatenate((Ln, Lnn), axis = 1)
			self.L = np.concatenate((top, bottom), axis = 0)

		# In any case, we will just add the data into the model
		self.X = np.concatenate((self.X, X))

		# For the y data vector, we need to again make sure we slightly whiten the data to zero mean
		self.y += self.yMean
		self.y = np.concatenate((self.y, y))
		self.yMean = np.mean(self.y)
		self.y -= self.yMean

		# This is the number of observed data points and number of features from the data matrix
		(self.n, self.k) = np.shape(self.X)

		# We can initialise the appropriate identity matrix here for later use
		self.I = np.eye(self.n)

	# This function is only used by 'self.addData(X, y)' to retain the trained properties of the model from the last training session
	# This function can also be externally whenever the user wants to change the state of the model to 'learned' or 'trained'
	def usePreviousLearningResult(self):

		# We can only force the model state to be LEARNED it has actually been trained before
		if self.kernelparams != None & self.sigma != None:

			# If so, we would have to recompute the cholesky decomposition again
			self._prepareCholesky()
			self.state = LEARNED

		# Otherwise, this function will raise an error
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This function sets the kernel function that this gaussian process model would be using
	def setKernelFunction(self, kernelfunction):

		# This is a reference to the kernel function we will be using, which is used to call the function
		self._kernel = kernelfunction

		# We need to remember to set the correct number of feature dimensions for the kernel
		kernels.setNumberOfParams(self._kernel, self.k)

		# For convenience, we will store our own copy of the names of the kernel hyperparameters
		self.kernelParamNames = self._kernel.thetaNames

		# When we set a new kernel, any previous learning result should not really make much sense, so we would reset the model state to INITIALISED
		# However, if the user really wants to use a previous learning result, they are welcome to use 'self.usePreviousLearningResult()'
		self.state = INITIALISED

	# This function can be called before 'self.learn()' to set starting point of the hyperparameters of the kernel before training
	def setInitialKernelParams(self, params):

		# If no kernel has been specified, raise an error
		if self._kernel == None:
			raise ValueError(MISSING_KERNEL_MSG)

		# Just to be safe, we will make sure the number of hyperparameters have been set correctly
		kernels.setNumberOfParams(self._kernel, self.k)
		
		# If the number of initial parameters supplied does not match the numbers needed, raise an error
		if len(params) != len(self._kernel.thetaInitial):
			raise ValueError(WRONG_PARAM_LENGTH_MSG)
		
		# Otherwise, simply set the initial parameters
		self.initialParams = params.copy()

		# It is assumed that when the user calls this function, they would like to train the model soon, so we will start with an INITIALISED state
		self.state = INITIALISED

	# This function can be called before 'self.learn()' to set starting point of the noise level before training
	def setInitialSigma(self, sigma):

		# This needs to be a positive number, obviously
		if sigma < 0:
			raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

		# Simply set the starting noise level
		self.initialSigma = sigma

		# It is assumed that when the user calls this function, they would like to train the model soon, so we will start with an INITIALISED state
		self.state = INITIALISED

	# This function returns the current or lastly used starting point of the initial kernel hyperparameters
	def getInitialKernelParams(self):

		return(self.initialParams)

	# This function returns the current or lastly used starting point of the initial noise level
	def getInitialSigma(self):

		return(self.initialSigma)

	# This function returns the name of the kernel used
	def getKernelName(self):

		# If no kernel has been specified, raise an error
		if self._kernel == None:
			raise ValueError(MISSING_KERNEL_MSG)

		# Otherwise, just return the name of the kernel
		else:
			return(self._kernel.name)

	# This function returns the name of the kernel hyperparameters
	def getKernelParamNames(self):

		# If no kernel has been specified, raise an error
		if self._kernel == None:
			raise ValueError(MISSING_KERNEL_MSG)

		# Otherwise, just return the name of the kernel hyperparameters
		else:
			return(self.kernelParamNames)

	# This function can be called to directly set a particular training or learning result (hyperparameters and noise level) to the model
	# A common usage of this function is when the user already knows the appropriate hyperparameters and noise levels for experience, and want to skip the potentially lengthy training time
	def setLearningResult(self, params, sigma):

		# Make sure a kernel is already selected
		if self._kernel == None:
			raise ValueError(MISSING_KERNEL_MSG)

		# Just to be safe, we will make sure the number of hyperparameters have been set correctly
		kernels.setNumberOfParams(self._kernel, self.k)

		# Make sure the number of hyperparameters proposed is correct
		if len(params) != len(self._kernel.theta):
			raise ValueError(WRONG_PARAM_LENGTH_MSG)

		# Make sure the value of noise level proposed is valid
		if sigma < 0:
			raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

		# Set the learning results (hyperparameters)
		self.kernelParams = params.copy()
		self._kernel.theta = params.copy()

		# Set the learning results (noise level)
		self.sigma = sigma

		# Compute the lower cholesky decomposition for prediction purposes
		self._prepareCholesky()

		# We have finished 'learning'
		self.state = LEARNED
		
	# This function is the training part of the gaussian process model
	# The keyword 'sigma' can be set to a particular value so that the noise level is kept the same during learning and the hyperparameters will be trained to this particular noise level
	# The keyword 'sigmaMax' can be set to a particular value to make sure the gaussian process will never train to a noise level above this threshold
	def learn(self, sigma = None, sigmaMax = None):

		# VERBOSE
		np.set_printoptions(precision = 2)

		# If the model has already been trained before, we can start off our learning from these values
		if self.state == LEARNED:
			self.initialParams = self.kernelParams.copy()
			self.initialSigma = self.sigma

		# Obviously, the kernel needs to be specified beforehand
		if self._kernel == None:
			raise ValueError(MISSING_KERNEL_MSG)

		# If there is no initial starting hyperparameters provided, we will just use the default one from the kernels
		if self.initialParams == None:
			kernels.setNumberOfParams(self._kernel, self.k)

		# This function calculates the negative log(evidence) without the constant term
		def negLogEvidence(theta, grad):

			self._kernel.theta = theta[0:-1].copy()
			self.sigma = theta[-1]

			self._prepareCholesky()
			alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
			negLogEvidenceValue = 0.5 * self.y.dot(alpha) + np.sum(np.log(self.L.diagonal()))
			
			# VERBOSE
			if negLogEvidence.lastPrintedSec != time.gmtime().tm_sec:
				negLogEvidence.lastPrintedSec = time.gmtime().tm_sec
				print('\t', theta, negLogEvidenceValue)

			return(negLogEvidenceValue)

		# VERBOSE
		negLogEvidence.lastPrintedSec = -1

		# This function calculates the negative log(evidence) without the constant term while keeping sigma fixed
		def negLogEvidenceNoSigma(theta, grad):

			self._kernel.theta = theta.copy()

			self._prepareCholesky()
			alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
			negLogEvidenceValue = 0.5 * self.y.dot(alpha) + np.sum(np.log(self.L.diagonal()))
			
			# VERBOSE
			if negLogEvidenceNoSigma.lastPrintedSec != time.gmtime().tm_sec:
				negLogEvidenceNoSigma.lastPrintedSec = time.gmtime().tm_sec
				print('\t', theta, negLogEvidenceValue)

			return(negLogEvidenceValue)

		# VERBOSE
		negLogEvidenceNoSigma.lastPrintedSec = -1

		# This is the constant involved in the computation of the log(evidence)
		evidenceConst = (self.n * np.log(2 * np.pi) / 2)

		# If the user did not specify to fix the noise level, then do the following
		if sigma == None:

			# The number of parameters in the optimisation/learning stage is 1 more (with the noise level) then the number of hyperparameters in the kernel
			kparam = self._kernel.theta.shape[0] + 1

			# If the user did not specify a initial noise level to start from, default it to some arbitary value
			if self.initialSigma == None:
				sigmaInitial = np.std(self.y) * 1e-3

			# Otherwise, use the user-provided value
			else:
				sigmaInitial = self.initialSigma

		# Otherwise, if the user did specify to fix the noise level, then do the following
		else:

			# Make sure the value of noise level proposed is valid
			if sigma < 0:
				raise ValueError(POSITIVE_SIGMA_REQUIRED_MSG)

			# The number of parameters in the optimisation/learning stage is the number of hyperparameters in the kernel
			kparam = self._kernel.theta.shape[0]

			# Fix the noise level
			self.sigma = sigma

		# Initialise our optimiser with the right amount of parameters
		opt = nlopt.opt(nlopt.LN_BOBYQA, kparam)

		# Set a non-zero but small lower bound for computational safety
		opt.set_lower_bounds(1e-8 * np.ones(kparam))

		# Set an appropriate tolerance level
		opt.set_ftol_rel(1e-3)
		opt.set_xtol_rel(1e-3)

		# If the user did not specify initial parameters to use, use the default one from the kernel
		if self.initialParams == None:
			paramsInitial = self._kernel.thetaInitial.copy()

		# Otherwise, use the user-specified one
		else:
			paramsInitial = self.initialParams.copy()

		# If the user did not fix the noise level, do the following
		if sigma == None:

			# Set the upper bound for sigma if the user specified to
			if sigmaMax != None:
				ub = np.inf*np.ones(kparam)
				ub[-1] = sigmaMax
				opt.set_upper_bounds(ub)
			
			# Set the objective function
			opt.set_min_objective(negLogEvidence)

			# Set the initial parameters to be optimised
			thetaInitial = np.append(paramsInitial, sigmaInitial)

			# VERBOSE
			print('Params:\t', self.getKernelParamNames())
			print('Format:\t [(--kernel-parameters--) (sigma)] negative(log(evidence) + ' + str(round(evidenceConst, 3)) + ')')

			# Obtain the final optimal parameters
			thetaFinal = opt.optimize(thetaInitial)

			# Obtain the optimal hyperparameters
			self._kernel.theta = thetaFinal[0:-1].copy()
			self.kernelParams = self._kernel.theta.copy()

			# Obtain the optimal noise level
			self.sigma = thetaFinal[-1]

			# Store the final log(evidence) value
			self.logevidence = -negLogEvidence(thetaFinal, thetaFinal) - evidenceConst

		else:

			# Set the objective function
			opt.set_min_objective(negLogEvidenceNoSigma)

			# VERBOSE
			print('Sigma has been set to', sigma)
			print('Params:\t', self.getKernelParamNames())
			print('Format:\t [(--kernel-parameters--) (sigma)] negative(log(evidence) + ' + str(round(evidenceConst, 3)) + ')')

			# Obtain the final optimal hyperparameters
			self._kernel.theta = opt.optimize(paramsInitial)
			self.kernelParams = self._kernel.theta.copy()

			# Store the final log(evidence) value
			self.logevidence = -negLogEvidenceNoSigma(thetaFinal, thetaFinal) - evidenceConst

		# Compute and prepare the lower cholesky decomposition matrix
		self._prepareCholesky()

		# We have finished learning
		self.state = LEARNED

		# VERBOSE
		print('Learned:\t Kernel HyperParameters:', self.kernelParams, '| Sigma:', self.sigma, '| log(evidence):', self.logevidence)

	# This function is the prediction part of the gaussian process model
	# Use this on a query data matrix to find either the predicted expected latent function and its covariance or the expected output function and its covariance
	# By default, the covariance corresponds to the latent function
	# To set it to be corresponding to the output function, set 'obs' to 'True'
	def predict(self, Xq, obs = False):

		# If the model has not learned, we should not predict anything
		self.requireLearned()

		# Set the learned result
		self._kernel.theta = self.kernelParams.copy()

		# Compute the prediction using the standard Gaussian Process Prediction Algorithm
		alpha = la.solve_triangular(self.L.T, la.solve_triangular(self.L, self.y, lower = True, check_finite = False), check_finite = False)
		Kq = self._kernel(self.X, Xq)
		fqExp = np.dot(Kq.T, alpha) + self.yMean
		v = la.solve_triangular(self.L, Kq, lower = True, check_finite = False)
		Kqq = self._kernel(Xq, Xq)
		fqVar = Kqq - np.dot(v.T, v)

		# Depending on if the user wants the one for latent function or output function, return the correct expected values and covariance matrix
		if obs == True:
			return(fqExp, addObsNoise(fqVar, self.sigma))
		else:
			return(fqExp, fqVar)

	# This returns the hyperparameters of the kernel if the model has been trained
	def getKernelParams(self):

		if self.state == LEARNED:
			return(self.kernelParams)
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the noise level if the model has been trained
	def getSigma(self):

		if self.state == LEARNED:
			return(self.sigma)
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the log(evidence) if the model has been trained
	def getLogEvidence(self):

		if self.state == LEARNED:
			try:
				return(self.logevidence)
			except AttributeError:
				return(None)
		else:
			raise ValueError(HAS_NOT_LEARNED_MSG)

	# This returns the raw state of the model in raw values
	# INITIALISED = 0
	# LEARNED = 1
	def getState(self):

		return(self.state)

	# This returns the state of the model in binary truth values
	def hasLearned(self):

		if self.state == LEARNED:
			return(True)
		else:
			return(False)

	# For debugging purposes, you can use this to stop the program at places where you require the gaussian process model to have been trained before proceeding
	def requireLearned(self):

		if self.state != LEARNED:
			raise ValueError(HAS_NOT_LEARNED_MSG)
			
	## --Below are private methods--

	# This prepares the lower cholesky decomposition of S
	# Again, the matrix S is defined to be
	# 		S := K + sigma^2 * I
	# where K is the Data Kernel Matrix, sigma is the noise level, and I is the identity matrix
	def _prepareCholesky(self):

		self.S = self._kernel(self.X, self.X) + self.sigma**2 * self.I
		self.L = choleskyjitter(self.S)

