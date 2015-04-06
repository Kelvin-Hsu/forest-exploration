import GaussianProcessModel
import numpy as np
from scipy import stats

# This creates the data matrix for multiple forests with characteristics 'characteristics' and trait values 'traits'
def formEcologyData(characteristics, traits):

	# This is the number of forest characteristics
	nChar = characteristics.shape[0]

	# This is the number of trait values we need for each forest characteristic
	nTrait = traits.shape[0]

	# Create our data matrix for these forests
	Xleft = np.repeat(characteristics, nTrait, axis = 0)
	Xright = np.tile(traits, nChar)[:, np.newaxis]
	Xv = np.concatenate((Xleft, Xright), axis = 1)
	return(Xv)

# This class is inherited from the Gaussian Process class
# It works the same way, but 
class ForestSelectionProcess(GaussianProcessModel.GaussianProcess):

	# Initialise the Selection
	# Note that it is necessary to initialise the data
	def __init__(self, X, y, kernel = None):

		# Initialise the Gaussian Process
		GaussianProcessModel.GaussianProcess.__init__(self, X, y, kernel = kernel)

		# The forest data must have three parameters
		if X.shape[1] != 3:
			raise ValueError('Forest data must have three parameters')
		
		# Those are the arrays of the virtual indices and corresponding trait lengths
		self.virtualIndices = np.array([])
		self.virtualTraitLengths = np.array([])

	# This adds multiple virtual forests, determined by their forest characteristics, with specific trait values, into our model
	def addMultipleVirtualForests(self, characteristics, traits):

		# This is the number of forest characteristics
		nChar = characteristics.shape[0]

		# This is the number of trait values we need for each forest characteristic
		nTrait = traits.shape[0]

		# Put the characteristic and traits into the standard data format
		Xv = formEcologyData(characteristics, traits)
		yv = self.predict(Xv)[0]
		
		# For our application, all fitness values are below zero, so we will hard-code this in
		yv[yv > 0] = 0

		# VirtualIndex is the starting index of the data vector of which we are inserting virtual data
		# For each pair of characteristics we added, we have one starting index and the number of trait
		# values it has 
		thisVirtualIndices = self.n + nTrait*np.arange(nChar)
		thisVirtualTraitLengths = np.repeat(nTrait, nChar)

		# Add it to the lists of all virtual indices
		self.virtualIndices = np.append(self.virtualIndices, thisVirtualIndices).astype(int)
		self.virtualTraitLengths = np.append(self.virtualTraitLengths, thisVirtualTraitLengths).astype(int)

		# Add the actual data
		self.addData(Xv, yv)

		# Return the virtual indices of the forests we have just added
		return(thisVirtualIndices)

	# This updates one virtual forest at a time given the virtual index
	def updateOneVirtualForest(self, virtualIndex, y):

		# Find out how many trait values there are for this forest
		nTrait = self.virtualTraitLengths[self.virtualIndices == virtualIndex][0]

		# The indices in the data vector corresponding to this forest is this
		indices = np.arange(virtualIndex, virtualIndex + nTrait)

		# So update the y values 
		self.y[indices] = y

		# We can now delete the virtual index corresponding to this forest
		mask = np.ones(self.virtualIndices.shape[0], dtype = bool)
		mask[self.virtualIndices == virtualIndex] = False
		self.virtualIndices = self.virtualIndices[mask]
		self.virtualTraitLengths = self.virtualTraitLengths[mask]

	# This updates one virtual forest at a time given the forest characteristics
	# Use this only if you cannot use the virtual index method since float comparisons may not be precise
	def updateOneVirtualForestCharacteristic(self, characteristic, y):

		# Find the places with the given characteristic and update the y values
		truth = (self.X[:, 0:2] == characteristic)
		truth = truth[:, 0]
		self.y[truth] = y

	def getCharacteristicFromVirtualIndex(self, virtualIndex):

		return(self.X[virtualIndex][0:2])

	# This computes how interesting a forest is given the trait values we are looking at
	def howInteresting(self, characteristic, traits):

		# Predict the behaviour of this forest
		Xq = formEcologyData(np.array([characteristic]), traits)
		(yqExp, yqVar) = self.predict(Xq)

		# metric = np.sum(np.sqrt(yqVar.diagonal()))
		# return(metric)

		# # Metric 4:
		# n = yqExp.shape[0]
		# zeros = np.zeros(n)

		# yqVars = yqVar.diagonal()
		# yqStd = np.sqrt(yqVars)
		# probs = 1 - stats.norm.cdf(zeros, yqExp, yqStd)

		# probs2 = probs**2
		# metric = probs2.dot(yqVars)
		
		# return(metric)


		# Metric 5:
		n = yqExp.shape[0]
		zeros = np.zeros(n)

		yqStd = np.sqrt(yqVar.diagonal())
		probs = 1 - stats.norm.cdf(zeros, yqExp, yqStd)

		metric = probs.dot(yqStd)
		
		# Return the computed metric
		return(metric)

	# This selects the most interesting forest based on the previous metric from a list of forests
	def selectBestForest(self, characteristics, traits):

		# Below is the standard maximisation algorithm that goes through all candidate characteristics
		bestInterestMetric = -np.inf

		for char in characteristics:

			interestMetric = self.howInteresting(char, traits)

			if interestMetric > bestInterestMetric:

				bestInterestMetric = interestMetric
				bestChar = char

			# VERBOSE
			# print('Characteristics:', char, '\tBest so far:', bestChar)

		return(bestChar)

	# This selects the 'topN' most interesting forests with data feedback
	def selectMultipleBestForests(self, characteristics, traits, topN = 1):

		# Initialise the best characteristics and places to store the correponding virtual indices
		bestChars = np.zeros((topN, 2))
		thisVirtualIndices = np.zeros(topN)

		# In each iteration
		for i in range(topN):

			# Find the current best forest
			bestCharsNow = self.selectBestForest(characteristics, traits)

			# Add the virtual forest and find out the virtual index
			thisVirtualIndicesNow = self.addMultipleVirtualForests(np.array([bestCharsNow]), traits)

			# Store our best characteristics and corresponding virtual index
			bestChars[i] = bestCharsNow
			thisVirtualIndices[i] = thisVirtualIndicesNow

			# VERBOSE
			# print('Chosen', i + 1, ':', bestCharsNow)

		# Return our results
		return(bestChars, thisVirtualIndices.astype(int))





















