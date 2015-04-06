import numpy as np
import pandas as pd
import ForestSelection
import kernels
import ecologydata as ecodata
from scipy.spatial.distance import cdist
import multiprocessing as mp
import time

# Run this on a computer in which the 'tree' package is installed
import tree

#################################################################################################### 
# Program starts here
# Note: I got rid of all the printing and plotting steps in this script, just to simplify things. We can add 
# it back later on if we want to.

# To save time, this function can remove the candidate forests nearby the training forests within a radius of 0.05
# This is so that our program avoid checking forests that we are already sure we don't want to check anyway
def removeClosebyCandidates(candidateCharacteristics, trainingCharacteristics):

	distsq = cdist(candidateCharacteristics, trainingCharacteristics, 'sqeuclidean')
	(c, t) = distsq.shape
	distmask = np.ones((c, t), dtype = bool)
	distmask[distsq < 0.05**2] = False
	charmask = np.ones(c, dtype = bool) 
	charmask[distmask.sum(axis = 1) < t] = False
	return(candidateCharacteristics[charmask])

# This is the job that simulates a particular forest
# NOTE: characteristic contains log(TD) and B4, so this needs to be scaled properly before we pass it to the simulator
# When we get the fitness, we need to log it again since our model is using log(fitness)
def simulate(characteristic, traits, virtualIndex):

	p1 = np.exp(characteristic[0])
	p2 = characteristic[1]
	forest = tree.TreeModel(p1, p2)
	forest.evolve(100)
	fitness = forest.fitness(traits)
	return(np.log(fitness), virtualIndex)

# This is used for logging the results of each simulations
resultList = []
def logResult(result):
    resultList.append(result)

# This displays how many jobs we are currently running
def displayCurrentNumberOfJobs(currentNumberOfJobs):

	if currentNumberOfJobs == 0:
		print('no job running')
	elif currentNumberOfJobs == 1:
		print(currentNumberOfJobs, 'job running')
	else:
		print(currentNumberOfJobs, 'jobs running')

def main():    

	global resultList

	# Start the clock (this is just for printing purposes)
	time.clock()
	time.sleep(1)

	# This is the total number of jobs we would like to do
	TOTALJOBS = 5

	### Initialisation stage
	# We will be using the Matern 3/2 Kernel
	kernelchoice = kernels.m32

	# This is the maximum number of training forests we will start off with
	numberOfTrainingForests = 10
	numberOfCandidateForests = 500

	# We will be using these trait values for now
	traits = np.linspace(-5, 0, num = 50)

	### Set Randomisation and Generation Stage
	# Permute all the possible forest IDs to randomise the training forest choices
	allForestIDsPermutated = np.random.permutation(int(np.max(ecodata.ID)))

	# Determining training set
	trainingForestIDs = allForestIDsPermutated[:numberOfTrainingForests]

	# Getting our actual training set
	(Xt, yt, idt) = ecodata.getForests(trainingForestIDs, logtimedisturbance = True)
	actualTrainingForestIDs = pd.unique(idt)
	actualNumberOfTrainingForests = actualTrainingForestIDs.shape[0]
	actualNumberOfTrainingDataPoints = idt.shape[0]

	# Obtain the forest characteristics of the training set
	trainingCharacteristics = ecodata.getForestCharacteristics(trainingForestIDs, logtimedisturbance = True)[0]

	# Initialise the Gaussian Process Forest Selection Class
	gpfs = ForestSelection.ForestSelectionProcess(Xt, yt, kernel = kernelchoice)

	# Train the model, starting with the following parameters to maybe speed up the initial training stage
	initialParams = np.array([5.2548212808, 6.1294312135, 0.8751383158, 1.4410416625])
	initialSigma = 3.51958264701e-05
	gpfs.setInitialKernelParams(initialParams)
	gpfs.setInitialSigma(initialSigma)
	gpfs.learn()

	# So far we have not allocated any jobs
	currentNumberOfJobs = 0

	# This is the last time we displayed anything (only for printing purposes)
	displaytime = 0

	# Set up parallel job
	pool = mp.Pool()

	### Selection Stage
	while True:

		# When we can, show the number of jobs currently running
		if time.clock() > displaytime + 1:

			displayCurrentNumberOfJobs(currentNumberOfJobs)
			displaytime = time.clock()

		# Fill the queue if not filled
		if currentNumberOfJobs <= TOTALJOBS:

			# Re-Generate 500 candidate forests randomly where logTD is in [0, 5] and B4 is in [0, 3]
			candidateCharacteristicsAll = np.random.rand(500, 2)
			candidateCharacteristicsAll *= np.array([5, 3])

			# Remove candidates nearby the training forests to save some time
			candidateCharacteristics = removeClosebyCandidates(candidateCharacteristicsAll, trainingCharacteristics)

			# Choose the best forest characteristics to look at 
			(oneBestChars, oneVirtualIndices) = gpfs.selectMultipleBestForests(candidateCharacteristics, traits, topN = 1)
			print('\t\t\tSelected:', oneBestChars[0])
			pool.apply_async(simulate, args = (oneBestChars[0], traits, oneVirtualIndices[0]), callback = logResult)
			currentNumberOfJobs += 1

		# Update the model if a new result has finished simulating
		if len(resultList) > 0:

			# One job has finished
			currentNumberOfJobs -= 1

			# Grab the first result
			result = resultList[0]

			# Those are the results of our simulation
			yActual = result[0]
			finishedVirtualIndex = result[1]

			charBeingUpdated = gpfs.getCharacteristicFromVirtualIndex(finishedVirtualIndex)

			# Update our model
			print('\t\t\t\t\t\t\tUpdating Forest Characteristic:', charBeingUpdated)
			gpfs.updateOneVirtualForest(finishedVirtualIndex, yActual)
			print('\t\t\t\t\t\t\tFinished Updating Forest [' + str(len(resultList) - 1) + ' more to update]')

			# We can delete this result now
			resultList = resultList[1:]

	print('_____')


if __name__ == "__main__":
	main()
