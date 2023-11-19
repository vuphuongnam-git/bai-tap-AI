'''
	TEMPLATE FOR MACHINE LEARNING HOMEWORK
	AUTHOR Eric Eaton, Chris Clingerman
'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def evaluatePerformance():
	'''
	Evaluate the performance of decision trees,  
	averaged over 100 trials of 10-fold cross validation
	
	Return:
	  a matrix giving the performance that will contain:
	  stats[0,0] = mean accuracy of decision tree
	  stats[0,1] = std deviation of decision tree accuracy
	  stats[1,0] = mean accuracy of decision stump
	  stats[1,1] = std deviation of decision stump
	  stats[2,0] = mean accuracy of 3-level decision tree
	  stats[2,1] = std deviation of 3-level decision tree
	  
	** Note that your implementation must follow this API**
	'''

	# Load data
	filename = 'data/SPECTF.dat'
	data = np.loadtxt(filename, delimiter=',')
	X = data[:, 1:] 
	y = data[:, 0]

	# Define variables to store stats
	mean_dt_acc = []
	mean_ds_acc = []
	mean_dt3_acc = []

	std_dt_acc = []
	std_ds_acc = [] 
	std_dt3_acc = []

	percents = np.arange(0.1, 1.1, 0.1)
	# 100 trials
	for i in range(100):
		
		# Shuffle data
		idx = np.arange(X.shape[0])
		np.random.shuffle(idx)
		X = X[idx]
		y = y[idx]
		
		# Split data into 10 folds 
		X_folds = np.array_split(X, 10)
		y_folds = np.array_split(y, 10)
		
		# 10-fold cross validation
		dt_scores = []
		ds_scores = []
		dt3_scores = []
		
		for j in range(10):
			# Use fold j as test, others as train
			X_test = X_folds[j] 
			y_test = y_folds[j]
			
			X_train = np.concatenate(X_folds[:j] + X_folds[j+1:], axis=0) 
			y_train = np.concatenate(y_folds[:j] + y_folds[j+1:], axis=0)
			
			# Decision tree
			dt = tree.DecisionTreeClassifier()
			dt.fit(X_train, y_train)
			preds = dt.predict(X_test)
			acc = accuracy_score(y_test, preds)
			dt_scores.append(acc)
			
			# Decision stump
			ds = tree.DecisionTreeClassifier(max_depth=1)
			ds.fit(X_train, y_train)
			preds = ds.predict(X_test)
			acc = accuracy_score(y_test, preds)
			ds_scores.append(acc)
			
			# 3-level decision tree
			dt3 = tree.DecisionTreeClassifier(max_depth=3)
			dt3.fit(X_train, y_train)
			preds = dt3.predict(X_test)
			acc = accuracy_score(y_test, preds)
			dt3_scores.append(acc)
			
		# Compute stats for this trial        
		mean_dt_acc.append(np.mean(dt_scores))
		std_dt_acc.append(np.std(dt_scores))
		
		mean_ds_acc.append(np.mean(ds_scores))
		std_ds_acc.append(np.std(ds_scores))
		
		mean_dt3_acc.append(np.mean(dt3_scores))
		std_dt3_acc.append(np.std(dt3_scores))

	# Overall stats
	stats = np.zeros((3, 2))
	stats[0, 0] = np.mean(mean_dt_acc)
	stats[0, 1] = np.mean(std_dt_acc)
	
	stats[1, 0] = np.mean(mean_ds_acc)
	stats[1, 1] = np.mean(std_ds_acc)
	
	stats[2, 0] = np.mean(mean_dt3_acc)
	stats[2, 1] = np.mean(std_dt3_acc)
	
	return stats

# Do not modify from HERE...
if __name__ == "__main__":
	
	stats = evaluatePerformance()
	print("Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")")
	print("Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")")
	print("3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")")
# ...to HERE.