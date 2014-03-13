
Tool for evaluation of classifier performance in parallel

Use parallel_evaluate.py to evaluate the performance of a given classifier over
training and testing sets in parallel threads for shorter evaluation time.
Currently, this tool only supports classifiers from the Scikit-learn library like
SVC (support vector classifier) and Random Forests. It does not support AdaBoost with
C4.5 classifier. For AdaBoost C4.5 classifier use "boosted_test.py".

The tool computes global and per class accuracy with the corresponding confusion matrix. If 
specified, the tool computes the the global and per class accuracy with confusion matrix for
the case where errors between ambiguous classes are ignored. The list of ambiguous classes
simply contains pairs of symbols considered ambiguous, one pair per line. Separate the 
symbols on each line with comma.


Usage: python parallel_evaluate.py training_set testing_set classifier normalize
								   workers [test_only] [ambiguous]
Where
        training_set    = Path to the file of the training set
        testing_set     = Path to the file of the testing set
        classifier      = File that contains the pickled classifier
        normalized      = Whether training data was normalized prior training
        workers         = Number of parallel threads to use
        test_only       = Optional, Only execute for testing set
        ambiguous       = Optional, file that contains the list of ambiguous
		

================================================		
	Notes 
================================================

- For SVM classifiers remember to use the "normalized" option as 1.