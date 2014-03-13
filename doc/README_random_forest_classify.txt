
Tool for training and testing of Random Forest classifiers

Use random_forest_classify.py to train and test the performance of Random Forest classifiers 
over the given training and testing sets. Must specify the number of decision trees to include
on each forest and the maximum depth of each tree. Also, the current implementation of random 
forest used is included in the Scikit-Learn library. This implementation randomize the available 
features at each split, and the max_feats parameter is used to set the maximum number of features
to consider at each split. Two split criterion are available: Gini impurity and Entropy.

Since random forests introduce randomness on the training process, a single classifier might not be
enough to define the final performance of this type of classifier over the given dataset. The 
parameter times allows the user to define how many classifiers will be trained and tested and the
final mean of them is going to be used for the final metric. At the end, the system keeps the classifier
that achieved the highest global accuracy and stores it to a final named: [Name of training set]".best.RF".

Use n_jobs parameter to define the number of threads to use during the training process. if omitted, 
everything will be done on a single thread.

Usage: python random_forest_classify.py training_set testing_set N_trees max_D max_feats 
										type times [n_jobs]
Where
        training_set    = Path to the file of the training set
        testing_set     = Path to the file of the testing set
        N_trees         = Number of trees to use
        max_D           = Maximum Depth
        max_feats       = Maximum Features
        type            = Type of Decision trees (criterion for splits)
                                0 - Gini
                                1 - Entropy
        times           = Number of times to repeat experiments
        n_jobs          = Optional, number of parallel threads to use
		
