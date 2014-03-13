
Tool for training and evaluation of SVM with Radial-Basis Function (RBF) kernel.

Use svm_rbf_classifier.py to train and evaluate the performance of SVM classifier 
with RBF kernel over the specified dataset. Since testing can take very long for large 
datasets, the "evaluate" option is provided. By default the program will evaluate 
the classifier training and testing error after the training process is complete. 
However, if evaluate parameter is 0, the program will terminate after training is 
done and parallel_evalaute.py or parallel_prob_evalaute.py can be used for evaluation 
later with even more complete output. The probab parameter can be specified to make 
the classifier probabilistic. Note that top-1 accuracy of probabilistic classifiers 
might be a little bit lower than non-probabilistic classifier. Probabilistic 
classifiers are required to compute top-5 classification accuracy. After training 
is finished, the program will store the trained classifier on a file called: 
[Name of training]".LSVM"

Usage: python svm_rbf_classifier.py training testing C Gamma eval [probab]
Where
        training        = Path to training set
        testing         = Path to testing set
        C               = C parameter of the RBF SVM
        Gamma           = Gamma parameter of the RBF SVM
        eval            = Optional, will not evaluate if equal to 1
        probab          = Optional, make it a probabilistic classifier