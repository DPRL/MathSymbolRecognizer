
Tool for training and evaluation of C4.5 decision tree classifier.

Use train_c45.py to train and evaluate the performance of a single C4.5 
decision trees over the specified dataset. 

Usage: python train_c45.py training_set testing_set 
Where
        training_set    = Path to the file of the training set
        testing_set     = Path to the file of the testing set        
		
		
================================================		
	Notes 
================================================

- All tools that work with C4.5 decision trees require the AdaBoost library 
  to be located on the same directory. This is a shared library that can be 
  compiled from the provided source code using the following commands:
  
	Linux:
		gcc -shared -fPIC adaboost_c45.c -o adaboost_c45.so
	
	Windows (using MinGW):
		gcc -shared adaboost_c45.c -o adaboost_c45.so
		
	