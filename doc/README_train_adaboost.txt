
Tool for training and evaluation of AdaBoost with C4.5 decision trees classifier.

Use train_adaboost.py to train and evaluate the performance of AdaBoost with C4.5 
decision trees over the specified dataset. After training is finished, 
the program will store the trained classifier on a file called: 
[Name of training]".BC45"

Usage: python train_adaboost.py training_set testing_set rounds
Where
        training_set    = Path to the file of the training set
        testing_set     = Path to the file of the testing set
        rounds          = Rounds to use for AdaBoost
        

================================================		
	Notes 
================================================

- All tools that work with AdaBoost require the AdaBoost library to be located 
  on the same directory. This is a shared library that can be compiled from
  the provided source code using the following commands:
  
	Linux:
		gcc -shared -fPIC adaboost_c45.c -o adaboost_c45.so
	
	Windows (using MinGW):
		gcc -shared adaboost_c45.c -o adaboost_c45.so
		
	