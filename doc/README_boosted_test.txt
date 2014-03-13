
Tool for evaluating AdaBoost C4.5 classifier

This tool evaluates the training and testing performance of an AdaBoost with C4.5 decision trees
classifier. It stores the final confusion matrix into a file. It can also save the failure cases
for the testing set in .SVG format if the file with the sources for each sample is present on 
current directory. Saving failure cases requires a directory called "output" to be created on 
the same directory where the tool is located.

Usage: python boosted_test.py classifier training_set testing_set [save_fail]
Where
        classifier      = Path to the .bc45 file that contains the classifier
        training_set    = Path to the file of the training set
        testing_set     = Path to the file of the testing set
        save_fail       = Optional, will output failure case if specified
		

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
		
	