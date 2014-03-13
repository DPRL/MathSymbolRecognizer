
Tool for Application of PCA parameters

The tool will take a file containing a dataset, a file containing PCA parameters 
(Normalization, Eigenvectors, Eigenvalues, Max PC to use) and will produce the projected version
of the dataset and store it in the specified file.

	Usage: python apply_PCA_parameters.py training_set PCA_params output
	Where
		training_set    = Path to the file of the training_set
		PCA_params      = Path to the file of the PCA parameters
        output  		= File to output the final dataset with reduced dimensionality


