
Tool for extraction of PCA parameters

The tool will take a file containing a dataset and will compute the corresponding
PCA parameters (Normalization, Eigenvectors, Eigenvalues, max number of PCA).
and will store them to a file. The user can specify a maximum number of PCA to use
and a maximum value of variance, the program will use the smallest of these two
as the final number of principal components to store.

Usage: python get_PCA_parameters.py training_set output_params var_max k_max
Where
        training_set    = Path to the file of the training_set
        output_params   = File to output PCA preprocessing parameters
        var_max         = Maximum percentage of variance to add
        k_max           = Maximum number of Principal Components to Add
