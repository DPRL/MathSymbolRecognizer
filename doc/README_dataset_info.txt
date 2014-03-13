
Tool to extract general information from dataset file

Use dataset_info.py to compute the number of classes present in dataset file and 
the number of samples per class. It also shows the number of current attributes.
An histogram of the class representation is built using the specified number of 
bins (10 by default).

Usage: python dataset_info.py dataset [n_bins]
Where
        dataset = Path to file that contains the data set
        n_bins  = Optional, number of bins for histogram of class representation
		
