
Tool for count and extraction of data from common classes between datasets

Use count_common.py to analyze sets of common labels between two given datasets.
If specified, the program can extract the data of the common classes from the second
dataset and store it in a new file called: [Name of dataset 2]".common.txt".

This tool can be used for example to extract data from CROHME 2013 dataset that
have the same labels as a CROHME 2012 dataset.


Usage: python count_common.py dataset_1 dataset_2 extract
Where
        dataset_1       = Path to the file of the first dataset
        dataset_2       = Path to the file of the second dataset
        extract 		= Extract common samples from second dataset
		
		