
Tool used for expansion of a dataset

Use get_enhanced_clustered_set.py to expand a dataset using synthetic data. Two ways of expansion are
provided, one for under-represented classes and the second for over-represented classes. The min_prc
parameter defines a threshold that separates the under_represented classes from the over-represented. 
Note that min_prc also defines the minimum representation per class based on the size of the largest class.
For example, if the largest class has 5,000 samples and min_prc is set to 0.20, then all classes will have
at least 20% x 5,0000 = 1,000 samples. For over-represented classes clustering is applied, and then the 
clust_prc parameter is used to defined the minimum final size for each cluster per class based on the size 
of largest cluster per each class. 

Usage: python get_enhanced_clustered_set.py inkml_path output min_prc diag_dist
											max_clusters clust_prc [verbose] [count_only]
Where
        inkml_path      = Path to directory that contains the inkml files
        output          = File name of the output file
        min_prc         = Minimum representation based on (%) of largest class
        diag_dist       = Distortion factor relative to length of main diagonal
        max_clusters    = Maximum number of clusters per large class
        clust_prc       = Minimum cluster size based on (%) of largest
        verbose 		= Optional, print detailed messages
        count_only      = Will only count what will be the final size of dataset
	
	
================================================		
	Notes 
================================================

- All tools that work with synthetic data generation require the distorter library 
  to be located on the same directory. This is a shared library that can be compiled from
  the provided source code using the following commands:
  
	Linux:
		gcc -shared -fPIC distorter_lib.c -o distorter_lib.so
	
	Windows (using MinGW):
		gcc -shared distorter_lib.c -o distorter_lib.so
		
- To only expand data without using the clustering option for large classes
  use max_clusters = 1 and clust_prc = 0.0. 