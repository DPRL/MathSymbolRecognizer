
Tool for correcting labels in CROHME datasets

CROHME 2013 datasets contain more labels than classes as defined by the competition. To fix this
issue use this tool to automatically correct the additional labels to the set of 101 valid classes.

Usage: python correct_labels.py training_set output_set
Where
        training_set    = Path to the file of the training_set
        output_set      = File to output file with corrected labels

		