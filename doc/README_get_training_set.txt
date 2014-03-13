
Tool used to produce a training set from INKML files

Use get_training_set.py tool to process and extract the features of the isolated 
symbols present in a set of INKML files. The system first extract all symbols found 
in the inkml files present in the given directory and then for each symbol it will 
extract the current set of features as defined in MathSymbol.py. Then, final dataset
ready to use for training will be stored in the specified output file.

Usage: python get_training_set.py inkml_path output
Where
        inkml_path      = Path to directory that contains the inkml files
        output          = File name of the output file
