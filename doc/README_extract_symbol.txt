
Tool to extract an specific symbol from an inkml file

Use extract_symbol.py to extract a symbol and store in .SVG format on the specified
output file. Requires the full path to the inkml_file where the desired symbol is 
located and the id of the symbol to extract from that file. This tool is useful to 
visualize special cases.

Usage: python extract_symbol.py inkml_file sym_id output
Where
        inkml_file      = Path to the inkml file that contains the symbol
        sym_id          = Id of the symbol to extract
        output          = File where the extracted symbol will be stored

		