"""
    DPRL Math Symbol Recognizers 
    Copyright (c) 2012-2014 Kenny Davila, Richard Zanibbi

    This file is part of DPRL Math Symbol Recognizers.

    DPRL Math Symbol Recognizers is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DPRL Math Symbol Recognizers is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DPRL Math Symbol Recognizers.  If not, see <http://www.gnu.org/licenses/>.

    Contact:
        - Kenny Davila: kxd7282@rit.edu
        - Richard Zanibbi: rlaz@cs.rit.edu 
"""
import sys
from load_inkml import *

#=====================================================================
#  Extract a symbol from a given inkml file using a sym id, and then
#  outputs the symbol to a SVG file
#
#  Created by:
#      - Kenny Davila (Feb 11, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 11, 2012-2014)
#
#=====================================================================

def output_sample( file_path, sym_id, out_path ):
    #load file...
    symbols = load_inkml(file_path, True)

    #find symbol ...
    for symbol in symbols:
        if symbol.id == sym_id:
            print("Symbol found, class: " + symbol.truth)
            symbol.saveAsSVG( out_path )

            return True

    return False

def main():
    #usage check
    if len(sys.argv) != 4:
        print("Usage: python extract_symbol.py inkml_file sym_id output")
        print("Where")
        print("\tinkml_file\t= Path to the inkml file that contains the symbol")
        print("\tsym_id\t\t= Id of the symbol to extract")
        print("\toutput\t\t= File where the extracted symbol will be stored")
        return

    file_path = sys.argv[1]
    sym_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if output_sample(file_path, sym_id, output_path):
        print("Sample extracted successfully!")
    else:
        print("Sample was not found in the given file!")

main()
