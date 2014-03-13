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
import os
import sys
import fnmatch
import string
from traceInfo import *
from mathSymbol import *
from load_inkml import *

#=====================================================================
#  generates a training set from a directory containing the inkml
#
#  Created by:
#      - Kenny Davila (Oct, 2012)
#  Modified By:
#      - Kenny Davila (Oct 19, 2012)
#      - Kenny Davila (Nov 25, 2013)
#         - Added ID to symbol
#         - Added AUX file to output origin for each sample in DS
#      - Kenny Davila (Jan 16, 2012-2014)
#         - Print number of attributes found
#
#=====================================================================
 
def main():
    #usage check
    if len(sys.argv) != 3:
        print("Usage: python get_training_set.py inkml_path output")
        print("Where")
        print("\tinkml_path\t= Path to directory that contains the inkml files")
        print("\toutput\t\t= File name of the output file")
        return
    
    #load and filter the list of files, the result is a list of inkml files only
    try:
        complete_list = os.listdir(sys.argv[1])
        filtered_list = []
        for file in complete_list:
            if fnmatch.fnmatch(file, '*.inkml'):
                filtered_list.append( file )
    except:
        print( "The inkml path <" + sys.argv[1] + "> is invalid!" )
        return

    samples = []
    labels_found = {}
    sources = []
    #read every file in the path specified...        
    for i in range(len(filtered_list)):
        file_name = filtered_list[i]
        file_path = sys.argv[1] + '//' + file_name;
        advance = float(i) / len(filtered_list)
        print(("Processing => {:.2%} => "  + file_path).format( advance ))
        
        symbols = load_inkml( file_path, True )                    
            
        for new_symbol in symbols:
            #now generate the features and add them to the list, including the tag
            #for the expected class....
            sample = new_symbol.getFeatures() + [ new_symbol.truth ]        
            samples.append( sample )

            #count samples per class
            if not new_symbol.truth in labels_found:
                labels_found[ new_symbol.truth ] = 1
            else:
                labels_found[ new_symbol.truth ] += 1

            #the source of current symbol will be
            #exported as auxiliary file
            sources.append( ( file_path, new_symbol.id) )

            
    
    print( "Found: " + str(len(labels_found.keys())) + " different classes" )
    if len(samples) > 0:
        print( "Found: " + str(len(samples[0]) - 1 ) + " different attributes" )

    print "Saving main .... "
    #now that all the samples have been collected, write them all 
    #in the output file
    try:
        file = open(sys.argv[2], 'w')
    except:
        print( "File <" + sys.argv[2] + "> could not be created")
        return
    
    content = ''
    #print as headers the types for each feature...
    feature_types = new_symbol.getFeaturesTypes()
    for i, feat_type in enumerate(feature_types):
        if i > 0:
            content += '; '
        content += feat_type
    content += '\r\n'   
        
    for sample in samples:
        line = ''
        for i, v in enumerate(sample):
            if i > 0:
                line += '; '
            
            if v.__class__.__name__ == "list":
                #multiple values...
                for j, sv in enumerate(v):
                    if j > 0:
                        line += '; '
                    line += str(sv)
            else:
                #single value...
                line += str(v)
                
        line += '\r\n'                     
        content += line
        
        if len(content) >= 50000:
            file.write(content)
            content = ''
        
    file.write(content)
    
    file.close()

    print "Saving auxiliary.... "

    #Now, add the auxiliary file
    try:
        aux_file = open(sys.argv[2] + ".sources.txt" , 'w')
    except:
        print( "File <" + sys.argv[2]  + ".sources.txt> could not be created")
        return

    content = ''
    for source_path, sym_id in sources:
        content += source_path + ', ' + str(sym_id) + '\r\n'

    aux_file.write(content)

    aux_file.close()

    print "Done!"
main()
