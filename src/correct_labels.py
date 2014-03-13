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
from dataset_ops import *


#=====================================================================
#  Corrects certain common labeling errors usually found in CROHME
#  datasets
#
#  Created by:
#      - Kenny Davila (Oct, 2012)
#  Modified By:
#      - Kenny Davila (Oct 25, 2013)
#      - Kenny Davila (Feb 5, 2012-2014)
#
#=====================================================================

def get_classes_list(labels):
    all_classes = {}

    for label in labels:
        if not label in all_classes:
            all_classes[label] = True

    return all_classes.keys()


def replace_labels(labels):
    new_labels = []
    
    for label in labels:
        if label == "\\tg":
            new_label = "\\tan"
        elif label == '>':
            new_label = '\gt'
        elif label == '<':
            new_label = '\lt'
        elif label == '\'':
            new_label = '\prime'
        elif label == '\cdots':
            new_label = '\ldots'
        elif label == '\\vec':
            new_label = '\\rightarrow'
        elif label == '\cdot':
            new_label = '.'
        elif label == ',':
            new_label = 'COMMA'
        elif label == '\\frac':
            new_label = '-'
        else:
            #unchanged...
            new_label = label

        new_labels.append(new_label)

    return new_labels


def main():
    #usage check
    if len(sys.argv) != 3:
        print("Usage: python correct_labels.py training_set output_set ")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training_set")        
        print("\toutput_set\t= File to output file with corrected labels")        
        return

    #...load training data from file...
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    #file_name = 'ds_test_2012.txt'
    training, labels_l, att_types = load_dataset(input_filename)
    
    if training is None:
        print("Data not could not be loaded")
    else:
        #extract list of classes...
        all_classes = get_classes_list(labels_l)

        print("Original number of classes = " + str(len(all_classes)))

        new_labels = replace_labels(labels_l)
        
        new_all_classes = get_classes_list(new_labels)
        print("New number of classes = " + str(len(new_all_classes)))

        save_dataset_string_labels(training, new_labels, att_types, output_filename)

        print("Success!")

main()

