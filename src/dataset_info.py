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
import math
import numpy as np
from dataset_ops import *
#=====================================================================
#  Simple script that loads an existing dataset from a file and
#  outputs the most general information.
#
#  Created by:
#      - Kenny Davila (Feb 7, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 7, 2012-2014)
#      - Kenny Davila (Feb 7, 2012-2014)
#
#=====================================================================


def main():
    #usage check
    if len(sys.argv) < 2:
        print("Usage: python dataset_info.py datset [n_bins]")
        print("Where")
        print("\tdataset\t= Path to file that contains the data set")
        print("\tn_bins\t= Optional, number of bins for histogram of class representation")
        return

    input_filename = sys.argv[1]

    if len(sys.argv) >= 3:
        try:
            n_bins = int(sys.argv[2])
            if n_bins < 1:
                print("Invalid n_bins value")
                return
        except:
            print("Invalid n_bins value")
            return
    else:
        n_bins = 10

    print("Loading data....")
    training, labels_l, att_types = load_dataset(input_filename);
    print("Data loaded!")

    print("Getting information...")
    n_samples = np.size(training, 0)
    n_atts = np.size(training, 1)

    #...counts per class...
    count_per_class = {}
    for idx in range(n_samples):
        #s_label = labels_train[idx, 0]
        s_label = labels_l[idx]

        if s_label in count_per_class:
            count_per_class[s_label] += 1
        else:
            count_per_class[s_label] = 1

    #...distribution...
    #...first pass, compute minimum and maximum...
    smallest_class_size = n_samples
    smallest_class_label = ""
    largest_class_size = 0
    largest_class_label = ""
    for label in count_per_class:
        #...check minimum
        if count_per_class[label] < smallest_class_size:
            smallest_class_size = count_per_class[label]
            smallest_class_label = label

        #...check maximum
        if count_per_class[label] > largest_class_size:
            largest_class_size = count_per_class[label]
            largest_class_label = label

        print("Class\t" + label + "\tCount\t" + str(count_per_class[label]))

    #...second pass... create histogram...
    count_bins = [0 for x in range(n_bins)]
    samples_bins = [0 for x in range(n_bins)]
    size_per_bin = int(math.ceil(float(largest_class_size + 1) / float(n_bins)))
    for label in count_per_class:
        current_bin = int(count_per_class[label] / size_per_bin)
        count_bins[current_bin] += 1
        samples_bins[current_bin] += count_per_class[label]

    #...print... bins...
    print("Class sizes distribution")
    for i in range(n_bins):
        start = i * size_per_bin + 1
        end = (i + 1) * size_per_bin
        percentage = (float(samples_bins[i]) / float(n_samples)) * 100.0
        print("... From " + str(start) + "\t to " + str(end) + "\t : " +
              str(count_bins[i]) + "\t (" + str(percentage) + " of data)")

    n_classes = len(count_per_class.keys())

    print("Total Samples: " + str(n_samples))
    print("Total Attributes: " + str(n_atts))
    print("Total Classes: " + str(n_classes))
    print("-> Largest Class: " + largest_class_label + "\t: " + str(largest_class_size) + " samples")
    print("-> Smallest Class: " + smallest_class_label + "\t: " + str(smallest_class_size) + " samples")
    print("Finished...")

main()
