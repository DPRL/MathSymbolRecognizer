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
import numpy as np
from dataset_ops import *


def get_counts_per_class(labels_l):
    count_per_class = {}

    for label in labels_l:
        if label in count_per_class:
            count_per_class[label] += 1
        else:
            count_per_class[label] = 1

    return count_per_class


def count_in_common(counts_l1, counts_l2):
    common_1 = 0
    common_2 = 0
    common_labels = []
    only1 = []
    only2 = []

    for label1 in counts_l1:
        if label1 in counts_l2:
            common_1 += counts_l1[label1]
            common_2 += counts_l2[label1]

            common_labels.append(label1)
        else:
            only1.append(label1)

    for label2 in counts_l2:
        if not label2 in counts_l1:
            only2.append(label2)


    return common_1, common_2, common_labels, only1, only2


def extract_common_classes(dataset, labels, common_labels):
    #...first, find the references to samples from common labels...
    common_refs = []
    for idx, label in enumerate(labels):
        if label in common_labels:
            common_refs.append(idx)

    #now... create a new dataset only with common refs...
    n_common_size = len(common_refs)
    n_atts = np.size(dataset, 1)
    common_set_data = np.zeros((n_common_size, n_atts), dtype=np.float64)
    common_set_labels = []

    for idx, ref_idx in enumerate(common_refs):
        common_set_data[idx, :] = dataset[ref_idx, :]
        common_set_labels.append(labels[ref_idx])

    return common_set_data, common_set_labels

def main():
    if len(sys.argv) != 4:
        print("Usage: python count_common.py dataset_1 dataset_2")
        print("Where")
        print("\tdataset_1\t= Path to the file of the first dataset")
        print("\tdataset_2\t= Path to the file of the second dataset")
        print("\textract\t= Extract common samples from second dataset")
        return

    data1_filename = sys.argv[1]
    data2_filename = sys.argv[2]

    try:
        do_extraction = int(sys.argv[3]) > 0
    except:
        print("Invalid value for extract parameter")
        return

    #...loading dataset...
    print("...Loading data set!")
    data1, labels_l1, att_types_1 = load_dataset(data1_filename)
    data2, labels_l2, att_types_2 = load_dataset(data2_filename)

    print("...Getting counts!")
    counts_l1 = get_counts_per_class(labels_l1)
    counts_l2 = get_counts_per_class(labels_l2)

    print("...Finding class overlap!")
    common1, common2, common_labels, only1, only2 = count_in_common(counts_l1, counts_l2)

    print("Classes on dataset 1: " + str(len(counts_l1.keys())))
    print("Classes on dataset 2: " + str(len(counts_l2.keys())))
    print("Total common classes: " + str(len(common_labels)))
    for k in common_labels:
        print(k)
    print("Samples of common classes on dataset 1: " + str(common1))
    print("Samples of common classes on dataset 2: " + str(common2))

    print("Total classes only on 1: " + str(len(only1)))
    for k in only1:
        print(k)
    print("Total classes only on 2: " + str(len(only2)))
    for k in only2:
        print(k)

    if do_extraction:
        print("Extracting samples with common labels from dataset 2")
        filtered_data, filtered_labels = extract_common_classes(data2, labels_l2, common_labels)

        print("Saving filtered samples ")
        save_dataset_string_labels(filtered_data,filtered_labels,att_types_2, data2_filename + ".common.txt")

    print("Finished!")

main()
