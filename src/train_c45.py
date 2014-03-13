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
import ctypes
import time
import sys
import numpy as np
from dataset_ops import *
from evaluation_ops import *

#=====================================================================
#  this program takes as input a data set and through AdaBoost M1 it generates
#  a set of decision trees to classify data. The data set is divided in folds
#  and cross validation is then applied, the best classifier of all the 
#  folds is then written to a file for further use.
#
#  Created by:
#      - Kenny Davila (Dec 27, 2013)
#  Modified By:
#      - Kenny Davila (Jan 16, 2013)
#      - Kenny Davila (Jan 19, 2013)
#        - Dataset functions are now imported from external file
#
#=====================================================================

c45_lib = ctypes.CDLL('./adaboost_c45.so')


def get_majority_class(labels, weights, classes):
    l_counts = {}
    n_samples = np.size( labels, 0)
    majority_class = None
    majority_w = 0
    for i in range(n_samples):
        label = labels[i, 0]
        
        if label in l_counts:
            l_counts[ label ] += weights[i, 0]
        else:
            l_counts[ label ] = weights[i, 0]

        if majority_class == None:
            majority_class = label
            majority_w = l_counts[ label ]
        else:
            if majority_w < l_counts[ label ]:
               majority_w = l_counts[ label ]
               majority_class = label

    """
    for l in l_counts:
        print str(classes[l]) + " - " + str(l) + " - " + str(l_counts[l] )
    """

    return majority_class, majority_w


def tree_predict(root, data):
    n_samples = np.size(data, 0)
    predicted = np.zeros(n_samples)

    for k in range(n_samples):
        c_sample = data[k, :]
        c_sample_p = c_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        predicted[k] = c45_lib.c45_node_evaluate(root, c_sample_p)

    return predicted


def main():
    if len(sys.argv) < 3:
        print("Usage: python train_c45.py training_set testing_set [mapped]")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\tmapped\t= Optional, reads class mapping if specified")
        return
    
    #...load training data from file...
    #file_name = 'ds_2012.txt'
    file_name = sys.argv[1]
    training, labels_l, att_types = load_dataset(file_name)
    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    #...check for class mapping...
    if len(sys.argv) >= 4:
        try:
            mapped = int(sys.argv[3]) > 0
        except:
            mapped = False

        #...will over-write the class list and class dict, and will extract the class mapping
        #...used for the clustered training set
        if mapped:
            o_classes_l, o_classes_dict, class_mapping = load_label_mapping(file_name + ".mapping.txt")

    else:
        mapped = False


    #...load testing data from file...
    #file_name = 'ds_test_2012.txt'
    file_name = sys.argv[2]
    testing, test_labels_l, att_types = load_dataset(file_name)
    #...generate mapped labels...
    if mapped:
        labels_test = get_mapped_labels(test_labels_l, o_classes_dict)
    else:
        labels_test = get_mapped_labels(test_labels_l, classes_dict)
        #print("UNCOMMENT MAPPING OF TESTING LABELS")



    print("Training: " + sys.argv[1])
    print("Testing: " + sys.argv[2])

    #for learning and testing....
    n_samples = np.size(training, 0)
    n_test_samples = np.size(testing, 0)
    n_atts = np.size(att_types, 0)
    n_classes = len( classes_l )
    print "Training Samples : " + str(n_samples)
    print "Testing Samples : " + str(n_test_samples)
    print "Atts: " + str(n_atts)
    print "Classes: " + str(n_classes)    

    #...create distribution    
    init_value = 1.0 / float(n_samples)
    distrib_train = np.zeros((n_samples, 1), dtype=np.float64)
    for i in range(n_samples):
        distrib_train[i, 0] = init_value
    
    #prepare to pass data to C-side
    samples_p = training.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    
    labels_p = labels_train.ctypes.data_as( ctypes.POINTER( ctypes.c_int ) )
    atts_p = att_types.ctypes.data_as( ctypes.POINTER( ctypes.c_int ) )
    distrib_p = distrib_train.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )



    #get majority class...
    majority_l, majority_w = get_majority_class(labels_train, distrib_train, classes_l)

    parent = None
    majority = int(majority_l)
    #print majority
    #print type(majority)

    start_time = time.time()

    #test construction...
    times = 1
    for i in range( times ):
        root = c45_lib.c45_tree_construct(samples_p, labels_p, atts_p, distrib_p, n_samples, n_atts, n_classes, majority, 5)

        """
        #un-comment to test file
        print "to write ... "
        c45_lib.c45_save_to_file( root, "tree_2012.tree" )
        print "to read ... "
        root = c45_lib.c45_load_from_file( "tree_2012.tree" )
        #return
        """
        
        #test accuracy of tree...
        predicted = tree_predict(root, training)

        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
        accuracy = (float(total_correct) / float(n_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        print "Results\tAccuracy\tClass Average\tClass STD "
        print "BP. Training\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

        #...testing...
        predicted = tree_predict(root, testing)

        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_test, n_classes)
        accuracy = (float(total_correct) / float(n_test_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        print "BP. Testing\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

        #do pruning
        c45_lib.c45_prune_tree.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_double )
        #c45_lib.c45_prune_tree( root, 0, 0.25 )
        c45_lib.c45_prune_tree(root, 1, 0.25)

        #test accuracy of tree...
        #...training...
        predicted = tree_predict(root, training)

        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
        accuracy = (float(total_correct) / float(n_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        print "AP. Training\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

        #...testing...
        predicted = tree_predict(root, testing)

        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_test, n_classes)
        accuracy = (float(total_correct) / float(n_test_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        print "AP. Testing\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)
        

        c45_lib.c45_node_release( root, 1 )

    end_time = time.time()
    total_elapsed = end_time - start_time
    mean_elapsed = total_elapsed / times
    print "Total Elapsed: " + str(total_elapsed)
    print "Mean Elapsed: " + str(mean_elapsed)
    print("")

main()


