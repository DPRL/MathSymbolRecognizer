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
import ctypes
import time
import numpy as np
from dataset_ops import *
from evaluation_ops import *

#=====================================================================
#  this program takes as input a data set and through AdaBoost M1 it generates
#  a set of decision trees to classify data. the classifier is then
#  written to a file for further use.
#
#  Created by:
#      - Kenny Davila (Dec 27, 2013)
#  Modified By:
#      - Kenny Davila (Dec 27, 2013)
#      - Kenny Davila (Jan 19, 2013)
#        - Dataset functions are now imported from external file
#      - Kenny Davila (Jan 27, 2013)
#        - Added mapping
#        - Save result to file
#
#=====================================================================

c45_lib = ctypes.CDLL('./adaboost_c45.so')


def ensemble_predict(classifier, data):
    n_samples = np.size(data, 0)
    predicted = np.zeros(n_samples)

    for k in range(n_samples):
        c_sample = data[k, :]
        c_sample_p = c_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        predicted[k] = c45_lib.boosted_c45_classify(classifier, c_sample_p)

    return predicted

def main():
    #usage check
    if len(sys.argv) < 4:
        print("Usage: python train_adaboost.py training_set testing_set rounds [mapped] [save_weights]")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\trounds\t\t= Rounds to use for AdaBoost")
        print("\tmapped\t= Optional, reads class mapping if specified")
        print("\tsave_weights\t= Optional, save the weights used for boosting")
        return

    try:
        rounds = int(sys.argv[3])
        if ( rounds < 1):
            print("Must be at least 1 round")
            return
    except:
        print("Invalid number of rounds")
        return
    
    #...load training data from file...
    #file_name = 'ds_2012.txt'
    file_name = sys.argv[1]
    #file_name = 'ds_test_2012.txt'
    training, labels_l, att_types = load_dataset(file_name)
    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    if len(sys.argv) >= 5:
        try:
            mapped = int(sys.argv[4]) > 0
        except:
            mapped = False

        #...used for the clustered training set
        if mapped:
            o_classes_l, o_classes_dict, class_mapping = load_label_mapping(file_name + ".mapping.txt")
    else:
        mapped =False

    if len(sys.argv) >= 6:
        try:
            save_weights = int(sys.argv[5]) > 0
        except:
            save_weights = False
    else:
        save_weights = False

    #...load testing data from file...
    #file_name = 'ds_test_2012.txt'
    file_name = sys.argv[2]
    #file_name = 'ds_2012.txt'
    testing, test_labels_l, att_types = load_dataset(file_name)
    #...generate mapped labels...
    if mapped:
        labels_test = get_mapped_labels(test_labels_l, o_classes_dict)
    else:
        labels_test = get_mapped_labels(test_labels_l, classes_dict)

    #... for learning....
    n_samples = np.size(training, 0)
    n_atts = np.size(att_types, 0)
    n_classes = len( classes_l )
    print "Samples : " + str(n_samples)
    print "Atts: " + str(n_atts)
    print "Classes: " + str(n_classes)    

    #...create distribution    
    init_value = 1.0 / float(n_samples)
    distrib_train = np.zeros( (n_samples, 1), dtype=np.float64 )
    for i in range(n_samples):
        distrib_train[i, 0] = init_value
    
    #prepare to pass data to C-side
    samples_p = training.ctypes.data_as( ctypes.POINTER( ctypes.c_double ) )
    
    labels_p = labels_train.ctypes.data_as( ctypes.POINTER( ctypes.c_int ) )
    atts_p = att_types.ctypes.data_as( ctypes.POINTER( ctypes.c_int ) )

    #for testing
    n_test_samples = np.size(testing, 0)

    start_time = time.time()

    #test construction...
    m_rounds = rounds
    ensemble = c45_lib.created_boosted_c45( samples_p, labels_p, atts_p, n_samples, n_atts, n_classes, m_rounds, 5, 1, "Fold #1")

    print "Saving..."
    c45_lib.boosted_c45_save(ensemble, sys.argv[1] + ".bc45")

    if save_weights:
        c45_lib.boosted_c45_save_training_weights(ensemble, sys.argv[1] + ".weights.bin", 0)
        c45_lib.boosted_c45_save_training_weights(ensemble, sys.argv[1] + ".weights.txt", 1)

    #test accuracy of tree...

    #...training...
    predicted = ensemble_predict(ensemble, training)
    total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
    accuracy = (float(total_correct) / float(n_samples)) * 100
    avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
    print "Training Samples: " + str(n_samples)
    print "Training Results"
    print "Accuracy\tClass Average\tClass STD "
    print str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

    #...testing...
    n_testing_samples = np.size(testing, 0)
    predicted = ensemble_predict(ensemble, testing)
    total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_test, n_classes)
    accuracy = (float(total_correct) / float(n_testing_samples)) * 100
    avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
    print "Testing Samples: " + str(n_testing_samples)
    print "Testing Results"
    print "Accuracy\tClass Average\tClass STD "
    print str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)


    c45_lib.release_boosted_c45( ensemble )

    end_time = time.time()
    total_elapsed = end_time - start_time
    print "Total Elapsed: " + str(total_elapsed)
    print("Training: " + sys.argv[1])
    print("Testing: " + sys.argv[2])

main()


