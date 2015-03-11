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
import time
import math
import numpy as np
import cPickle
import multiprocessing
from sklearn.preprocessing import StandardScaler
from dataset_ops import *
from evaluation_ops import *
from symbol_classifier import SymbolClassifier

#=====================================================================
#  this program takes as input a training set, a testing set and a
#  classifier and runs the evaluation on parallel, then metrics
#  are computed.
#
#  Created by:
#      - Kenny Davila (Feb 12, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 12, 2012-2014)
#      - Kenny Davila (Feb 25, 2012-2014
#        - Now includes top-5 accuracy too
#      - Kenny Davila (Mar 11, 2015)
#        - Symbol classifier class now adopted
#
#=====================================================================


def evaluate_data(classifier, data, first, last, top_n, results_queue):
    #use probabilistic evaluation to get probability per class...
    predicted = classifier.predict_proba(data)

    n_samples = np.size(data, 0)
    n_classes = np.size(predicted, 1)
    results = np.zeros((n_samples, top_n))

    #find the top N values per sample...
    raw_classes = classifier.get_raw_classes()
    for i in range(n_samples):
        tempo_values = []
        for k in range(n_classes):
            tempo_values.append((predicted[i, k], raw_classes[k]))

        #now, sort!
        tempo_values = sorted(tempo_values, reverse=True)

        #use the top-N
        for k in range(top_n):
            results[i, k] = tempo_values[k][1]

    results_queue.put((first, last, results))

def parallel_evaluate(classifier, dataset, workers, top_n):
    #...determine the ranges per worker...
    n_samples = np.size(dataset, 0)
    max_per_worker = int(math.ceil(float(n_samples) / float(workers)))

    #...prepare for parallel processing...
    results_queue = multiprocessing.Queue()
    processes = []
    final_results = np.zeros((n_samples, top_n))

    #...start parallel threads....
    for i in range(workers):
        first_element = max_per_worker * i
        last_element = min(max_per_worker * (i + 1), n_samples)

        params = (classifier, dataset[first_element:last_element, :], first_element, last_element, top_n, results_queue)
        p = multiprocessing.Process(target=evaluate_data, args=params)
        p.start()

        processes.append(p)

    #...await for al results to be ready...
    for i in range(workers):
        first, last, predicted = results_queue.get()

        #merge with current final results...
        final_results[first:last, :] = predicted

    return final_results


def main():
    if len(sys.argv) < 5:
        print("Usage: python parallel_prob_evaluate.py training_set testing_set classifier normalize " +
              "workers [test_only] [ambiguous]")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\tclassifier\t= File that contains the pickled classifier")
        print("\tworkers\t\t= Number of parallel threads to use")
        print("\ttest_only\t= Optional, Only execute for testing set")
        print("\tambiguous\t= Optional, file that contains the list of ambiguous")
        return

    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    classifier_file = sys.argv[3]

    try:
        workers = int(sys.argv[4])
        if workers < 1:
            print("Invalid number of workers")
            return
    except:
        print("Invalid number of workers")
        return

    if len(sys.argv) >= 6:
        try:
            test_only = int(sys.argv[5]) > 0
        except:
            print("Invalid value for test_only")
            return
    else:
        test_only = False

    if len(sys.argv) >= 7:
        allograph_file = sys.argv[6]
    else:
        allograph_file = None
        ambiguous = None

    print("Loading classifier...")

    in_file = open(classifier_file, 'rb')
    classifier = cPickle.load(in_file)
    in_file.close()

    if not isinstance(classifier, SymbolClassifier):
        print("Invalid classifier file!")
        return

    print("Loading data...")

    # get mapping
    classes_dict = classifier.classes_dict
    classes_l = classifier.classes_list
    n_classes = len(classes_l)

    if not test_only:
        #...loading traininig data...
        training, labels_l, att_types = load_dataset(training_file)
        if att_types is None:
            print("Error loading File <" + training_file + ">")
            return

        #...generate mapped labels...
        labels_train = get_mapped_labels(labels_l, classes_dict)
    else:
        training, labels_l, labels_train = (None, None, None)

    #...loading testing data...
    testing, test_labels_l, att_types = load_dataset(testing_file)

    if att_types is None:
        print("Error loading File <" + testing_file + ">")
        return

    labels_test = get_mapped_labels(test_labels_l, classes_dict)

    if classifier.scaler is not None:
        print("Normalizing...")

        scaler = classifier.scaler
        if not test_only:
            training = scaler.transform(training)

        testing = scaler.transform(testing)

    if not allograph_file is None:
        print("Loading ambiguous file...")
        ambiguous, total_ambiguous = load_ambiguous(allograph_file, classes_dict, True)

        print("...A total of " + str(total_ambiguous) + " were found")

    print("Training data set: " + training_file)
    print("Testing data set: " + testing_file)
    print("Evaluating in multiple threads...")

    start_time = time.time()

    top_n = 5

    if not test_only:
        # Training data
        n_training_samples = np.size(training, 0)

        # ....first, evaluate samples on multiple threads
        predicted = parallel_evaluate(classifier, training, workers, top_n)

        print "Training Samples: " + str(n_training_samples)
        print "Training Results"
        print "Top\tAccuracy\tClass Average\tClass STD "

        # ....on main thread, compute final statistics
        for i in range(top_n):
            total_correct, counts_per_class, errors_per_class = compute_topn_error_counts(predicted, labels_train,
                                                                                          n_classes, i + 1)
            accuracy = (float(total_correct) / float(n_training_samples)) * 100
            avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)

            print str(i+1) + "\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)
    else:
        print("...Skipping Training set...")

    n_testing_samples = np.size(testing, 0)

    # Testing data
    # ....first, evaluate samples on multiple threads
    predicted = parallel_evaluate(classifier, testing, workers, top_n)

    #....on main thread, compute final statistics
    print "Testing Samples: " + str(n_testing_samples)
    print "Testing Results"
    print "Top\tAccuracy\tClass Average\tClass STD "

    for i in range(top_n):
        total_correct, counts_per_class, errors_per_class = compute_topn_error_counts(predicted, labels_test,
                                                                                      n_classes, i + 1)
        accuracy = (float(total_correct) / float(n_testing_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)

        print str(i+1) + "\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

    end_time = time.time()
    total_elapsed = end_time - start_time

    print "Total Elapsed: " + str(total_elapsed)

    print("...Finished!")

if __name__ == '__main__':
    main()

