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

#=====================================================================
#  this program takes as input a training set, a testing set and a
#  classifier and runs the evaluation on parallel, then metrics
#  are computed.
#
#  Created by:
#      - Kenny Davila (Feb 12, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 12, 2012-2014)
#
#=====================================================================


def evaluate_data(classifier, data, first, last, results_queue):

    predicted = classifier.predict(data)

    results_queue.put((first, last, predicted))

def parallel_evaluate(classifier, dataset, workers):
    #...determine the ranges per worker...
    n_samples = np.size(dataset, 0)
    max_per_worker = int(math.ceil(float(n_samples) / float(workers)))

    #...prepare for parallel processing...
    results_queue = multiprocessing.Queue()
    processes = []
    final_results = np.zeros(n_samples)

    #...start parallel threads....
    for i in range(workers):
        first_element = max_per_worker * i
        last_element = min(max_per_worker * (i + 1), n_samples)

        params = (classifier, dataset[first_element:last_element, :], first_element, last_element, results_queue)
        p = multiprocessing.Process(target=evaluate_data, args=params)
        p.start()

        processes.append(p)

    #...await for al results to be ready...
    for i in range(workers):
        first, last, predicted = results_queue.get()

        #merge with current final results...
        final_results[first:last] = predicted

    return final_results


def main():
    if len(sys.argv) < 6:
        print("Usage: python parallel_evaluate.py training_set testing_set classifier normalize " +
              "workers [test_only] [ambiguous]")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\tclassifier\t= File that contains the pickled classifier")
        print("\tnormalized\t= Whether training data was normalized prior training")
        print("\tworkers\t\t= Number of parallel threads to use")
        print("\ttest_only\t= Optional, Only execute for testing set")
        print("\tambiguous\t= Optional, file that contains the list of ambiguous")
        return

    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    classifier_file = sys.argv[3]

    try:
        normalized = int(sys.argv[4]) > 0
    except:
        print("Invalid normalized value")
        return

    try:
        workers = int(sys.argv[5])
        if workers < 1:
            print("Invalid number of workers")
            return
    except:
        print("Invalid number of workers")
        return

    if len(sys.argv) >= 7:
        try:
            test_only = int(sys.argv[6]) > 0
        except:
            print("Invalid value for test_only")
            return
    else:
        test_only = False

    if len(sys.argv) >= 8:
        allograph_file = sys.argv[7]
    else:
        allograph_file = None
        ambiguous = None

    print("Loading data...")

    #...loading traininig data...
    training, labels_l, att_types = load_dataset(training_file)

    if att_types is None:
        print("Error loading File <" + training_file + ">")
        return

    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    n_classes = len(classes_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    #...loading testing data...
    testing, test_labels_l, att_types = load_dataset(testing_file)

    if att_types is None:
        print("Error loading File <" + testing_file + ">")
        return

    labels_test = get_mapped_labels(test_labels_l, classes_dict)

    if normalized:
        print("Normalizing...")

        scaler = StandardScaler()
        training = scaler.fit_transform(training)
        testing = scaler.transform(testing)


    print("Loading classifier...")

    in_file = open(classifier_file, 'rb')
    classifier = cPickle.load(in_file)
    in_file.close()

    if not allograph_file is None:
        print("Loading ambiguous file...")
        ambiguous, total_ambiguous = load_ambiguous(allograph_file, classes_dict, True)

        print("...A total of " + str(total_ambiguous) + " were found")

    print("Evaluating in multiple threads...")

    start_time = time.time()

    n_training_samples = np.size(training, 0)
    n_testing_samples = np.size(testing, 0)

    if not test_only:
        #Training data
        #....first, evaluate samples on multiple threads
        predicted = parallel_evaluate(classifier, training, workers)
        #....on main thread, compute final statistics
        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
        accuracy = (float(total_correct) / float(n_training_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        print "Training Samples: " + str(n_training_samples)
        print "Training Results"
        print "Accuracy\tClass Average\tClass STD "
        print str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)
    else:
        print("...Skipping Training set...")

    #Testing data
    #....first, evaluate samples on multiple threads
    predicted = parallel_evaluate(classifier, testing, workers)
    #....on main thread, compute final statistics
    total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_test, n_classes)
    accuracy = (float(total_correct) / float(n_testing_samples)) * 100
    avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
    print "Testing Samples: " + str(n_testing_samples)
    print "Testing Results"
    print "Accuracy\tClass Average\tClass STD "
    print str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

    end_time = time.time()
    total_elapsed = end_time - start_time

    print "Total Elapsed: " + str(total_elapsed)

    #get confusion matrix...
    print("Generating confusion matrix for test...")
    confussion = compute_confusion_matrix(predicted, labels_test, n_classes)
    print("Saving results...")
    results_file = classifier_file + ".results.csv"
    save_evaluation_results(classes_l, confussion, results_file, ambiguous)

    #...check for ambiguous...
    if not ambiguous is None:
        #...recompute metrics considering ambiguous...
        confussion = compute_ambiguous_confusion_matrix(predicted, labels_test, n_classes, ambiguous)

        results_file = classifier_file + ".results_ambiguous.csv"
        save_evaluation_results(classes_l, confussion, results_file, ambiguous)

    print("...Finished!")

if __name__ == '__main__':
    main()




