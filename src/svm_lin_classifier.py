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
import numpy as np
import cPickle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from dataset_ops import *
from evaluation_ops import *


def main():
    #usage check...
    if len(sys.argv) < 3:
        print("Usage: python svm_lin_classifier.py training testing [evaluate] [probab]")
        print("Where")
        print("\ttraining\t= Path to training set")
        print("\ttesting\t= Path to testing set")
        print("\tevaluate\t= Optional, run evaluation or just training ")
        print("\tprobab\t= Optional, make it a probabilistic classifier")
        return

    print("loading data")

    #...load training data from file...
    file_name = sys.argv[1]
    #file_name = 'ds_test_2012.txt'
    training, labels_l, att_types = load_dataset(file_name)
    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    n_classes = len(classes_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    #now, load the testing set...
    file_name = sys.argv[2]
    testing, test_labels_l, att_types = load_dataset(file_name)
    #...generate mapped labels...
    labels_test = get_mapped_labels(test_labels_l, classes_dict)

    if len(sys.argv) >= 4:
        try:
            evaluate = int(sys.argv[3]) > 0
        except:
            print("Invalid evaluate value")
            return
    else:
        evaluate = True

    if len(sys.argv) >= 5:
        try:
            make_probabilistic = int(sys.argv[4]) > 0
        except:
            print("Invalid probab value")
            return
    else:
        make_probabilistic = False

    print("Training with: " + sys.argv[1])
    print("Training Samples: " + str(np.size(training, 0)))
    print("Testing with: " + sys.argv[2])
    print("Testing Samples: " + str(np.size(testing, 0)))

    start_time = time.time()

    #try scaling...
    scaler = StandardScaler()
    training = scaler.fit_transform(training)
    testing = scaler.transform(testing)

    print("...training...")
    classifier = svm.SVC(kernel='linear', probability=make_probabilistic)
    classifier.fit(training, np.ravel(labels_train) )

    print("...Saving to file...")
    out_file = open(sys.argv[1] + ".LSVM", 'wb')
    cPickle.dump(classifier, out_file, cPickle.HIGHEST_PROTOCOL)
    out_file.close()

    if not evaluate:
        #...finish early...
        end_time = time.time()
        total_elapsed = end_time - start_time

        print "Total Elapsed: " + str(total_elapsed)
        print("Finished!")
        return

    print("...Evaluating...")

    #...first, training error
    n_training_samples = np.size(training, 0)
    predicted = classifier.predict(training)
    total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
    accuracy = (float(total_correct) / float(n_training_samples)) * 100
    avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
    print "Training Samples: " + str(n_training_samples)
    print "Training Results"
    print "Accuracy\tClass Average\tClass STD "
    print str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

    #...then, testing error
    n_testing_samples = np.size(testing, 0)
    predicted = classifier.predict(testing)
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

    print("Finished")

main()
