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
import cPickle
import math
from sklearn.ensemble import RandomForestClassifier
from dataset_ops import *
from evaluation_ops import *

#=====================================================================
#  this program takes as input a data set and using Random forest
#  it generates a set of decision trees to classify data.
#
#  Created by:
#      - Kenny Davila (Feb 1, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 1, 2015)
#
#=====================================================================

def predict_in_chunks(data, classifier, max_chunk_size):
    n_samples = np.size(data, 0)
    all_predicted = np.zeros(n_samples)
    pos = 0
    while pos < n_samples:
        last = min(pos + max_chunk_size, n_samples)

        #...get predictions for current chuck...
        predicted = classifier.predict(data[pos:last])
        #...add them to result...
        all_predicted[pos:last] = predicted

        #print("...Evaluated " + str(last) + " out of " + str(n_samples) )

        pos = last

    return all_predicted



def main():
    #usage check
    if len(sys.argv) < 8:
        print("Usage: python random_forest_classify.py training_set testing_set N_trees max_D max_feats "
              "type times [n_jobs]")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\tN_trees\t\t= Number of trees to use")
        print("\tmax_D\t\t= Maximum Depth")
        print("\tmax_feats\t= Maximum Features")
        print("\ttype\t\t= Type of Decision trees (criterion for splits)")
        print("\t\t\t\t0 - Gini")
        print("\t\t\t\t1 - Entropy")
        print("\ttimes\t\t= Number of times to repeat experiments")
        print ("\tn_jobs\t\t= Optional, number of parallel threads to use")
        return

    print("Loading data....")
    #...load training data from file...
    train_filename = sys.argv[1]
    training, labels_l, att_types = load_dataset(train_filename)
    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    n_classes = len(classes_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    #...load testing data from file...
    test_filename = sys.argv[2]
    testing, test_labels_l, att_types = load_dataset(test_filename)
    #...generate mapped labels...
    labels_test = get_mapped_labels(test_labels_l, classes_dict)

    try:
        n_trees = int(sys.argv[3])
        if n_trees < 1:
            print("Invalid N_trees value")
            return
    except:
        print("Invalid N_trees value")
        return

    try:
        max_D = int(sys.argv[4])
        if max_D < 1:
            print("Invalid max_D value")
            return
    except:
        print("Invalid max_D value")
        return


    try:
        max_features = int(sys.argv[5])
        if max_features < 1:
            print("Invalid max_feats value")
            return
    except:
        print("Invalid max_feats value")
        return

    try:
        criterion_t = 'gini' if int(sys.argv[6]) == 0 else 'entropy'
    except:
        print("Invalid type value")
        return

    try:
        times = int(sys.argv[7])
        if times < 1:
            print("Invalid times value")
            return
    except:
        print("Invalid times value")
        return

    if len(sys.argv) >= 9:
        try:
            num_jobs = int(sys.argv[8])
        except:
            print("Invalid number of jobs")
            return
    else:
        num_jobs = 1

    print("....Data loaded!")

    start_time = time.time()
    total_training_time = 0
    total_testing_time = 0

    all_training_accuracies = np.zeros(times)
    all_training_averages = np.zeros(times)
    all_training_stds = np.zeros(times)
    all_testing_accuracies = np.zeros(times)
    all_testing_averages = np.zeros(times)
    all_testing_stds = np.zeros(times)

    n_training_samples = np.size(training, 0)
    n_testing_samples = np.size(testing, 0)

    #tree = DecisionTreeClassifier(criterion="gini")
    #tree = DecisionTreeClassifier(criterion="entropy")
    #boosting = AdaBoostClassifier(DecisionTreeClassifier(criterion="gini", max_depth=25), n_estimators=50,learning_rate=1.0)

    print "    \tTrain\tTrain\tTrain\tTest\tTest\tTest"
    print " #  \tACC\tC. AVG\tC. STD\tACC\tC. AVG\tC. STD"
    best_forest_ref = None
    best_forest_accuracy = None

    for i in range(times):
        start_training_time = time.time()
        #print "Training #" + str(i + 1)
        forest = RandomForestClassifier(n_estimators=n_trees,criterion=criterion_t,
                                        max_features=max_features, max_depth=max_D,n_jobs=num_jobs) #n_jobs=-1?

        forest.fit(training, np.ravel(labels_train))

        end_training_time = time.time()
        total_training_time += end_training_time - start_training_time

        start_testing_time = time.time()

        #...first, training error
        predicted = predict_in_chunks(training, forest, 1000)
        #...compute metrics...
        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_train, n_classes)
        accuracy = (float(total_correct) / float(n_training_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        all_training_accuracies[i] = accuracy
        all_training_averages[i] = avg_accuracy * 100.0
        all_training_stds[i] = std_accuracy * 100.0

        #...then, testing error
        #predicted = forest.predict(testing)
        predicted = predict_in_chunks(testing, forest, 1000)
        #...compute metrics ...
        total_correct, counts_per_class, errors_per_class = compute_error_counts(predicted, labels_test, n_classes)
        accuracy = (float(total_correct) / float(n_testing_samples)) * 100
        test_accuracy = accuracy
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)
        all_testing_accuracies[i] = accuracy
        all_testing_averages[i] = avg_accuracy * 100.0
        all_testing_stds[i] = std_accuracy * 100.0

        end_testing_time = time.time()

        total_testing_time += end_testing_time - start_testing_time

        print(" " + str(i+1) + "\t" + str(round(all_training_accuracies[i], 3)) + "\t" +
              str(round(all_training_averages[i], 3)) + "\t" +
              str(round(all_training_stds[i], 3)) + "\t" +
              str(round(all_testing_accuracies[i], 3)) + "\t" +
              str(round(all_testing_averages[i], 3)) + "\t" +
              str(round(all_testing_stds[i], 3)))

        if best_forest_ref is None or best_forest_accuracy < test_accuracy:
            best_forest_ref = forest
            best_forest_accuracy = test_accuracy


    end_time = time.time()
    total_elapsed = end_time - start_time

    print(" Mean\t" + str(round(all_training_accuracies.mean(), 3)) + "\t" +
          str(round(all_training_averages.mean(), 3)) + "\t" +
          str(round(all_training_stds.mean(), 3)) + "\t" +
          str(round(all_testing_accuracies.mean(), 3)) + "\t" +
          str(round(all_testing_averages.mean(), 3)) + "\t" +
          str(round(all_testing_stds.mean(), 3)))

    print(" STD\t" + str(round(all_training_accuracies.std(), 3)) + "\t" +
          str(round(all_training_averages.std(), 3)) + "\t" +
          str(round(all_training_stds.std(), 3)) + "\t" +
          str(round(all_testing_accuracies.std(), 3)) + "\t" +
          str(round(all_testing_averages.std(), 3)) + "\t" +
          str(round(all_testing_stds.std(), 3)))

    #...Save to file...
    out_file = open(sys.argv[1] + ".best.RF", 'wb')
    cPickle.dump(best_forest_ref, out_file, cPickle.HIGHEST_PROTOCOL)
    out_file.close()


    print("Total Elapsed: " + str(total_elapsed))
    print("Mean Elapsed Time: " + str(total_elapsed / times))
    print("Total Training time: " + str(total_training_time))
    print("Mean Training time: " + str(total_training_time / times))
    print("Total Evaluating time: " + str(total_testing_time))
    print("Mean Evaluating time: " + str(total_testing_time / times))

    print("Training File: " + train_filename)
    print("Training samples: " + str(n_training_samples))
    print("Testing File: " + test_filename)
    print("Testing samples: " + str(n_testing_samples))
    print("N_trees: " + str(n_trees))
    print("Criterion: " + criterion_t)
    print("Max features: " + str(max_features))
    print("Max Depth: " + str(max_D))
    print("Finished!")

if __name__ == '__main__':
    main()
