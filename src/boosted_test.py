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
import numpy as np
from dataset_ops import *
from evaluation_ops import *
from load_inkml import *

#=====================================================================
#  this program takes as input boosted classifier and the dataset
#  used for training, and a testing set and computes the classifier accuracy
#
#  Created by:
#      - Kenny Davila (Jan 19, 2013)
#  Modified By:
#      - Kenny Davila (Jan 19, 2013)
#      - Kenny Davila (Feb 26, 2012-2014)
#        - eliminated mapping compatibility
#        - Added evaluation metrics: top-n, per-class avg
#
#=====================================================================

c45_lib = ctypes.CDLL('./adaboost_c45.so')


def count_per_class(labels, n_samples, n_classes):
    result_counts = [0 for x in range(n_classes)]

    for i in range(n_samples):
        result_counts[labels[i]] += 1

    return result_counts


def output_sample(file_path, sym_id, out_path):
    #load file...
    symbols = load_inkml( file_path, True )

    #find symbol ...
    for symbol in symbols:
        if symbol.id == sym_id:
            symbol.saveAsSVG( out_path )

            return True

    return False


def predict_top_n_data(classifier, data, top_n, n_classes):
    n_samples = np.size(data, 0)
    p_classes = np.zeros((1, n_classes), dtype=np.float64)
    p_classes_p = p_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    results = np.zeros((n_samples, top_n))

    for i in range(n_samples):
        c_sample = data[i, :]
        c_sample_p = c_sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        c45_lib.boosted_c45_probabilistic_classify(classifier, c_sample_p, p_classes_p)

        tempo_values = []
        for k in range(n_classes):
            tempo_values.append((p_classes[0, k], k))

        #now, sort!
        tempo_values = sorted(tempo_values, reverse=True)

        #use the top-N
        for k in range(top_n):
            results[i, k] = tempo_values[k][1]

    return results


def main():
    #usage check
    if len(sys.argv) < 4:
        print("Usage: python boosted_test.py classifier training_set testing_set [save_fail]")
        print("Where")
        print("\tclassifier\t= Path to the .bc45 file that contains the classifier")
        print("\ttraining_set\t= Path to the file of the training set")
        print("\ttesting_set\t= Path to the file of the testing set")
        print("\tsave_fail\t= Optional, will output failure case if specified")
        return 

    #...load training data from file...
    print "...Loading Training Data..."

    file_name = sys.argv[2]
    training, labels_l, att_types = load_dataset(file_name)
    #...generate mapping...
    classes_dict, classes_l = get_label_mapping(labels_l)
    #...generate mapped labels...
    labels_train = get_mapped_labels(labels_l, classes_dict)

    if len(sys.argv) >= 5:
        save_fail = int(sys.argv[4]) > 0
    else:
        save_fail = False

    #...load testing data from file...
    print "...Loading Testing Data..."

    testing, test_labels_l, att_types = load_dataset(sys.argv[3])
    #...generate mapped labels...
    labels_test = get_mapped_labels(test_labels_l, classes_dict)

    if save_fail:
        #need to load sources...
        test_sources = load_ds_sources(sys.argv[3] + ".sources.txt")

        if test_sources is None:
            print("Sources are unavailable")
            save_fail = False

    print "...Loading classifier..."
    ensemble = c45_lib.boosted_c45_load( sys.argv[1])

    #...info....
    n_classes = len(classes_l)
    n_train_samples = np.size(training, 0)
    n_test_samples = np.size(testing, 0)

    print("Classifier: " + sys.argv[1])
    print("Training Set: " + sys.argv[2])
    print("Testing Set: " + sys.argv[3])


    print "...evaluating..."

    top_n = 5
    predicted = predict_top_n_data(ensemble, training, top_n, n_classes)
    print "Training Samples: " + str(n_train_samples)
    print "Training Results"
    print "Top\tAccuracy\tClass Average\tClass STD "

    #....on main thread, compute final statistics
    for i in range(top_n):
        total_correct, counts_per_class, errors_per_class = compute_topn_error_counts(predicted, labels_train,
                                                                                      n_classes, i + 1)
        accuracy = (float(total_correct) / float(n_train_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)

        print str(i+1) + "\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)

    predicted = predict_top_n_data(ensemble, testing, top_n, n_classes)
    print "Testing Samples: " + str(n_test_samples)
    print "Testing Results"
    print "Top\tAccuracy\tClass Average\tClass STD "
    for i in range(top_n):
        total_correct, counts_per_class, errors_per_class = compute_topn_error_counts(predicted, labels_test,
                                                                                      n_classes, i + 1)
        accuracy = (float(total_correct) / float(n_test_samples)) * 100
        avg_accuracy, std_accuracy = get_average_class_accuracy(counts_per_class, errors_per_class, n_classes)

        print str(i+1) + "\t" + str(accuracy) + "\t" + str(avg_accuracy * 100.0) + "\t" + str(std_accuracy * 100.0)


    #repeat for Top=1 accuracy...
    all_failure_info = []
    confusion_matrix = np.zeros((n_classes, n_classes), dtype = np.int32)
    total_correct = 0
    train_counts = count_per_class(labels_train, n_train_samples, n_classes)
    for k in range(n_test_samples):
        top_label = predicted[i, 0]

        if top_label != labels_test[k, 0]:
            #inccorrect...
            if save_fail:
                file_path, sym_id = test_sources[k]
                output_sample(file_path, sym_id, "output//error_" + str(k) + ".svg")

                all_failure_info.append((k, file_path, sym_id, classes_l[top_label], classes_l[labels_test[k, 0]]))
        else:
            total_correct += 1

        confusion_matrix[labels_test[k, 0], top_label] += 1

    if save_fail:
        #...save additional info of errors...
        content = 'id, path, sym_id, predicted, expected\n'
        for error_info in all_failure_info:

            for i in range(len(error_info)):
                if i > 0:
                    content += ","

                content += str(error_info[i])

            content += "\n"

        file_name = sys.argv[1] + ".failures.csv"
        try:
            f = open(file_name, 'w')
            f.write(content)
            f.close()
        except:
            print("ERROR WRITING RESULTS TO FILE! <" + file_name + ">")


    #... print results....
    accuracy = (float(total_correct) / float(n_test_samples)) * 100.0
    print "Testing Accuracy = " + str(accuracy)

    #.... save confusion matrix to file...
    out_str = "Samples:," + str(n_test_samples) + "\n"
    out_str += "Correct:," + str(total_correct) + "\n"
    out_str += "Wrong:," + str(n_test_samples - total_correct) + "\n"
    out_str += "Accuracy:," + str(accuracy) + "\n\n\n"
    out_str += "Rows:,Expected \n"
    out_str += "Column:,Predicted \n\n"
    out_str += "Full Matrix\n"
    out_str += "X"
    only_err = "X"
    #...build header...
    for i in range(n_classes ):
        c_class = classes_l[i]
        if c_class == ",":
            c_class = "COMMA"

        out_str += "," + c_class
        only_err += "," + c_class

    out_str += ",Total,Train Count\n"
    only_err += ",Total\n"

    #...build the content...
    for i in range(n_classes):
        c_class = classes_l[i]
        if c_class == ",":
            c_class = "COMMA"

        out_str += c_class
        only_err += c_class

        t_errs = 0
        t_samples = 0
        for k in range(n_classes):
            out_str += "," + str(confusion_matrix[i, k])

            t_samples += confusion_matrix[i, k]
            if i == k:
                only_err += ",0"
            else:
                t_errs += confusion_matrix[i, k]
                only_err += "," + str(confusion_matrix[i, k])

        out_str += "," + str(t_samples) + "," + str(train_counts[i]) + "\n"
        only_err += "," + str(t_errs) + "\n"

    #...build the totals row...
    out_str += "Total"
    only_err += "Total"
    for i in range(n_classes):
        t_errs = 0
        t_samples = 0

        for k in range(n_classes):
            #...inverted, k = row, i = column
            t_samples += confusion_matrix[k, i]
            if i != k:
                t_errs += confusion_matrix[k, i]

        out_str += "," + str(t_samples)
        only_err += "," + str(t_errs)


    out_str += "\n\n\nOnly Errors\n\n" + only_err

    file_name = sys.argv[1] + ".results.csv"

    try:
        f = open(file_name, 'w')
        f.write( out_str )
        f.close()
    except:
        print("ERROR WRITING RESULTS TO FILE! <" + file_name + ">")

    print "Finished!"

main()
