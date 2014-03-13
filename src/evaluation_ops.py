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
import numpy as np

#=====================================================================
#  General functions used for evaluation
#
#  Created by:
#      - Kenny Davila (Feb 6, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 6, 2012-2014)
#      - Kenny Davila (Feb 23, 2012-2014)
#        - changes to handle ambiguous classes
#        - output confusion matrices
#      - Kenny Davila (Feb 25, 2012-2014)
#        - Add Top-N evaluation functions
#
#=====================================================================


def get_average_class_accuracy(counts_per_class, errors_per_class, n_classes):
    #now, only consider classes that existed in testing set...
    valid_classes = counts_per_class > 0

    all_accuracies = 1.0 - errors_per_class[valid_classes] / counts_per_class[valid_classes]

    avg_accuracy = all_accuracies.mean()
    std_accuracy = all_accuracies.std()

    return avg_accuracy, std_accuracy

def compute_error_counts(predicted, labels, n_classes):
    counts_per_class = np.zeros(n_classes)
    errors_per_class = np.zeros(n_classes)
    total_correct = 0
    n_samples = np.size(labels, 0)
    for k in range(n_samples):
        expected = labels[k, 0]

        counts_per_class[expected] += 1

        if predicted[k] == expected:
            total_correct += 1
        else:
            errors_per_class[expected] += 1

    return total_correct, counts_per_class, errors_per_class


def compute_topn_error_counts(predicted, labels, n_classes, top_n):
    counts_per_class = np.zeros(n_classes)
    errors_per_class = np.zeros(n_classes)
    total_correct = 0
    n_samples = np.size(labels, 0)
    for k in range(n_samples):
        expected = labels[k, 0]

        counts_per_class[expected] += 1

        found = False
        for n in range(top_n):
            if predicted[k, n] == expected:
                found = True
                break

        if found:
            total_correct += 1
        else:
            errors_per_class[expected] += 1

    return total_correct, counts_per_class, errors_per_class


def compute_confusion_matrix(predicted, labels, n_classes):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    n_test_samples = np.size(labels, 0)

    #... for each sample
    for k in range(n_test_samples):
        expected = int(labels[k, 0])

        confusion_matrix[expected, int(predicted[k])] += 1

    return confusion_matrix


def compute_ambiguous_confusion_matrix(all_predicted, labels, n_classes, ambiguous):
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    n_test_samples = np.size(labels, 0)

    #... for each sample
    for k in range(n_test_samples):
        expected = int(labels[k, 0])
        predicted = int(all_predicted[k])

        if ambiguous[expected, predicted] == 1:
            #no error will be reported between ambiguous classes
            confusion_matrix[expected, expected] += 1
        else:
            #usual error condition
            confusion_matrix[expected, predicted] += 1

    return confusion_matrix

def save_evaluation_results(classes_l, confusion_matrix, file_name, ambiguous):

    n_classes = np.size(confusion_matrix, 0)

    per_class_correct = np.zeros(n_classes)
    per_class_counts = confusion_matrix.sum(1)

    class_avg_list = []
    for i in range(n_classes):
        per_class_correct[i] = confusion_matrix[i, i]

        if per_class_counts[i] > 0:
            class_acc = per_class_correct[i] / per_class_counts[i]

            class_avg_list.append((class_acc, i, classes_l[i]))

    #...sort the list...
    class_avg_list = sorted(class_avg_list,reverse=True)

    #now, only consider classes that existed in testing set...
    valid_classes = per_class_counts > 0
    per_class_acc = per_class_correct[valid_classes] / per_class_counts[valid_classes]

    n_test_samples = confusion_matrix.sum()
    total_correct = per_class_correct.sum()

    accuracy = (float(total_correct) / float(n_test_samples)) * 100.0
    per_class_avg = per_class_acc.mean() * 100.0
    per_class_std = per_class_acc.std() * 100.0

    #.... save confusion matrix to file...
    out_str = "Samples:," + str(n_test_samples) + "\n"
    out_str += "N Classes:," + str(n_classes) + "\n"
    out_str += "Correct:," + str(total_correct) + "\n"
    out_str += "Wrong:," + str(n_test_samples - total_correct) + "\n"
    out_str += "Accuracy:," + str(accuracy) + "\n"
    out_str += "Per class AVG:, " + str(per_class_avg) + "\n"
    out_str += "Per class STD:, " + str(per_class_std) + "\n"
    out_str += "\n\n"
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

        out_str += "," + str(t_samples) + "\n"
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

    out_str += "\n\n\nOnly Errors\n\n" + only_err + "\n\n"

    #now, add the class average per class....
    #...also, if list of ambiguous is available... mark them!
    if not ambiguous is None:
        ambiguous_list = ambiguous.sum(0)

        out_str += "idx,Class,Accuracy,Ambiguous\n"
    else:
        out_str += "idx,Class,Accuracy\n"

    for class_acc, idx, class_name in class_avg_list:
        if class_name == ",":
            class_name = "COMMA"

        out_str += str(idx) + "," + class_name + "," + str(class_acc)
        if not ambiguous is None:
            if ambiguous_list[idx] > 0:
                out_str += ",1"
            else:
                out_str += ",0"
        out_str += "\n"

    try:
        f = open(file_name, 'w')
        f.write(out_str)
        f.close()
    except:
        print("ERROR WRITING RESULTS TO FILE! <" + file_name + ">")


def load_ambiguous(filename, class_dict, skip_failures):
    #first, load all the content from the file...
    file_ambiguous = open(filename, 'r')
    all_ambiguous_lines = file_ambiguous.readlines()
    file_ambiguous.close()

    n_classes = len(class_dict.keys())

    total_ambiguous = 0
    ambiguous = np.zeros((n_classes, n_classes), dtype=np.int32)
    for idx, line in enumerate(all_ambiguous_lines):
        #split and check...
        parts = line.split(',')
        if len(parts) != 2:
            print("Invalid content found in ambiguous file!")
            print("Line " + str(idx + 1) + ": " + line)
            return None

        grapheme_1 = parts[0].strip()
        grapheme_2 = parts[1].strip()

        #check if valid ...
        if not grapheme_1 in class_dict.keys():
            print("Invalid class found: " + str(grapheme_1) + ".")
            print("Line " + str(idx + 1) + ": " + line)

            if skip_failures:
                continue
            else:
                return

        if not grapheme_2 in class_dict.keys():
            print("Invalid class found: " + str(grapheme_2) + ".")
            print("Line " + str(idx + 1) + ": " + line)

            if skip_failures:
                continue
            else:
                return

        #...they are valid, now build the dictionary but using their mapped versions
        class1 = class_dict[grapheme_1]
        class2 = class_dict[grapheme_2]

        #...mark them as ambiguous (symmetric relation)
        if ambiguous[class1, class2] == 0:
            ambiguous[class1, class2] = 1
            ambiguous[class2, class1] = 1

            total_ambiguous += 1


    return ambiguous, total_ambiguous
