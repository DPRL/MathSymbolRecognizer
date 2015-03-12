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
import cPickle
from symbol_classifier import SymbolClassifier

def main():
    # usage check...
    if len(sys.argv) < 2:
        print("Usage: python test_classifer.py classifier")
        print("Where")
        print("\tclassifier\t= Path to trained symbol classifier")
        return

    classifier_file = sys.argv[1]

    print("Loading classifier")

    in_file = open(classifier_file, 'rb')
    classifier = cPickle.load(in_file)
    in_file.close()

    if not isinstance(classifier, SymbolClassifier):
        print("Invalid classifier file!")
        return

    # get mapping
    classes_dict = classifier.classes_dict
    classes_l = classifier.classes_list
    n_classes = len(classes_l)

    # create test characters from points
    sample_x = [[(-0.8, -0.8), (0.1, 0.12), (0.8, 0.79)], [(-0.85, 0.79), (0.01, 0.005), (0.79, -0.83)]]
    sample_1 = [[(0.15, 0.7), (0.2, 1.0), (0.21, -1.2)], [(0.05, -1.19), (0.3, -1.25)]]
    sample_eq = [[(-1.5, -0.4), (1.5, -0.4)], [(-1.5, 0.4), (1.5, 0.4)]]

    # classify them
    class_x = classifier.classify_points(sample_x)
    class_1 = classifier.classify_points(sample_1)
    class_eq = classifier.classify_points(sample_eq)

    print("X classified as " + class_x)
    print("1 classified as " + class_1)
    print("= classified as " + class_eq)

    # now with confidence ...
    classes_x = classifier.classify_points_prob(sample_x, 3)
    classes_1 = classifier.classify_points_prob(sample_1, 3)
    classes_eq = classifier.classify_points_prob(sample_eq, 3)

    print("X top classes are: " + str(classes_x))
    print("1 top classes are: " + str(classes_1))
    print("= top classes are: " + str(classes_eq))

if __name__ == '__main__':
    main()