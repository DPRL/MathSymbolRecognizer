"""
    DPRL Math Symbol Recognizers
    Copyright (c) 2012-2016 Kenny Davila, Richard Zanibbi

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
from load_inkml import *
from symbol_classifier import SymbolClassifier

def main():
    # usage check...
    if len(sys.argv) < 3:
        print("Usage: python test_classifer.py classifier inkml_file")
        print("Where")
        print("\tclassifier\t= Path to trained symbol classifier")
        print("\tinkml_file\t= Path to inkml file to load")
        return

    classifier_file = sys.argv[1]
    inkml_file = sys.argv[2]

    print("Loading classifier")

    in_file = open(classifier_file, 'rb')
    classifier = cPickle.load(in_file)
    in_file.close()

    if not isinstance(classifier, SymbolClassifier):
        print("Invalid classifier file!")
        return

    ground_truth_available = True
    try:
        symbols = load_inkml(inkml_file, True )
    except:
        ground_truth_available = False
        try:
            symbols = load_inkml(inkml_file, False)
        except:
            print("Failed processing: " + inkml_file)
            return

    # run the classifier for each symbol in the file
    for symbol in symbols:
        trace_ids = [str(trace.id) for trace in symbol.traces]
        print("Symbol id: " + str(symbol.id))
        print("=> Traces: " + ", ".join(trace_ids))
        print("=> Ground Truth Class: " + symbol.truth)

        # classify
        main_class = classifier.classify_symbol(symbol)
        print("=> Predicted Class: " + main_class)

        top_5 = classifier.classify_symbol_prob(symbol, 5)
        top_5_desc = [class_name + " ({0:.2f}%)".format(prob * 100) for class_name, prob in top_5]
        print("=> Top 5 classes: " + ",".join(top_5_desc))
        print("")

    print("Finished")

if __name__ == '__main__':
    main()