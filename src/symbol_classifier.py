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
from traceInfo import TraceInfo
from mathSymbol import MathSymbol

class SymbolClassifier:
    TypeRandomForest = 1
    TypeSVMLIN = 2
    TypeSVMRBF = 3

    def __init__(self, type, trained_classifier, classes_list, classes_dict, scaler=None, probabilistic=False):
        self.type = type
        self.trained_classifier = trained_classifier
        self.classes_list = classes_list
        self.classes_dict = classes_dict
        self.scaler = scaler
        self.probabilistic = probabilistic

    def predict(self, dataset):
        return self.trained_classifier.predict(dataset)

    def predict_proba(self, dataset):
        return self.trained_classifier.predict_proba(dataset)

    def get_raw_classes(self):
        return self.trained_classifier.classes_

    def get_symbol_from_points(self, points_lists):

        traces = []
        for trace_id, point_list in enumerate(points_lists):
            object_trace = TraceInfo(trace_id, point_list)

            traces.append(object_trace)

            # apply general trace pre processing...
            # 1) first step of pre processing: Remove duplicated points
            object_trace.removeDuplicatedPoints()

            # Add points to the trace...
            object_trace.addMissingPoints()

            # Apply smoothing to the trace...
            object_trace.applySmoothing()

            # it should not ... but .....
            if object_trace.hasDuplicatedPoints():
                # ...remove them! ....
                object_trace.removeDuplicatedPoints()

        new_symbol = MathSymbol(0, traces, '{Unknown}')

        # normalize size and locations
        new_symbol.normalize()

        return new_symbol

    def get_symbol_features(self, symbol):
        # get raw features
        features = symbol.getFeatures()

        # put them in python format
        mat_features = np.mat(features, dtype=np.float64)

        # automatically transform features
        if self.scaler is not None:
            mat_features = self.scaler.transform(mat_features)

        return mat_features

    def classify_points(self, points_lists):
        symbol = self.get_symbol_from_points(points_lists)

        return self.classify_symbol(symbol)

    def classify_points_prob(self, points_lists, top_n=None):
        symbol = self.get_symbol_from_points(points_lists)

        return self.classify_symbol_prob(symbol, top_n)

    def classify_symbol(self, symbol):
        features = self.get_symbol_features(symbol)

        predicted = self.trained_classifier.predict(features)

        return self.classes_list[predicted[0]]


    def classify_symbol_prob(self, symbol, top_n=None):
        features = self.get_symbol_features(symbol)

        try:
            predicted = self.trained_classifier.predict_proba(features)
        except:
            raise Exception("Classifier was not trained as probabilistic classifier")

        scores = sorted([(predicted[0, k], k) for k in range(predicted.shape[1])], reverse=True)

        tempo_classes = self.trained_classifier.classes_
        n_classes = len(tempo_classes)
        if top_n is None or top_n > n_classes:
            top_n = n_classes

        confidences = [(self.classes_list[tempo_classes[scores[k][1]]], scores[k][0]) for k in range(top_n)]

        return confidences


