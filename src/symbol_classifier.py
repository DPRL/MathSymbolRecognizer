__author__ = 'Mauricio'

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

