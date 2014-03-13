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
import pickle
from sklearn.preprocessing import StandardScaler
from dataset_ops import *

#=====================================================================
#  Uses the provided PCA paramaters and applies them to a given
#  dataset
#
#  Created by:
#      - Kenny Davila (Feb 1, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 1, 2012-2014)
#
#=====================================================================


def load_PCA_parameters(file_name):
    file_params = open(file_name, 'r')

    normalization = pickle.load(file_params)
    pca_vector = pickle.load(file_params)
    pca_k = pickle.load(file_params)

    file_params.close()

    return (normalization, pca_vector, pca_k)


def project_PCA(training_set, eig_vectors, k):
    #in case that K > # of atts, then just clamp....
    n_samples = training_set.shape[0]
    n_atts = training_set.shape[1]

    k = min(k, n_atts)

    projected = np.zeros((n_samples, k))

    #...for each sample...
    for n in range(n_samples):
        x = np.mat(training_set[n, :])

        #...for each eigenvector
        for i in range(k):
            #use dot product to project...
            #p = np.dot(x, eig_vectors[i])[0, 0]
            p = np.dot(x, eig_vectors[i])[0, 0]

            projected[n, i] = p.real

    return projected


def main():
    #usage check
    if len(sys.argv) != 4:
        print("Usage: python apply_PCA_parameters.py training_set PCA_params output")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training_set")
        print("\tPCA_params\t= Path to the file of the PCA parameters")
        print("\toutput\t= File to output the final dataset with reduced dimensionality")
        return

    input_filename = sys.argv[1]
    params_filename = sys.argv[2]
    output_filename = sys.argv[3]

    #...load training set ...
    print("Loading data....")
    training, labels_l, att_types = load_dataset(input_filename)

    #...the parameters....
    print("Loading parameters...")
    #normalization, pca_vector, pca_k = load_PCA_parameters(params_filename)
    scaler, pca_vector, pca_k = load_PCA_parameters(params_filename)

    #...apply normalization...
    print("Normalizing data...")
    #new_data = normalize_data_from_params(training, normalization)
    new_data = scaler.transform(training)

    #...transform....
    print("Applying transformation...")
    projected = project_PCA(new_data, pca_vector, pca_k)

    #...save final version...
    print("Saving to file...")
    final_atts = np.zeros((pca_k, 1), dtype=np.int32)
    final_atts[:, :] = 1  # Continuous attributes

    save_dataset_string_labels(projected, labels_l, final_atts, output_filename)

    print("Finished!")

main()

