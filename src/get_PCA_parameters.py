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
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from dataset_ops import *

#=====================================================================
#  Find the PCA parameters for a given training set. The learn
#  parameters can be later applied for dimensionality reduction
#  of the dataset
#
#  Created by:
#      - Kenny Davila (Feb 1, 2012-2014)
#  Modified By:
#      - Kenny Davila (Feb 1, 2012-2014)
#
#=====================================================================


def get_PCA( training_set ):
    #get the covariance matrix...
    cov_matrix = np.cov(training_set.transpose())

    #...obtain eigenvectors and eigenvalues...
    eig_values, eig_matrix = np.linalg.eig(cov_matrix)

    #...sort them....
    pair_list = [(eig_values[i], i) for i in range(cov_matrix.shape[0])]
    pair_list = sorted( pair_list, reverse=True )

    sorted_values = []
    sorted_vectors = []

    for eigenvalue, idx in pair_list:
        sorted_values.append(eigenvalue)
        sorted_vectors.append(np.mat(eig_matrix[:, idx]).T)

    return sorted_values, sorted_vectors


def get_variance_K(values, variance):
    #...get the variance percentages...
    n_atts = len(values)
    total_variance = 0.0
    cumulative_variance = []
    for i in xrange(n_atts):
        eigenvalue = abs(values[i])

        total_variance += eigenvalue
        cumulative_variance.append(total_variance)

    #...normalize...
    for i in xrange(n_atts):
        cumulative_variance[i] /= total_variance

        if cumulative_variance[i] >= variance:
            return i + 1, cumulative_variance

    return n_atts, cumulative_variance


def save_PCA_parameters(file_name,  normalization, pca_vector, pca_k):
    file_params = open(file_name, 'w')
    pickle.dump(normalization, file_params)
    pickle.dump(pca_vector, file_params)
    pickle.dump(pca_k, file_params)
    file_params.close()


def main():
    #usage check
    if len(sys.argv) != 5:
        print("Usage: python get_PCA_parameters.py training_set output_params var_max k_max")
        print("Where")
        print("\ttraining_set\t= Path to the file of the training_set")
        print("\toutput_params\t= File to output PCA preprocessing parameters")
        print("\tvar_max\t\t= Maximum percentage of variance to add")
        print("\tk_max\t\t= Maximum number of Principal Components to Add")
        return

    #get parameters
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    try:
        var_max = float(sys.argv[3])
        if var_max <= 0.0 or var_max > 1.0:
            print("Invalid var_max value! ")
            return
    except:
        print("Invalid var_max value! ")
        return

    try:
        k_max = int(sys.argv[4])
        if k_max <= 0:
            print("Invalid k_max value!")
            return
    except:
        print("Invalid k_max value!")
        return


    #...load training set ...
    print("Loading data....")
    training, labels_l, att_types = load_dataset(input_filename)

    print("Data loaded! ... normalizing ...")
    #new_data, norm_params = normalize_data(training)
    scaler = StandardScaler()
    new_data = scaler.fit_transform(training)

    print( "Normalized! ... Applying PCA ..." )
    values, vectors = get_PCA(new_data)

    variance_k, all_variances = get_variance_K(values, var_max)

    final_k = min(variance_k, k_max)

    print("Final K = " + str(final_k) + ", variance = " + str(all_variances[final_k - 1]) )

    print("Saving Parameters ... ")
    save_PCA_parameters(output_filename, scaler, vectors, final_k)

    print("Finished!")


main()
