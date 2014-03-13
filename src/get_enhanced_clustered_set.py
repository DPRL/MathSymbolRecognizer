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
import os
import sys
import fnmatch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Ward
from traceInfo import *
from mathSymbol import *
from load_inkml import *
from distorter import *
from dataset_ops import *

#=====================================================================
#  generates an enhanced training set from a directory containing
#  inkml files by adding extra samples generated with a distortion
#  model. it also applies clustering for to automatically identify
#  under represented classes.
#
#  Created by:
#      - Kenny Davila (Jan 27, 2012-2014)
#  Modified By:
#      - Kenny Davila (Jan 27, 2012-2014)
#
#=====================================================================

def print_overwrite(text):
    sys.stdout.write("\r")
    sys.stdout.write(" " * 79)
    sys.stdout.write("\r")
    sys.stdout.write(text)


def replace_labels(labels):
    new_labels = []

    for label in labels:
        if label == "\\tg":
            new_label = "\\tan"
        elif label == '>':
            new_label = '\gt'
        elif label == '<':
            new_label = '\lt'
        elif label == '\'':
            new_label = '\prime'
        elif label == '\cdots':
            new_label = '\ldots'
        elif label == '\\vec':
            new_label = '\\rightarrow'
        elif label == '\cdot':
            new_label = '.'
        elif label == ',':
            new_label = 'COMMA'
        elif label == '\\frac':
            new_label = '-'
        else:
            #unchanged...
            new_label = label

        new_labels.append(new_label)

    return new_labels


def get_symbol_features(symbols, n_atts, verbose):
    if verbose:
        print("...Getting features for samples....")

    #...get features....
    total_extra = len(symbols)
    extra_features = np.zeros((total_extra, n_atts))

    for idx, symbol in enumerate(symbols):
        features = symbol.getFeatures()

        #copy features from list to numpy array
        for k in range(n_atts):
            extra_features[idx, k] = features[k]

        #...some output....
        if verbose and (idx == total_extra - 1 or idx % 10 == 0):
            print_overwrite("...Processed " + str(idx + 1) + " of " + str(total_extra))

    return extra_features


def main():
    #usage check
    if len(sys.argv) < 7:
        print("Usage: python get_enhanced_clustered_set.py inkml_path output min_prc diag_dist max_clusters " +
              "clust_prc [verbose] [count_only]")
        print("Where")
        print("\tinkml_path\t= Path to directory that contains the inkml files")
        print("\toutput\t\t= File name of the output file")
        print("\tmin_prc\t\t= Minimum representation based on (%) of largest class")
        print("\tdiag_dist\t= Distortion factor relative to length of main diagonal")
        print("\tmax_clusters\t= Maximum number of clusters per large class")
        print("\tclust_prc\t= Minimum cluster size based on (%) of largest ")
        print("\tverbose\t= Optional, print detailed messages ")
        print("\tcount_only\t= Will only count what will be the final size of dataset")
        return

    #load and filter the list of files, the result is a list of inkml files only
    try:
        complete_list = os.listdir(sys.argv[1])
        filtered_list = []
        for file in complete_list:
            if fnmatch.fnmatch(file, '*.inkml'):
                filtered_list.append( file )
    except:
        print( "The inkml path <" + sys.argv[1] + "> is invalid!" )
        return

    output_filename = sys.argv[2]

    try:
        min_prc = float(sys.argv[3])
        if min_prc < 0.0:
            print("Invalid minimum percentage")
            return
    except:
        print("Invalid minimum percentage")
        return

    try:
        diag = float(sys.argv[4])
        if diag < 0.0:
            print("Invalid distortion factor")
            return
    except:
        print("Invalid distortion factor")
        return

    try:
        components = int(sys.argv[5])
        if components < 0:
            print("Invalid max_clusters")
    except:
        print("Invalid max_clusters")
        return

    try:
        clust_prc = float(sys.argv[6])
        if clust_prc < 0.0:
            print("Invalid minimum cluster percentage")
            return
    except:
        print("Invalid minimum cluster percentage")
        return

    if len(sys.argv) > 7:
        try:
            verbose = int(sys.argv[7]) > 0
        except:
            print("Invalid value for verbose")
            return
    else:
        #by default, print all messages
        verbose = True

    if len(sys.argv) > 8:
        try:
            count_only = int(sys.argv[8]) > 0
        except:
            print("Invalid value for count_only")
            return
    else:
        #by default...
        count_only = False

    #....read every inkml file in the path specified...
    #....create the initial symbol objects...
    all_symbols = []
    sources = []
    print("Loading samples from files.... ")
    for i in range(len(filtered_list)):
        file_name = filtered_list[i]
        file_path = sys.argv[1] + '//' + file_name
        advance = float(i) / len(filtered_list)
        if verbose:
            print_overwrite(("Processing => {:.2%} => " + file_path).format(advance))

        #print()

        symbols = load_inkml(file_path, True)

        for new_symbol in symbols:
            all_symbols.append(new_symbol)
            sources.append((file_path, new_symbol.id))

    print("")
    print("....samples loaded!")

    #...get features of original training set...
    print("Getting features of base training set...")
    training = None
    labels_l = None
    att_types = None
    n_original_samples = len(all_symbols)
    for idx, symbol in enumerate(all_symbols):
        features = symbol.getFeatures()

        if idx == 0:
            #...create training set ....

            #...first, the feature types...
            n_atts = len(features)
            feature_types = symbol.getFeaturesTypes()
            att_types = np.zeros((n_atts, 1), dtype=np.int32)
            for i in xrange(n_atts):
                if feature_types[i] == 'D':
                    att_types[i] = 2
                else:
                    att_types[i] = 1

            #...for labels...
            labels_l = []

            #...finally, the matrix for training set itself...
            training = np.zeros((n_original_samples, n_atts))

        #copy features from list to numpy array
        for k in range(n_atts):
            training[idx, k] = features[k]

        #copy label
        labels_l.append(symbol.truth)

        #...some output....
        if verbose and (idx == n_original_samples - 1 or idx % 10 == 0):
            print_overwrite("...Processed " + str(idx + 1) + " of " + str(n_original_samples))

    #...correct labels....
    labels_l = replace_labels(labels_l)


    #...scale original data...
    print("Scaling original data...")
    scaler = StandardScaler()
    scaled_training = scaler.fit_transform(training)

    #...save original samples to file....
    if not count_only:
        print("Saving original data...")
        save_dataset_string_labels(training, labels_l, att_types, output_filename)
    #append_dataset_string_labels

    #... identify under-represented classes ...
    #... first, count samples per class ...
    print("Getting counts...")
    count_per_class = {}
    refs_per_class = {}
    for idx in range(n_original_samples):
        #s_label = labels_train[idx, 0]
        s_label = labels_l[idx]

        if s_label in count_per_class:
            count_per_class[s_label] += 1
            refs_per_class[s_label].append(idx)
        else:
            count_per_class[s_label] = 1
            refs_per_class[s_label] = [idx]

    n_classes = len(count_per_class.keys())

    #...distribute data in classes...
    largest_size = 0
    for label in count_per_class:
        #...check largest...
        if count_per_class[label] > largest_size:
            largest_size = count_per_class[label]

    print("Samples on largest class: " + str(largest_size))

    #Now, do the data enhancement....
    distorter = Distorter()

    #....for each class....
    n_extra_samples = 0
    min_elements = int(math.ceil(min_prc * largest_size))
    print("Analyzing data per class...")
    for label in count_per_class:
        current_refs = refs_per_class[label]
        n_class_samples = count_per_class[label]

        if n_class_samples < min_elements:
            print("Class: " + label + ", count = " + str(n_class_samples) + ", adding samples...")

            #class is under represented, generate distorted artificial samples....
            to_create = min_elements - n_class_samples
            n_extra_samples += to_create

            if not count_only:
                #...create...
                extra_samples = []
                for i in range(to_create):
                    #...take one element from original samples
                    base_symbol = all_symbols[current_refs[(i % n_class_samples)]]

                    #...create distorted version...
                    new_symbol = distorter.distortSymbol(base_symbol, diag)

                    #...add...
                    extra_samples.append(new_symbol)

                #...get features....
                extra_features = get_symbol_features(extra_samples, n_atts, verbose)
                #...create labels....
                extra_labels = [label] * len(extra_samples)

                #...now, save the extra samples to file...
                out_file = open(output_filename, 'a')
                append_dataset_string_labels(extra_features, extra_labels, out_file)
                out_file.close()
        else:
            #class has more than enough data, try clustering and then enhancing small clusters...

            print("Class: " + label + ", count = " + str(n_class_samples) + ", clustering...")

            if components == 0:
                print("...Skipping clustering...")
                continue

            #create empty dataset ...
            class_data = np.zeros((n_class_samples, n_atts))
            #fill with samples...
            for i in xrange(n_class_samples):
                class_data[i, :] = scaled_training[current_refs[i], :]

            #apply clustering...
            ward = Ward(n_clusters=components).fit(class_data)
            cluster_labels = ward.labels_

            #separate references per cluster, and get counts...
            counts_per_cluster = [0] * components
            refs_per_cluster = {}
            for i in range(n_class_samples):
                c_label = int(cluster_labels[i])

                if c_label in refs_per_cluster:
                    refs_per_cluster[c_label].append(current_refs[i])
                else:
                    refs_per_cluster[c_label] = [current_refs[i]]

                #increase the count of samples per class...
                counts_per_cluster[c_label] += 1

            #...check minimum size for enhancement
            largest_cluster = max(counts_per_cluster)
            #print "Largest cluster: " + str(largest_cluster)
            min_cluster_size = int(math.ceil(clust_prc * largest_cluster))
            final_counts_per_cluster = []

            #...for each cluster...
            extra_samples = []
            for i in range(components):
                #...if has les than the minimum number of elements...
                if counts_per_cluster[i] < min_cluster_size:
                    #enhance....
                    c_cluster = counts_per_cluster[i]
                    to_create = min_cluster_size - c_cluster
                    cluster_refs = refs_per_cluster[i]

                    n_extra_samples += to_create

                    if not count_only:
                        #...create...
                        for k in range(to_create):
                            #...take one element from original samples
                            base_symbol = all_symbols[cluster_refs[(k % c_cluster)]]
                            #...create distorted version...
                            new_symbol = distorter.distortSymbol(base_symbol, diag)
                            #...modify label (use new mapped label)...
                            #new_symbol.truth = classes_dict[new_symbol.truth]
                            #...add...
                            extra_samples.append(new_symbol)

                    final_counts_per_cluster.append(min_cluster_size)
                else:
                    final_counts_per_cluster.append(counts_per_cluster[i])

                #print counts_per_cluster
                #print final_counts_per_cluster

                print ("Cluster #" + str(i + 1) + ", i. size = " + str(counts_per_cluster[i]) +
                       ", f. size = " + str(final_counts_per_cluster[i]))

            if not count_only:
                #...get features....
                extra_features = get_symbol_features(extra_samples, n_atts, verbose)

                #...create labels matrix....
                extra_labels = [label] * len(extra_samples)

                #...now, save the extra samples to file...
                out_file = open(output_filename, 'a')
                #append_dataset(extra_features, extra_labels, out_file)
                append_dataset_string_labels(extra_features, extra_labels, out_file)
                out_file.close()

            extra_samples = []

    print("Initial size: " + str(n_original_samples))
    print("Extra samples: " + str(n_extra_samples))
    print("Final size: " + str(n_extra_samples + n_original_samples))
    print("Added Ratio: " + str((float(n_extra_samples) / float(n_original_samples)) * 100.0))

    print("Finished!")

main()
