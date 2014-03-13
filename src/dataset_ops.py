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
#  Most general functions used to load and save datasets from
#  text files separated by semi-colon
#
#  Created by:
#      - Kenny Davila (Dic 1, 2013)
#  Modified By:
#      - Kenny Davila (Jan 26, 2012-2014)
#        - Added function to save dataset
#      - Kenny Davila (Feb 1, 2012-2014)
#        - Added functions for data normalization
#      - Kenny Davila (Feb 10, 2012-2014)
#        - empty lines are now skipped!
#      - Kenny Davila (Feb 18, 2012-2014)
#        - load_dataset is now more memory efficient
#
#=====================================================================

#=========================================
# Loads a dataset from a file
#   returns
#     Samples, Labels, Att Types
#     (NP Matrix, List, NP Matrix)
#=========================================
def load_dataset(file_name):
    try:
        
        data_file = open(file_name, 'r')
        lines = data_file.readlines()
        data_file.close()

        #now process every line...
        #first line must be the attribute type for each feature...
        att_types_s1 = lines[0].split(';')
        att_types_s2 = [ s.strip().upper() for s in att_types_s1 ]
        n_atts = len( att_types_s2 )

        #..proces the attributes...
        att_types = np.zeros( (n_atts, 1), dtype=np.int32 )
        for i in xrange(n_atts):
            if att_types_s2[i] == 'D':
                att_types[i] = 2
            else:
                att_types[i] = 1

        estimated_samples = len(lines) - 1
        tempo_samples = np.zeros( (estimated_samples, n_atts), dtype = np.float64 )

        count_samples = 0
        labels_l = []
        for i in xrange(1, len(lines)):
            values_s = lines[i].split(';')
            if len(values_s) == 1:
                if values_s[0].strip() == "":
                    #just skip the empty line...
                    continue

            #assume last value is class label
            label = values_s[-1].strip()
            del values_s[-1]

            #read values...
            values = []
            for idx, att_type in enumerate(att_types):
                values.append( float( values_s[idx].strip() ) )


            #validate number of attributes on the sample
            if len(values) != n_atts:
                print("Number of values is different to number of attributes")
                print("Atts: " + str(n_atts))
                print("Values: " + str(len(values)))
                return None, None, None

            #add sample        
            for att in xrange(n_atts):
                tempo_samples[count_samples, att] = values[att]

            count_samples += 1

            labels_l.append( label )

        n_samples = count_samples
        if n_samples != estimated_samples:
            samples = tempo_samples[:n_samples, :].copy()
            tempo_samples = None
        else:
            samples = tempo_samples

        
        return samples, labels_l, att_types
        
    except Exception as e:
        print("Error loading dataset from file")
        print( e )
        return None, None, None
        


#==============================================
#  Generates a mapping for the unique set of
#  classes present on the given list of labels
#==============================================
def get_label_mapping(labels_l):
    classes_dict = {}
    classes_l = []

    #... for each sample...
    n_samples = len(labels_l)
    for i in xrange( n_samples ):
        label = labels_l[ i ]

        #check mapping of labels...
        if not label in classes_dict:
            #...add label to mapping...
            label_val = len( classes_l )
            classes_l.append( label )

            classes_dict[ label ] = label_val

    return classes_dict, classes_l

def get_mapped_labels(labels_l, classes_dict):
    n_samples = len( labels_l )

    #...for the mapped labels...
    labels = np.zeros( (n_samples, 1), dtype = np.int32 )

    #... for each sample ...
    for i in xrange( n_samples ):
        #get current label...
        label = labels_l[ i ]
        #get mapped label...
        label_val = classes_dict[ label ]
        #add mapped label
        labels[ i, 0 ] = label_val

    return labels


def load_ds_sources(file_name):
    try:
        #read all lines...
        file_source = open(file_name, 'r')
        lines = file_source.readlines()
        file_source.close()

        all_sources = []

        #get filename, symbol id per each symbol in DS
        for i in range(len(lines)):
            values_s = lines[i].split(',')

            #should contain 2 values...
            if len(values_s) != 2:
                print( "Invalid line <" + str(i) + "> in Auxiliary file: " + lines[i] )
                return None
            else:
                file_path = values_s[0].strip()
                sym_id = int( values_s[1] )

                all_sources.append( (file_path, sym_id) )

        return all_sources

    except Exception as e:
        print(e)
        return None

def save_dataset(data, labels, att_types, out_file):
    n_samples = np.size(data, 0)

    try:
        out_file = open(out_file, 'w')
    except:
        print( "File <" + out_file + "> could not be created")
        return

    #...writing first header....
    content = ''
    #print as headers the types for each feature...
    n_atts = np.size(att_types, 0)
    for i in range(n_atts):
        if i > 0:
            content += '; '

        if att_types[i, 0] == 2:
            content += 'D'
        else:
            content += 'C'

    content += '\r\n'
    out_file.write(content)

    #...now, write the samples....
    content = ''
    for idx in range(n_samples):
        #...add the values...
        line = ''
        for k in range(n_atts):
            if k > 0:
                line += '; '

            line += str(data[idx, k])

        #... add the label...
        line += "; " + str(labels[idx, 0]) + "\r\n"

        content += line
        #....check if buffer is full!
        if len(content) >= 50000:
            out_file.write(content)
            content = ''

    #....write any remaining content
    out_file.write(content)

    out_file.close()

def save_label_mapping(base_classes, extra_mapping, file_name):
    #...create....
    try:
        out_file = open(file_name, 'w')
    except:
        print( "File <" + file_name + "> could not be created")
        return

    #... First, write the sizes of both mappings
    content = str(len(base_classes)) + ";" + str(len(extra_mapping.keys())) + "\r\n"
    out_file.write(content)

    #...write the original class list...
    content = ''
    for c_class in base_classes:
        content += c_class + "\r\n"
    out_file.write(content)

    #...write now the extra mapping...
    content = ''
    for key in extra_mapping:
        content += str(key) + ";" + str(extra_mapping[key]) + "\r\n"

    out_file.write(content)

    #...close...
    out_file.close()


def load_label_mapping(file_name):
    #...open....
    try:
        data_file = open(file_name, 'r')
        lines = data_file.readlines()
        data_file.close()

        #first line should contain the size of the mappings..
        sizes_s = lines[0].split(';')
        n_classes = int(sizes_s[0])
        n_mapped = int(sizes_s[1])

        print "...Loading class mapping..."
        print "N-real-classes: " + str(n_classes)
        print "N-virtual-classes: " + str(n_mapped)

        class_l = []
        class_dict = {}
        for i in range(n_classes):
            label = lines[1 + i].strip()

            class_l.append(label)
            class_dict[label] = i

        class_mapping = {}
        for i in range(n_mapped):
            #get the key -> value pair...
            mapped = lines[1 + n_classes + i].split(';')

            #...split...
            new_label = int(mapped[0])
            original_label = int(mapped[1])

            class_mapping[new_label] = original_label

        return class_l, class_dict, class_mapping
    except Exception as e:
        print(e)
        return None, None


def append_symbols(symbols, out_file):
    n_samples = len(symbols)

    print("...adding samples " + str(n_samples) + " to output file...")

    n_atts = 0
    content = ''
    for idx, symbol in enumerate(symbols):
        sample = symbol.getFeatures() + [ symbol.truth ]
        if idx == 0:
            n_atts = len(sample) - 1

        line = ''
        for i, v in enumerate(sample):
            if i > 0:
                line += '; '

            line += str(v)

        line += '\r\n'
        content += line

        #check if buffer is full!
        if len(content) >= 50000:
            out_file.write(content)
            content = ''

    #write any remaining content
    out_file.write(content)

    print("... samples added to file successfully!")

    return n_atts


def append_dataset(data, labels, out_file):
    #...will append samples in dataset to output file...
    n_samples = np.size(data, 0)
    n_atts = np.size(data, 1)

    print("...adding samples " + str(n_samples) + " to output file...")

    #...now, write the samples....
    content = ''
    for idx in range(n_samples):
        #...add the values...
        line = ''
        for k in range(n_atts):
            if k > 0:
                line += '; '

            line += str(data[idx, k])

        #... add the label...
        line += "; " + str(labels[idx, 0]) + "\r\n"

        content += line
        #....check if buffer is full!
        if len(content) >= 50000:
            out_file.write(content)
            content = ''

    #....write any remaining content
    out_file.write(content)

    print("... samples added to file successfully!")


def append_dataset_string_labels(data, labels, out_file):
    #...will append samples in dataset to output file...
    n_samples = np.size(data, 0)
    n_atts = np.size(data, 1)

    print("...adding samples " + str(n_samples) + " to output file...")

    #...now, write the samples....
    content = ''
    for idx in range(n_samples):
        #...add the values...
        line = ''
        for k in range(n_atts):
            if k > 0:
                line += '; '

            line += str(data[idx, k])

        #... add the label...
        line += "; " + labels[idx] + "\r\n"

        content += line
        #....check if buffer is full!
        if len(content) >= 50000:
            out_file.write(content)
            content = ''

    #....write any remaining content
    out_file.write(content)

    print("... samples added to file successfully!")


def save_dataset_string_labels(data, labels, att_types, out_filename):
    n_samples = np.size(data, 0)

    try:
        out_file = open(out_filename, 'w')
    except:
        print( "File <" + out_filename + "> could not be created")
        return

    #...writing first header....
    content = ''
    #print as headers the types for each feature...
    n_atts = np.size(att_types, 0)
    for i in range(n_atts):
        if i > 0:
            content += '; '

        if att_types[i, 0] == 2:
            content += 'D'
        else:
            content += 'C'

    content += '\r\n'
    out_file.write(content)

    #...now, write the samples....
    content = ''
    for idx in range(n_samples):
        #...add the values...
        line = ''
        for k in range(n_atts):
            if k > 0:
                line += '; '

            line += str(data[idx, k])

        #... add the label...
        line += "; " + str(labels[idx]) + "\r\n"

        content += line
        #....check if buffer is full!
        if len(content) >= 50000:
            out_file.write(content)
            content = ''

    #....write any remaining content
    out_file.write(content)

    out_file.close()


#====================================================
#  Data Normalization: Centering and scaling
#====================================================

#----------------------------------------------------
#  Function that computes normalization parameters
#  and applies them to given data returning a
#  normalized version of it
#----------------------------------------------------
def normalize_data(data):
    params = []
    n_samples = np.size(data,0)
    n_atts = np.size(data, 1)
    new_data = np.zeros((n_samples, n_atts))

    #for each attribute...
    for att in range(n_atts):
        #the mean and std_dev for each att
        cut = data[:, att]
        cut_mean = cut.mean()
        cut_std = cut.std()

        #add to parameters list...
        params.append((cut_mean, cut_std))

        #now normalize...
        if cut_std > 0.0:
            new_data[:, att] = (data[:, att] - cut_mean) / cut_std
        else:
            #only center...
            new_data[:, att] = (data[:, att] - cut_mean)

    return new_data, params

#------------------------------------------------------
#  Function that receives a dataset and normalization
#  parameters, and applies them returning a normalized
#  version of the data
#------------------------------------------------------
def normalize_data_from_params(data, params):
    n_samples = np.size(data,0)
    n_atts = np.size(data, 1)
    new_data = np.zeros((n_samples, n_atts))

    #for each attribute...
    for att in range(n_atts):
        cut_mean, cut_std = params[att]

        #now normalize...
        if cut_std > 0.0:
            new_data[:, att] = (data[:, att] - cut_mean) / cut_std
        else:
            #only center...
            new_data[:, att] = (data[:, att] - cut_mean)

    return  new_data

