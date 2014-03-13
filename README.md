# DPRL Math Symbol Recognizers 

Copyright (c) 2012-2014 Kenny Davila, Richard Zanibbi
***

RIT DPRL Math Symbol Recognizers is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

RIT DPRL Math Symbol Recognizers is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
RIT DPRL Math Symbol Recognizers.  If not, see <http://www.gnu.org/licenses/>.

Contact:
* Kenny Davila: kxd7282@rit.edu
* Richard Zanibbi: rlaz@cs.rit.edu

***
The system divides is composed of different tools for data extraction, training
and evaluation and other miscellaneous tools for isolated math symbol
recognition. 

A README file is included in the doc/ directory for each tool that describes
its purpose, how to use it and what its parameters are.

A README file is included in the doc/ directory for each tool that describes
its purpose, how to use it and what its parameters are.

The executable scripts on this release are the following:

* Preprocessing of data:
        apply_PCA_parameters.py
        correct_labels.py
	    get_enhanced_clustered_set.py
	    get_PCA_parameters.py
	    get_training_set.py
* Analysis of datasets:
        count_common.py    
		dataset_info.py
		extract_symbol.py

* Training a symbol classifier:
        random_forest_classify.py
    	svm_lin_classifier.py
		svm_rbf_classifier.py
		train_adaboost.py
		train_c45.py
* Tools for evaluation
        boosted_test.py
    	parallel_evaluate.py
		parallel_prob_evaluate.py



***
# SOURCE FILES
Source code (in Python and C) is provided in the src/ directory.