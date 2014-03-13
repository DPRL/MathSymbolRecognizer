/*
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
*/


//Compile using:
//	gcc -shared adaboost_c45.c -o adaboost_c45.so
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "c45.h"

typedef struct Boosted_C45 {
	C45_node_t** 	trees;
	double* 		alphas;	
	double**		train_weights;
	int				n_samples;
	int				n_classes;
	int 			n_trees;
	double			train_error;
} Boosted_C45;

void str_set_current_time( char* buffer ){
	time_t now;  
	struct tm* timeinfo;
	
	time(&now);
	timeinfo = localtime(&now);
	strftime (buffer, 50, "%Y-%m-%d %H:%M:%S", timeinfo);
}

double adaboost_get_training_error(Boosted_C45* classifier){
    return classifier->train_error;
}

double adaboost_compute_error( C45_node_t* tree, double** samples, int* labels, int n_samples, double* distribution, double* out_errors)
{
	double total_error = 0.0;
	
	int i, value;
	for ( i = 0; i < n_samples; i++ )
	{
		value = c45_node_evaluate( tree, samples[i] );
		if ( value == labels[i] ){
			//No error
			out_errors[i] = 1;
		} else {
			//Misclassified...
			out_errors[i] = -1;
			total_error += distribution[i];
		}
	}
	
	return total_error;
}

int boosted_c45_classify(Boosted_C45* classifier, double* values) {
	double* output_class = (double *)calloc( sizeof(double), classifier->n_classes );
	
	int max_class = -1;
	
	int i;
	for ( i = 0; i < classifier->n_trees; i++){
		int label = c45_node_evaluate( classifier->trees[i], values );
		
		output_class[label] += classifier->alphas[i];
		
		//Choose maximum...
		if ( max_class == -1){
			//First maximum...
			max_class = label;
		} else {
			if (output_class[max_class] < output_class[label] ){
				//New maximum...
				max_class = label;
			}
		}		
	}
	
	free( output_class );
	
	return max_class;
}

void boosted_c45_probabilistic_classify(Boosted_C45* classifier, double* values, double* output_class){
	int i, k;
	for (i = 0; i < classifier->n_classes; i++){
		output_class[i] = 0.0;
	}
	
	double total_alpha = 0.0;
	double* labels;
	double total_weight;
	double t_weight;
	for ( i = 0; i < classifier->n_trees; i++){
		labels = c45_node_weighted_evaluate( classifier->trees[i], values );
		
		//get total weight
		total_weight = 0.0;
		for (k = 0; k < classifier->n_classes; k++){
			total_weight += labels[k];
		}
		
		for (k = 0; k < classifier->n_classes; k++){
			//Probability of class k in current tree...
			t_weight = labels[k] / total_weight;
			
			//add the alpha value multiplied by the current probability
			output_class[k] += classifier->alphas[i] * t_weight;			
		}
		
		total_alpha += classifier->alphas[i];    
	}
	
	//Finally, normalize....
	for (k = 0; k < classifier->n_classes; k++){
		output_class[k] /= total_alpha;
	}
}

double boosted_c45_compute_error(Boosted_C45* classifier, double** p_samples, int* labels, int n_samples){
	int errors = 0;
	
	//Evaluate classifier against the labeled data
	int i, predicted;
	for ( i = 0; i < n_samples; i++ ) 
	{
		predicted = boosted_c45_classify(classifier, p_samples[i]);
		
		if (predicted != labels[i] ){
			errors++;
		}
	}
	
	return (double)errors / (double)n_samples;
}

double boosted_c45_compute_error_external(Boosted_C45* classifier, double* samples, int* labels, int n_samples, int n_attributes){
	int errors = 0;

	//Evaluate classifier against the labeled data
	int i, predicted;
	double* p_sample;
	for ( i = 0; i < n_samples; i++ )
	{
	    p_sample = samples + (n_attributes * i);
		predicted = boosted_c45_classify(classifier, p_sample);

		if (predicted != labels[i] ){
			errors++;
		}
	}

	return (double)errors / (double)n_samples;
}

/*
	Create a boosted ensemble of C4.5 Decision trees using the given training data...
*/
Boosted_C45* created_boosted_c45(double* samples, int* labels, int* att_types, 
								int n_samples, int n_attributes, int n_classes, 
								int max_rounds, int max_splits, int verbose, char* verb_prefix)
{
	//do boosting ... 
	
	//...prepare data...
	double** p_samples = (double **)calloc(n_samples, sizeof(double *));
	double** train_weights = (double **)calloc(n_samples, sizeof(double *));
	int i;
	for ( i = 0; i < n_samples; i++ ){
		p_samples[i] = samples + (n_attributes * i);
		train_weights[i] = (double *)calloc(max_rounds, sizeof(double));
	}
		
	//initially all samples have the same weight ...
	double init_value = 1.0 / n_samples;
	double* distribution = (double *)calloc(sizeof(double), n_samples);
	for ( i = 0; i < n_samples; i++){
		distribution[i] = init_value;
	}

	int T = max_rounds - 1;
	int t = 0;
	
	double* tempo_alphas = (double *)calloc( sizeof(double), max_rounds );
	C45_node_t** tempo_trees = (C45_node_t**)calloc( sizeof(C45_node_t*), max_rounds);	
	double* tempo_errors = (double *)calloc( sizeof(double), n_samples ) ;
	
	//use AdaBoost.M1
	char str_buffer[50];
	//Create buffer for splits...
	C45_split_buffer* split_buffer = c45_split_buffer_create(n_samples, n_classes);
	
	while ( t <= T ){		
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "%s - [%s] -> Boosting = Round #%d \n", verb_prefix, str_buffer, t + 1 );
		}
		
		//Copy the current weights...
		for ( i = 0; i < n_samples; i++){
			train_weights[i][t] = distribution[i];
		}
		
		//Get the current weighted majority class...
		C45_count_info c_info = c45_tree_counts_by_class(labels, distribution, n_samples, n_classes);
		
		int majority = c_info.majority_class;
		
		free( c_info.weighted_count );
		free( c_info.absolute_count );
		
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "> %s - [%s] -> Round #%d, Training \n", verb_prefix, str_buffer, t + 1 );
		}
		
		//Build the C4.5 decision tree...
		tempo_trees[t] = c45_tree_construct_rec( p_samples, labels, att_types, distribution, 
								n_samples,  n_attributes, n_classes, majority, 
								0, max_splits, 1, split_buffer );
																		
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "> %s - [%s] -> Round #%d, Pruning \n", verb_prefix, str_buffer, t + 1 );
		}
		
		//<TEMPORAL>
		//... save the tree...
		//sprintf( str_buffer, "tempo_tree_r%d.tree", t );
		//c45_save_to_file( tempo_trees[t], str_buffer );
		//printf( "> Temporal saved to file!\n");
		//</TEMPORAL>
		
		
		//...and prune the tree....
		c45_prune_tree( tempo_trees[t], 1, 0.25 );
		
		//...Now, calculate the weighted error....
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "> %s - [%s] -> Round #%d, Evaluating \n", verb_prefix, str_buffer, t + 1 );
		}
		
		double et = adaboost_compute_error( tempo_trees[t], p_samples, labels, n_samples, distribution, tempo_errors);
		
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "> %s - [%s] -> Round #%d, ---> e = %f \n", verb_prefix, str_buffer, t + 1, et );
		}
		
		//...stop criterion...
		if ( et >= 0.5 ){
			T = t - 1;
			break;
		}  
		if ( et == 0.0 ){
			//A "perfect" tree...
			//no errors = no changes in weights...
			T = t;
			break;
		}
		
		//log => natural logarithm
		double alpha_t = 0.5 * log( (1.0 - et) / et );
		
		tempo_alphas[t] = alpha_t;
		
		if ( verbose ){
			str_set_current_time( str_buffer );			
			printf( "> %s - [%s] -> Round #%d, ---> Preparing for next round \n", verb_prefix, str_buffer, t + 1 );
		}
		
		//... update distributions...
		double total_weights = 0.0;
		for ( i = 0; i < n_samples; i++ ){
			distribution[i] = ( distribution[i] ) * exp( -tempo_errors[i] * alpha_t );
			total_weights += distribution[i];
		}
		
		//...normalize distribution...
		for ( i = 0; i < n_samples; i++ ){
			distribution[i] /= total_weights;			
		}
		
        t++;				
	}
	
	//Destroy buffer of splits..
	c45_split_buffer_destroy(split_buffer);
	
	//Create struct that will be returned...
	Boosted_C45* classifier = (Boosted_C45*)calloc(sizeof(Boosted_C45), 1);
	
	classifier->n_samples = n_samples;
	classifier->n_trees = T + 1;
	classifier->trees = (C45_node_t**)calloc( sizeof(C45_node_t*), classifier->n_trees );
	classifier->alphas = (double *)calloc( sizeof(double), classifier->n_trees );
	for ( i = 0; i < classifier->n_trees; i++){
		classifier->trees[i] = tempo_trees[i];
		classifier->alphas[i] = tempo_alphas[i];
	}
	classifier->n_classes = n_classes;	
	classifier->train_error = boosted_c45_compute_error(classifier, p_samples, labels, n_samples);
	classifier->train_weights = train_weights;
	
	if ( verbose ){
		str_set_current_time( str_buffer );			
		printf( "> %s - [%s] -> Final Training Accuracy = %f \n", verb_prefix, str_buffer, (1.0 - classifier->train_error) * 100.0 );
	}	
	
	//Release memory ...
	free( tempo_alphas );
	free( tempo_trees );
	free( tempo_errors );	
	
	free( distribution );
	free( p_samples );
	
	return classifier;
}

/*
=======================================================================
	Release memory...
=======================================================================
*/
void release_boosted_c45(Boosted_C45* classifier)
{
	//First, release all the trees...
	int i;
	for (i = 0; i < classifier->n_trees; i++){
		//Release each tree
		c45_node_release( classifier->trees[i], 1 );
	}
	//Release the array of trees...
	free( classifier->trees );
	
	//Release the array of alphas...
	free( classifier->alphas );
	
	//if has train information...
	if ( classifier->train_weights != 0){
		//...release each sub array
		for (i = 0; i < classifier->n_samples; i++){
			free( classifier->train_weights[i] );
		}
		//...finally release the array of arrays.
		free(classifier->train_weights);
	}
	
	//Finally, release the classifier...
	free( classifier );
}


/*
=======================================================================
	File Handling....
=======================================================================
*/
int boosted_c45_save(Boosted_C45* classifier, char* file_name){
	FILE* out_file = fopen( file_name , "wb" );
	if ( out_file == 0 ){
		printf( "Could not write to %s \n", file_name );
		return 0;
	}
	
	//Save general attributes...
	fwrite( &classifier->n_trees, sizeof(int), 1, out_file );
	fwrite( &classifier->n_classes, sizeof(int), 1, out_file );	
	fwrite( &classifier->train_error, sizeof(double), 1, out_file );
	
	//... alphas ...
	fwrite( classifier->alphas, sizeof(double), classifier->n_trees, out_file);
	
	//... trees ...
	int i;
	for (i = 0; i < classifier->n_trees; i++ )
	{
		//... Save the tree ...
		c45_append_node_to_file( classifier->trees[i], out_file  );
	}
	
	fclose( out_file );
	
	return 1;
}

int boosted_c45_save_training_weights(Boosted_C45* classifier, char* file_name, int text_mode ){
	FILE* out_file;
	
	if (text_mode == 1){
		out_file = fopen( file_name , "w" );
	} else {
		out_file = fopen( file_name , "wb" );
	}
	
	if ( out_file == 0 ){
		printf( "Could not write to %s \n", file_name );
		return 0;
	}
	
	int i, k;
	
	if (text_mode == 1 ){
		//Save T
		fprintf(out_file, "%d trees \n", classifier->n_trees );
		//Save N samples
		fprintf(out_file, "%d samples \n", classifier->n_samples );		
		//Save N classes
		fprintf(out_file, "%d classes \n", classifier->n_classes );
		//Save Alphas
		fprintf(out_file, "alphas \n");
		for (i = 0; i < classifier->n_trees; i++ ){
			if (i == 0){
				//The first...
				fprintf(out_file, "%f", classifier->alphas[i] );
			} else {
				//The rest...
				fprintf(out_file, ", %f", classifier->alphas[i] );
			}
			
			if (i == classifier->n_trees - 1){
				//The last...
				fprintf(out_file, "\n");
			}
		}
		
		//Save Training Weights
		fprintf(out_file, "alphas \n");
		for (i = 0; i < classifier->n_samples; i++){
			for (k = 0; k < classifier->n_trees; k++){
				if (k == 0){
					//The first...
					fprintf(out_file, "%f", classifier->train_weights[i][k] );
				} else {
					//The rest...
					fprintf(out_file, ", %f", classifier->train_weights[i][k] );
				}
				
				if (k == classifier->n_trees - 1){
					//The last...
					fprintf(out_file, "\n");
				}
			}
		}
	} else { 
		//Save T
		fwrite( &classifier->n_trees, sizeof(int), 1, out_file );
		//Save N samples
		fwrite( &classifier->n_samples, sizeof(int), 1, out_file );
		//Save N classes
		fwrite( &classifier->n_classes, sizeof(int), 1, out_file );
		//Save Alphas
		fwrite( classifier->alphas, sizeof(double), classifier->n_trees, out_file);
		//Save Training Weights
		for (i = 0; i < classifier->n_samples; i++){
			//..Save current array...
			fwrite( classifier->train_weights[i], sizeof(double), classifier->n_trees, out_file);
		}
	}
	
	
	fclose( out_file );
	
	return 1;
}

Boosted_C45* boosted_c45_load( char* file_name){
	FILE* in_file = fopen( file_name , "rb" );
	
	if ( in_file == 0 ){
		printf( "Could not read file %s \n", file_name );
		return 0;
	}
	
	Boosted_C45* classifier = (Boosted_C45*)calloc( sizeof(Boosted_C45), 1 );
	
	//Load general attributes...
	fread( &classifier->n_trees, sizeof(int), 1, in_file );
	fread( &classifier->n_classes, sizeof(int), 1, in_file );	
	fread( &classifier->train_error, sizeof(double), 1, in_file );
	
	//... alphas ...
	classifier->alphas = (double *)calloc( sizeof(double), classifier->n_trees );
	fread( classifier->alphas, sizeof(double), classifier->n_trees, in_file);
	
	//... trees ...
	classifier->trees = (C45_node_t **)calloc( sizeof(C45_node_t*), classifier->n_trees );
	int i;
	for (i = 0; i < classifier->n_trees; i++ )
	{
		//... Save the tree ...
		classifier->trees[i] = c45_load_node_from_file(in_file, 1, 0 );		
	}
	
	//weights are not loaded directly from main file
	classifier->train_weights = 0;
	
	fclose( in_file );
	
	return classifier;
}

Boosted_C45* boosted_c45_load_training_weights(char* file_name){
	FILE* in_file = fopen( file_name , "rb" );

	Boosted_C45* classifier = (Boosted_C45*)calloc( sizeof(Boosted_C45), 1 );
	
	//Only the training weights will be loaded....
	//Load T
	fread( &classifier->n_trees, sizeof(int), 1, in_file );
	//Load N samples
	fread( &classifier->n_samples, sizeof(int), 1, in_file );
	//Load N classes
	fread( &classifier->n_classes, sizeof(int), 1, in_file );
	//Load Alphas
	classifier->alphas = (double *)calloc( sizeof(double), classifier->n_trees );
	fread( classifier->alphas, sizeof(double), classifier->n_trees, in_file);

	int i;
	//Load Training Weights
	classifier->train_weights = (double **)calloc(sizeof(double *), classifier->n_samples);
	for (i = 0; i < classifier->n_samples; i++){
		//..Load current array...
		classifier->train_weights[i] = (double *)calloc(sizeof(double), classifier->n_trees);
		fread( classifier->train_weights[i], sizeof(double), classifier->n_trees, in_file);	
	}
	
	fclose( in_file );
	
	return classifier;
}

/*
=======================================================================
	Retrieve basic properties....
=======================================================================
*/
int boosted_c45_get_n_trees(Boosted_C45* classifier){
	return classifier->n_trees;
}

int boosted_c45_get_n_samples(Boosted_C45* classifier){
	return classifier->n_samples;
}

int boosted_c45_get_n_classes(Boosted_C45* classifier){
	return classifier->n_classes;
}

void boosted_c45_copy_alphas(Boosted_C45* classifier, double* out_alphas){
	int i;
	for (i = 0; i < classifier->n_trees; i++){
		out_alphas[i] = classifier->alphas[i];
	}
}

void boosted_c45_copy_weights(Boosted_C45* classifier, double* out_weights){
	int i, k;
	for (i = 0; i < classifier->n_samples; i++){
		for (k = 0; k < classifier->n_trees; k++){
			out_weights[i * classifier->n_trees + k] = classifier->train_weights[i][k];
		}
	}
}
