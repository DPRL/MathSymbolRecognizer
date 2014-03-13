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

/*
#=====================================================================
#   This file represents the C4.5 Decision tree itself, defined by nodes 
#   that can be of three types:
#
#       1 - > Leafs 
#       2 - > Decision nodes for continuous attributes (splits)
#       3 - > Decision nodes for discrete attributes
#
#  Created by:
#      - Kenny Davila (Dec 4, 2013)
#  Modified By:
#      - Kenny Davila (Dec 4, 2013)
#      - Kenny Davila (Feb 5, 2012-2014)
#      - Kenny Davila (Feb 19, 2012-2014)
#
#=====================================================================
*/

//General includes
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


/*
#=====================================================================
#  Global auxiliary functions...
#=====================================================================
*/

//Sorting with quicksort
int c45_quicksort_values( double* values, int* indices, int start, int end){

	if (start >= end){
		//One element is always sorted...
		return (start == end ? 1 : 0);
	}
	
	//printf( " From %d To %d \n ", start, end );
	
	double *p_start = values + start;
	double *p_end = values + end;
	
	int *p_idx_start = indices + start;
	int *p_idx_end = indices + end;
	
	//First, we need a pivot value...
	//Use the smallest of the first two different values...
	double pivot;		
	int all_equal = 1;
	double* p_tempo = p_start;	
		
	//Also, check if there are different values in array
	while (p_tempo <= p_end ){
		//Check if different...
		if (*p_tempo != *p_start){
			all_equal = 0;
			if (*p_tempo < *p_start){
				pivot = *p_tempo;
			} else {
				pivot = *p_start;
			}
			break;
		}
		
		p_tempo++;			
				
	}

	//double pivot = sum / (end - start + 1);

	//if (pivot <= min_value || pivot >= max_value){
	if ( all_equal == 1){
		//All elements are equal ... no further need for sorting...		
		return 1;
	}

	double swap;		
	int swap_idx;

	//Put all elements 
	// <= pivot -> left side
	//  > pivot -> right side
	while (p_start < p_end ){
		//Now, switch elements...
		if ( *p_start > pivot ){
		    //Must switch to the right side...
			//Search for a value on the right side that must be switched
			while ( p_start < p_end && *p_end > pivot ){
			    p_end--;
				p_idx_end --;
			}
			
			//check condition for ending of previous loop
			if (p_start < p_end ){
			    //must swap values...
				swap = *p_start;
				*p_start = *p_end;
				*p_end = swap;
				//...indices too...
				swap_idx = *p_idx_start;
				*p_idx_start = *p_idx_end;
				*p_idx_end = swap_idx;
				//move to the next values...
				p_start ++;
				p_idx_start ++;
				p_end --;
				p_idx_end --;
			}
		} else {
			//Move to next value...
			p_start ++;
			p_idx_start ++;
		}
	}	
	
	//Recursive...
	int left_end = (p_start - values) - ( *p_start <= pivot ? 0 :  1 );

	int total_left = c45_quicksort_values( values, indices, start, left_end);
	int total_right = c45_quicksort_values( values, indices, left_end + 1, end);		
		
	return total_left + total_right;
}


typedef struct C45_sortable_pair {
	int index;
	double value;
} C45_sortable_pair;

int c45_compare(const void* elem1, const void* elem2){
	//double val1 = *((double*)elem1);
	//double val2 = *((double*)elem2);
	
	C45_sortable_pair* pair1 = (C45_sortable_pair*)elem1;
	C45_sortable_pair* pair2 = (C45_sortable_pair*)elem2;
	
	if (pair1->value > pair2->value){
		return 1;
	} else {
		if (pair1->value < pair2->value){
			return -1;
		} else {
			return 0;
		}
	}
}



/*
#=====================================================================
#  Definition of C4.5 Decision Tree nodes and basic functions
#=====================================================================
*/

//Define types of nodes
#define C45_NODE_LEAF 		0
#define C45_NODE_SPLIT 	  	1
#define C45_NODE_DISCRETE 	2

//Define the nodes
typedef struct C45_node C45_node_t;

struct C45_node {
	//Parent node...
    C45_node_t*		parent;
	//For Classification...
	int				n_classes;
    int 			own_class;
	//OTher Data ...
	int				type;			
	int				n_children;
	C45_node_t**	children;	
	int				attribute;
	double			split_threshold;	//For single SPLIT (Continuous)
	double*			disc_values;		//For multiple SPLITS (Discrete)
				
	//Sample Data
	int*			sample_counts;
	double*			sample_weights;	
	int				total_count;
	double			total_weight;		
	
	//For pruning...
	double 			predicted_e;
} ;

//Constructor of C4.5 Decision tree nodes
C45_node_t* c45_node_init(C45_node_t* parent, int n_classes, int own_class, int* sample_counts, double* sample_weights)
{
	C45_node_t* node = (C45_node_t *)malloc(1 * sizeof(C45_node_t));
	
	node->parent = parent;
	
	node->n_classes = n_classes;
	node->own_class = own_class;
	
	//By default assume leaf!
	node->type = C45_NODE_LEAF;
	
	//No children ...
	node->n_children = 0;
	node->children = 0;
	node->disc_values = 0;
	node->split_threshold = 0.0;
	node->attribute = -1;
	
	//Now the samples...
	node->sample_counts = sample_counts;
	node->sample_weights = sample_weights;
	node->total_count = 0;
	node->total_weight = 0.0;
	
	int i;
	for ( i = 0; i < n_classes; i++)
	{
		node->total_weight += sample_weights[i];
		node->total_count += sample_counts[i];
	}	
	
	return node;
} 

int c45_node_is_leaf( C45_node_t* node )
{
	return node->type == C45_NODE_LEAF;
}

void c45_node_init_children( C45_node_t* node, int n_children )
{
	node->n_children = n_children;
	
	//Create children array...
	//Initially all are Null!
	node->children = (C45_node_t **)calloc(n_children, sizeof(C45_node_t *));

}

//Make current node an internal split node
void c45_node_set_split( C45_node_t* node, int attribute, double threshold )
{
	//Set node properties
	node->type = C45_NODE_SPLIT;	
	node->attribute = attribute;
	node->split_threshold = threshold;
	
	//Create children array...
	c45_node_init_children( node, 2);
}

//Make current node a discrete node
void c45_node_set_discrete( C45_node_t* node, int attribute, double *values, int n_values )
{
	//Set node properties
	node->type = C45_NODE_DISCRETE;	
	node->attribute = attribute;	
	
	//Copy values...
	int i;		
	node->disc_values = (double *)calloc(n_values, sizeof(double));	
	//...copy...
	for ( i = 0; i < n_values; i++){
		node->disc_values[i] = values[i];
	}
	//Sort values...
	//...indices required by quicksort, not needed here...
	int* indices = (int *)calloc(n_values, sizeof(int)); 	
	c45_quicksort_values( node->disc_values, indices, 0, n_values - 1 );
	free( indices );
	
	//Create children array...
	c45_node_init_children( node, n_values);
}	

//set a child node...
void c45_node_set_child( C45_node_t* node, int key, C45_node_t* child )
{
	node->children[key] = child;	
}

//Evaluate decision tree selecting the most probable class    
int c45_node_evaluate( C45_node_t* node, double* sample )
{
	C45_node_t* child;
	int i;
	double att_value;
	switch (node->type)
	{
		case C45_NODE_LEAF:		
			return node->own_class;
		case C45_NODE_SPLIT:
			child = node->children[ (sample[ node->attribute ] <= node->split_threshold ? 0 : 1) ];
			return c45_node_evaluate( child, sample );			
		case C45_NODE_DISCRETE:		
			att_value = sample[ node->attribute ];			
			
			//Do small search for closest attribute value in attribute values...
			int closest_idx = 0;
			double closest_dist = fabs( node->disc_values[0] - att_value );
			
			double curr_dist;
			for (i = 1; i < node->n_children; i++){
				curr_dist = fabs(node->disc_values[i] - att_value);
				
				if ( curr_dist < closest_dist ){
					closest_dist = curr_dist;
					closest_idx = i;
				} else {
					//Array is sorted ... if distance is not smaller for current value
					//then it will just become larger for other values here, then just stop
					break;
				}
			}		
			
			child = node->children[ closest_idx ];
			return c45_node_evaluate( child, sample );			
		default:
			return -1;
	}	
}

//evaluate decision tree with probabilities for each class
double* c45_node_weighted_evaluate( C45_node_t* node, double* sample )
{
	C45_node_t* child;
	int i;
	double att_value;
	switch (node->type)
	{
		case C45_NODE_LEAF:		
			return node->sample_weights;
		case C45_NODE_SPLIT:
			child = node->children[ (sample[ node->attribute ] <= node->split_threshold ? 0 : 1) ];
			return c45_node_weighted_evaluate( child, sample );			
		case C45_NODE_DISCRETE:
			att_value = sample[ node->attribute ];			
			
			//Do small search for closest attribute value in attribute values...			
			int closest_idx = 0;
			double closest_dist = fabs( node->disc_values[0] - att_value );
			double curr_dist;
			for (i = 1; i < node->n_children; i++){
				curr_dist = fabs(node->disc_values[i] - att_value);
				if ( curr_dist < closest_dist ){
					closest_dist = curr_dist;
					closest_idx = i;
				} else {
					//Array is sorted ... if distance is not smaller for current value
					//then it will just become larger for other values here, then just stop
					break;
				}
			}			
			
			child = node->children[ closest_idx ];
			return c45_node_weighted_evaluate( child, sample );			
		default:
			return 0;
	}
}

/*
======================================================================================
	FUNCTIONS FOR STRING REPRESENTATION OF THE TREE!
======================================================================================
*/

typedef struct C45_string_buffer {
	char* string;
	int current_n;
	int max_n;
} C45_string_buffer;

C45_string_buffer* c45_node_create_str_buff( int length )
{
	C45_string_buffer* buffer = (C45_string_buffer*)malloc(sizeof(C45_string_buffer));
	
	buffer->string = (char *)malloc( length );
	buffer->current_n = 0;
	buffer->max_n = length;
	buffer->string[0] = '\0';
	
	return buffer;
}

char* c45_node_ints_to_string( int* values, int n )
{	
	if ( n <= 0){
		return "";
	}
	
	//Create buffer
	char* num_buffer = (char *)malloc( 12 );
	char* str_buffer = (char *)malloc( (12 + 2) * n );
	str_buffer[0] = '\0';
	
	int i;
	for (i = 0; i < n; i++ ){
		sprintf( num_buffer, (i == 0 ? "%d" : ", %d"), values[i] );
		strcat( str_buffer, num_buffer);
	}
	
	free( num_buffer );
	
	return str_buffer;
}

//Concatenate two strings
//if resulting string is larger than buffer, new buffer will be created
void c45_node_string_safe_cat( C45_string_buffer* buffer, char *add ){	 
	int add_len = strlen( add );
	
	//check final size
	int new_size = buffer->max_n;
	while (add_len + buffer->current_n >= new_size ){
		//try twice-larger buffer ...
		new_size *= 2;
	}
	
	//if required...
	if (new_size > buffer->max_n ){
		//For debugging!
		//printf( "Will resize from %d to %d \r\n", buffer->max_n, new_size );
		//...Resize the buffer ... 
		char *new_string = (char *)malloc( new_size);
		//... copy data to new buffer...
		strcpy( new_string, buffer->string );		
		//... release old buffer ...
		free( buffer->string );
		//... set new buffer ...	
		buffer->string = new_string;
		//... now larger...
		buffer->max_n = new_size;
	}
	
	//Concatenate...
	strcat( buffer->string + buffer->current_n, add );
	buffer->current_n += add_len;
}

//Generate a string representation of the C4.5 tree
//This function is called recursivelly for every node in the tree
void c45_node_to_string_rec(C45_node_t* node, C45_string_buffer* res_buffer, 
							C45_string_buffer* left_padding, char* extra, int depth )
{
	//printf( " => C_DEPTH %d \n", depth );

	char* counts_text = c45_node_ints_to_string( node->sample_counts, node->n_classes );
	int len_counts = strlen( counts_text );

	int len_extra = strlen( extra );

	int tempo_size = left_padding->current_n + len_extra + len_counts + 100;
	char* tempo_buff = (char *)malloc( tempo_size );
	char* padd =  "  ";
	int len_pad = strlen( padd );
	int i;
	char *child_extra = (char *)malloc( 50 );
	
	if ( node->type == C45_NODE_LEAF ){
		sprintf( tempo_buff, "%s%s CLASS (%d) %s \r\n", left_padding->string, extra, node->own_class, counts_text );				
		
		c45_node_string_safe_cat( res_buffer, tempo_buff );
	} else if (node->type == C45_NODE_DISCRETE ){
		sprintf( tempo_buff, "%s%sAtt (%d), Discrete \r\n", left_padding->string, extra, node->attribute);				
		
		c45_node_string_safe_cat( res_buffer, tempo_buff );
		
		//Save padding...
		int prev_pad = left_padding->current_n;
		//add more padding for children ...
		c45_node_string_safe_cat( left_padding, padd );
		
		for (i = 0; i < node->n_children; i++ )
		{
			sprintf( child_extra, "%f : ", node->disc_values[i] );
			
			c45_node_to_string_rec(node->children[i], res_buffer, left_padding, child_extra, depth + 1 );
		}
		//restore padding
		left_padding->string[ prev_pad ] = '\0';						
		left_padding->current_n -= len_pad;
	} else if (node->type == C45_NODE_SPLIT ){	
		sprintf( tempo_buff, "%s%sAtt (%d), Threshold = %f \r\n", left_padding->string, extra, node->attribute, node->split_threshold );				
		
		c45_node_string_safe_cat( res_buffer, tempo_buff );
		
		//Save padding...
		int prev_pad = left_padding->current_n;
		//add more padding for children ...
		c45_node_string_safe_cat( left_padding, padd );
		
		for (i = 0; i < node->n_children; i++ )
		{
			c45_node_to_string_rec(node->children[i], res_buffer, left_padding, (i == 0 ? "<= " : ">  "), depth + 1 );
		}
		//restore padding
		left_padding->string[ prev_pad ] = '\0';
		left_padding->current_n -= len_pad;
		
		//printf( " => C_DEPTH %d , c_n = %d \n", depth, left_padding->current_n );
	}
	
	free( child_extra );
	free( counts_text );
	free( tempo_buff );	
}

//Get string representation of a C4.5 decision tree
//This function start the recursion
char* c45_node_to_string(C45_node_t* node)
{	
	C45_string_buffer* res_buffer = c45_node_create_str_buff( 1024 );
	C45_string_buffer* pad_buffer = c45_node_create_str_buff( 1024 );
	
	c45_node_to_string_rec( node, res_buffer, pad_buffer, "", 1 );
	
	char* final_return = res_buffer->string;
	
	free( res_buffer );
	free( pad_buffer->string );
	free( pad_buffer );

	//Debugging...
	//printf( "before %p ... ", final_return );
	
	return final_return;
}

//release strings
void c45_node_release_string( char* pointer){
	//Debugging
	//printf( "later %p \n",pointer );
	
	free( pointer ) ;
}

//Release entire hierarchy of C4.5 Decision tree
//Starting from given node
void c45_node_release( C45_node_t* node, int release_sample_info )
{
	//...Recursivelly free children...
	int i;
	for (i = 0; i < node->n_children; i++){
		c45_node_release( node->children[i], release_sample_info );
	}	
	
	//...Inner arrays...
	if ( node->n_children > 0 ){
		free( node->children );
	}
	if ( node->type == C45_NODE_DISCRETE ){
		free( node->disc_values );
	}
	
	if (release_sample_info == 1){
		free( node->sample_counts );
		free( node->sample_weights );
	}
	
	//...Finally, free the node itself!....
	free( node );
}
	
/*
======================================================================================
	FUNCTIONS FOR WEIGHTED TRAINING OF A TREE!
======================================================================================
*/
typedef struct C45_count_info {
	double* weighted_count;
	int* absolute_count;
	int majority_class;
	int n_classes;
	int n_samples;
} C45_count_info;

typedef struct C45_threshold_info {
	double value;
	int* absolute_count;
	double* weighted_count;
} C45_threshold_info;

typedef struct C45_split_info {
	double threshold;	
	double ratio;
	int count_two;
	int max_split_size;
} C45_split_info;

typedef struct C45_info {
	double info;
	double total_weight;
} C45_info;

typedef struct C45_double_array {
	double* data;
	int size;
} C45_double_array;

typedef struct C45_split_buffer {
	C45_threshold_info* values;
	C45_sortable_pair* sorted_pairs; 
	double*	left_weights;
	double* right_weights;
	int* left_abs;
	int* right_abs;
	int max_size;
	int current_size;
	int n_classes;
} C45_split_buffer;


C45_split_buffer* c45_split_buffer_create(int max_size, int n_classes){
	//First, create the structure...
	C45_split_buffer* buffer = (C45_split_buffer*)calloc(1, sizeof(C45_split_buffer));
	
	//Now, create the empty array to hold values...
	buffer->values = (C45_threshold_info*)calloc(max_size, sizeof(C45_threshold_info));
	
	//For the array of sorted elements...
	buffer->sorted_pairs = (C45_sortable_pair*)calloc(max_size, sizeof(C45_sortable_pair));
	
	buffer->left_weights = (double *)calloc(n_classes, sizeof(double));
	buffer->right_weights = (double *)calloc(n_classes, sizeof(double));
	buffer->left_abs = (int *)calloc(n_classes, sizeof(int));
	buffer->right_abs = (int *)calloc(n_classes, sizeof(int));
	
	//Set initial values...
	buffer->max_size = max_size;
	buffer->current_size = 0;
	buffer->n_classes = n_classes;
	
	return buffer;
}

void c45_split_buffer_prepare(C45_split_buffer* buffer, int new_size){	
	int i;
	
	//Set current overlapping values to 0
	int limit = (new_size < buffer->current_size ? new_size : buffer->current_size);
	for ( i = 0; i < limit; i++){
		memset(buffer->values[i].absolute_count, 0, buffer->n_classes * sizeof(int));
		memset(buffer->values[i].weighted_count, 0, buffer->n_classes * sizeof(double));
	}
	
	//Check if additional arrays will have to be created...
	if (new_size > buffer->current_size){
		//Allocate additional arrays 
		for ( i = buffer->current_size; i < new_size; i++){
			buffer->values[i].absolute_count = (int *)calloc( buffer->n_classes, sizeof(int) );
			buffer->values[i].weighted_count = (double *)calloc( buffer->n_classes, sizeof(double) );																	
		}
		
		//Set new current size...
		buffer->current_size = new_size;
	}	
	
	//Finally, set the arrays of distributions with 0's
	memset(buffer->left_weights, 0, buffer->n_classes * sizeof(double));
	memset(buffer->right_weights, 0, buffer->n_classes * sizeof(double));
	memset(buffer->left_abs, 0, buffer->n_classes * sizeof(int));
	memset(buffer->right_abs, 0, buffer->n_classes * sizeof(int));
}

void c45_split_buffer_destroy(C45_split_buffer* buffer){	
	//Release the values...
	int i;
	for (i = 0; i < buffer->current_size; i++){
		//First, the inner arrays for each allocated value..
		free( buffer->values[i].absolute_count );
		free( buffer->values[i].weighted_count );
	}
	//then, the whole array of values...
	free(buffer->values);
	
	//the whole array of sorted pairs...
	free(buffer->sorted_pairs);
	
	//the array of the distributions...
	free(buffer->left_weights);
	free(buffer->right_weights);
	free(buffer->left_abs);
	free(buffer->right_abs);
	
	//Finally, the final pointer to whole structure
	free( buffer );
}

//Count how many elements are per class in the given dataset,
//and also computes the majority class
C45_count_info c45_tree_counts_by_class(int* labels, double* distribution, int n_samples, int n_classes){
	C45_count_info	info;
	
	info.weighted_count = (double *)calloc(n_classes, sizeof(double));
	info.absolute_count = (int *)calloc(n_classes, sizeof(int));
	info.majority_class = 0;
	info.n_samples = n_samples;
	info.n_classes = n_classes;
	
	//Count labels per class
	int i, curr_label;
	for (i = 0; i < n_samples; i++){
		curr_label = labels[i];
		
		if (curr_label < 0 || curr_label >= n_classes){
			printf("Invalid label found: %d \n", curr_label);
			system("pause");
		}
		
		info.weighted_count[ curr_label ] += distribution[i];
		info.absolute_count[ curr_label ] ++;
	}
	
	//Get majority
	for (i = 1; i < n_classes; i++){
		if ( info.weighted_count[ info.majority_class ] < info.weighted_count[ i ] ){
			info.majority_class = i;
		}		     
	}
		
	return info;
}


C45_info c45_calculate_info( double* weighted_count, int n_classes ){
	C45_info info;
	
	info.info = 0.0;
	info.total_weight = 0.0;
	
	//get total weight in the current distribution...
	int k;
	for ( k = 0; k < n_classes; k++){
		info.total_weight += weighted_count[k];
	}
	
	//Now calculate information gain
	double prior_k;
	for ( k = 0; k < n_classes; k++ ){
		//Prior is weighted...
		prior_k = weighted_count[k] / info.total_weight;
		
		//Check if prior of class k is above 0
		//0 prior probability means no samples of this class
		//and therefore it should just be left out of current computation
		if (prior_k > 0.0){
			info.info += prior_k * log2(prior_k);	
		}		
	}
	
	info.info *= -1; //Change sign...
	
	return info;
}

//Find the Best Split for a continuous attribute
C45_split_info c45_find_best_split( double** p_samples, int* labels, double* distribution, int attribute, int n_attributes, 
									C45_count_info c_info, C45_split_buffer* buffer ){
	C45_split_info best_split;
	
	//now, for every value, sort them  
    //and create a local count...
	
	double att_value, weight;
	int found, label, i, k;
		
	C45_sortable_pair* sorted_pairs = buffer->sorted_pairs;		
	C45_sortable_pair* tempo_pos = sorted_pairs;	
	
	//Copy data...
	for (i = 0; i < c_info.n_samples; i++){		
		sorted_pairs[i].index = i;
		sorted_pairs[i].value = *(p_samples[i] + attribute);
	}	
	
	//...Sort them!...	
	qsort(sorted_pairs, c_info.n_samples, sizeof(C45_sortable_pair), c45_compare);
	
	//count different values...
	int count_different = 1;
	for (i = 1; i < c_info.n_samples; i++){	
		if (sorted_pairs[i].value != sorted_pairs[i - 1].value){
			//A change of value...
			count_different++;
		}
	}

	//...count samples at each different value....
	//...prepare the buffer that will hold the information...	
	c45_split_buffer_prepare( buffer, count_different );
	
	C45_threshold_info* values = buffer->values;
	C45_threshold_info* p_values = values; 
		
	//...Start with a value lower than minimum...	
	//double prev_value = sorted_pairs[0].value - 1.0;
	double prev_value = 0.0;
	tempo_pos = sorted_pairs;
	
	//...for each SORTED VALUE ....
	int current_index;
	for (i = 0; i < c_info.n_samples; i++){
				
		if (i == 0 || prev_value != sorted_pairs[i].value){
			//A new different value has been found			
			if (i > 0){
				p_values++;
			}
			p_values->value = sorted_pairs[i].value;					
			
			//prev_value = *tempo_pos;			
			prev_value = tempo_pos->value;
		}
		
		current_index = tempo_pos->index;				
		
		//Add sample to current position...
		//...get label for original sample of current value..
		label = *(labels + current_index);
		weight = *(distribution + current_index);		
		
		p_values->absolute_count[ label ] ++;
		p_values->weighted_count[ label ] += weight;				
		
		tempo_pos++;		
	}
	
	//printf( "List completed ... found %d different values \n", count_different );
	
	if ( count_different == 1){
		//There is only one single value for all samples
		//No information gain...
		best_split.threshold = 0.0;
		best_split.ratio = 0.0;
		best_split.count_two = 0;							
		best_split.max_split_size = 0;
	} else {
			
		//Get global info....
		C45_info parent_info = c45_calculate_info( c_info.weighted_count, c_info.n_classes );

		//Now, evaluate the gain ratio at every point...		
		double*	left_weights = buffer->left_weights;
		double* right_weights = buffer->right_weights;
		int* left_abs = buffer->left_abs;
		int* right_abs = buffer->right_abs;
		
		//Copy values for the right side ...
		for ( i = 0; i < c_info.n_classes; i++ ){
			right_weights[i] = c_info.weighted_count[i];
			right_abs[i] = c_info.absolute_count[i];		
		}
		
		found = 0;
		//.... for correction of selected thresholds...    
		double min_weight = parent_info.total_weight * 0.05;
		for ( i = 0; i < count_different - 1; i++ ){
			//Test possible split at current value
			
			//Use local distribution ...
			C45_threshold_info local_dist = values[i];					
			
			// ... add to the left
			// ... subtract from the right...			
			for ( k = 0; k < c_info.n_classes; k++){				
			
				//On the left....
				left_weights[k] += local_dist.weighted_count[k];
				left_abs[k] += local_dist.absolute_count[k];						
				
				//On the right...
				right_weights[k] -= local_dist.weighted_count[k];
				right_abs[k] -= local_dist.absolute_count[k];
				
				if ( right_weights[k] < 0.0 ){
					right_weights[k] = 0.0;
				}
			}
			
			//evaluate split...
			//...entropy on each side...
			C45_info left_info = c45_calculate_info( left_weights, c_info.n_classes );
			C45_info right_info = c45_calculate_info( right_weights, c_info.n_classes );
			
			double left_prior = left_info.total_weight / parent_info.total_weight;
			double right_prior = right_info.total_weight / parent_info.total_weight;         
			
			//now calculate the new weighted entropy...
			//info(x)
			double info_x = left_prior * left_info.info + right_prior * right_info.info;
			//gain(x)            
			double current_gain = parent_info.info - info_x;
			//split_info(X)
			double split_info_x;
			if ( left_prior > 0.0 && right_prior > 0.0 ){
				split_info_x = -( left_prior * log2(left_prior) + right_prior * log2(right_prior) );
			} else {
				//Skip current split...				
				continue;
			}
			
			//gain_ratio(x)
			double gain_ratio = current_gain / split_info_x;
			
			//do the test of two...
			double total_left = 0;
			double total_right = 0;
			for ( k = 0; k < c_info.n_classes; k++ ){
				total_left += left_abs[k];
				total_right += right_abs[k];
			}
			
			int count_two = (total_left < 2 ? 1 : 0) + (total_right < 2 ? 1 : 0);						
			
			//only consider splits that will leave at least 
			//a given % of data on each side		
			if (left_info.total_weight > min_weight && 
				right_info.total_weight > min_weight){
				//Valid split... check if best split
				if ( found == 0 || gain_ratio > best_split.ratio ){					
					//use midpoint for threshold (as defined in Chapter. 2 of Quinlan's Book)
					double threshold = (values[i].value + values[i + 1].value) / 2.0;
					
					//only if it has no floating point issues....
					//(mid point has to be strictly between the two values,
					// it cannot be equal to any of them...)
					if (values[i].value < threshold && threshold < values[i + 1].value){				
						//New best split found
						found = 1;
						best_split.ratio = gain_ratio;
						best_split.count_two = count_two;
						best_split.max_split_size = (total_left > total_right ? total_left : total_right);						
						
						best_split.threshold = threshold;
					} else {
						printf(" a rounding error here.... \n");
						system("pause");
					}
				}                      
			}		
		}
		
		if ( found == 0){
			//no good split found that did not cut less than 5% on one of its sides
			//No information gain...
			best_split.threshold = 0.0;
			best_split.ratio = 0.0;
			best_split.count_two = 0;						
		}			
	}
	
	return best_split;
}

C45_double_array c45_find_unique_values( double** p_samples, int attribute, int n_attributes, int n_samples ){
	int i;

	double* sorted_values = (double *)calloc(n_samples, sizeof(double));	
	int* indices = (int *)calloc(n_samples, sizeof(int));
	
	//.... Copy values and sort them ...
	double* tempo_pos = sorted_values;
	int* idx_pos = indices;
	for (i = 0; i < n_samples; i++){
		//*tempo_pos = *(samples + (i * n_attributes + attribute));
		*tempo_pos = *(p_samples[i] + attribute);
		tempo_pos ++;
		
		*idx_pos = i;
		idx_pos ++;
	}		
	int count_different = c45_quicksort_values( sorted_values, indices, 0, n_samples - 1);
	
	C45_double_array all_unique;
	
	all_unique.size = count_different;
	all_unique.data = (double *)calloc(count_different, sizeof(double));
	
	double* p_values = all_unique.data;
	
	double prev_value = *sorted_values - 1.0; 
	tempo_pos = sorted_values;
	for ( i = 0; i < n_samples; i++){
		if (prev_value != *tempo_pos){
			//A new unique value has been found						
			*p_values = *tempo_pos;						
			
			p_values++;
			
			prev_value = *tempo_pos;			
		}
		
		tempo_pos++;
	}
	
	//Release all memory except array of unique values...
	free( sorted_values );
	free( indices );
	
	return all_unique;
}

C45_split_info c45_discrete_gain_ratio( double** p_samples, int* labels, double* distribution, int attribute, int n_attributes, C45_count_info c_info ){
	C45_split_info discrete_split;
	
	int i, k, label;
	double weight;
	
	//Calculate the different values present in data...	(similar to case for thresholds)
	double* sorted_values = (double *)calloc(c_info.n_samples, sizeof(double));	
	int* indices = (int *)calloc(c_info.n_samples, sizeof(int));
	
	//.... Copy values and sort them ...
	double* tempo_pos = sorted_values;
	int* idx_pos = indices;
	for (i = 0; i < c_info.n_samples; i++){
		//*tempo_pos = *(samples + (i * n_attributes + attribute));
		*tempo_pos = *(p_samples[i] + attribute);
		tempo_pos ++;
		
		*idx_pos = i;
		idx_pos ++;
	}		
	
	int count_different = c45_quicksort_values( sorted_values, indices, 0, c_info.n_samples - 1);
	
	//...count samples at each different value....
	C45_threshold_info* values = (C45_threshold_info*)calloc(count_different, sizeof(C45_threshold_info));
	//Start at position -1, invalid, however will become 0 before first access
	C45_threshold_info* p_values = values - 1; 
	
	//...Start with a value lower than minimum...
	double prev_value = *sorted_values - 1.0; 
	tempo_pos = sorted_values;
	//...for each SORTED VALUE ....
	for (i = 0; i < c_info.n_samples; i++){
		
		if (prev_value != *tempo_pos){
			//A new different value has been found			
			p_values++;
			p_values->value = *tempo_pos;
			p_values->absolute_count = (int *)calloc( c_info.n_classes, sizeof(int) );
			p_values->weighted_count = (double *)calloc( c_info.n_classes, sizeof(double) );					
			
			prev_value = *tempo_pos;			
		}
		
		//Add sample to current position...
		//...get label for original sample of current value..
		label = *(labels + indices[i] );
		weight = *(distribution + indices[i]);
		
		p_values->absolute_count[ label ] ++;
		p_values->weighted_count[ label ] += weight;		
		
		tempo_pos++;		
	}
	
	if ( count_different == 1 ){	
		discrete_split.ratio = 0.0;
		discrete_split.count_two = 0;		
	} else {	
		//calculate how many elements will be on each partition...
		
		//Get global info....
		C45_info parent_info = c45_calculate_info( c_info.weighted_count, c_info.n_classes );
		
		//Now, evaluate gain ratio....
		
		//...do it for every subdivision...	
		double info_x = 0.0;
		double split_info_x = 0.0;
		int max_split_size = 0;
		p_values = values;
		for ( k = 0; k < count_different; k++){
			int absolute_total = 0;
			for (i = 0; i < c_info.n_classes; i++){
				absolute_total += p_values->absolute_count[i];
			}
			
			if ( absolute_total < 2 ){
				discrete_split.count_two ++; 
			}
			if (absolute_total > max_split_size){
				max_split_size = absolute_total;
			}
			
			C45_info k_info = c45_calculate_info( p_values->weighted_count, c_info.n_classes );
			
			double k_prior = k_info.total_weight / parent_info.total_weight;
			
			//add to the weighted entropy... (info x)
			//info(x)
			info_x += k_prior * k_info.info;
			
			//split_info(X)
			split_info_x += k_prior * log2( k_prior );
						
			p_values++;
		}
		
		discrete_split.max_split_size = max_split_size;
	
		//now, evaluate the gain_ratio ...	
		//gain(x)            
		double current_gain = parent_info.info - info_x;
		
		split_info_x = -split_info_x;
	
		//gain_ratio(x)
		discrete_split.ratio = current_gain / split_info_x;			
	}
	
	//printf( " HERE THE RATIO WOLD BE %f \n",  discrete_split.ratio );

	//Release all memory allocated by this function...	
	free( sorted_values );
	free( indices );
	
	//The array used for testing thresholds...
	p_values = values;
	for ( i = 0; i < count_different; i++){
		free( p_values->absolute_count );
		free( p_values->weighted_count );
		p_values++;
	}
	free( values );				
	
	return discrete_split;
}

int c45_bin_search_value( C45_double_array array, double value){
	//Receives a sorted array of values and finds the index 
	//of the given value if exists
	int init, end, m;
	
	init = 0;
	end = array.size - 1;
	
	while (init <= end ){		
		m = (init + end ) / 2;
		
		if ( array.data[m] == value ){
			return m;
		} else {
			if ( array.data[m] < value ){
				init = m + 1;
			} else {
				end = m - 1;
			}
		}
	}
	
	return -1;
}


C45_node_t* c45_tree_construct_rec( double** p_samples, int* labels, int* att_types, double* distribution, 
								int n_samples, int n_attributes, int n_classes, int parent_class, 
								C45_node_t* parent, int max_splits, int c_depth, C45_split_buffer* split_buffer) 
{
	int i;
	C45_node_t* new_node;

	//printf("Starting tree-node construction! P = %p, DEPTH = %d \n", parent, c_depth);
	
	//For sample array:
	// - Each row represents a sample
	// - Each column represents an attribute
	
	//There are some possibilities...
	if (n_samples == 0){
    	//printf("Base case without samples! P = %p \n", parent);

		//A base case with no samples remaining? (empty-tree ... maybe?)
		int* tempo_counts = (int *)calloc(n_classes, sizeof(int) );
		double* tempo_weights = (double *)calloc(n_classes, sizeof(double));
		
		new_node = c45_node_init(parent, n_classes, parent_class, tempo_counts, tempo_weights);
	} else {
		//There are samples .... do c4.5
		
		//Get counts and majority class
		C45_count_info c_info = c45_tree_counts_by_class(labels, distribution, n_samples, n_classes);
        if (c_info.absolute_count[ c_info.majority_class ] == n_samples){

			//base case, all samples are of a single class
			new_node = c45_node_init(parent, n_classes, c_info.majority_class, c_info.absolute_count, c_info.weighted_count );
		}else{		    

			//Calculate best attribute for a split!
			//... Check available attributes ...
			int* att_available = (int *)malloc( sizeof(int) * n_attributes );
			int not_available = 0;
			//... assume all are available...
			for (i = 0; i < n_attributes; i++){			
				att_available[i]  = max_splits;
			}
			//...remove the ones already used on the path to the root!
			C45_node_t* current_node = parent;			
			while( current_node != 0 ){
				//Depending on the case ...
				if ( att_types[ current_node->attribute ] == C45_NODE_SPLIT ){
					//Reduce the number of available splits over this attribute
					att_available[ current_node->attribute ] --;
				} else {
					//Discrete can always be used only once
					att_available[ current_node->attribute ] = 0;
				}			

				//Count not available
				if ( att_available[ current_node->attribute ] == 0){
					not_available++;
				}
				
				current_node = current_node->parent;
			}
											
			//If all attributes have been used to the maximum allowed ...
			if ( not_available == n_attributes ){
			    //printf("No more available attributes! P = %p \n", parent);
				
				//Create a leaf...
				new_node = c45_node_init(parent, n_classes, c_info.majority_class, c_info.absolute_count, c_info.weighted_count );
			} else {			    
				//printf("Selecting the best split! P = %p \n", parent);
				
				//There are attributes to choose from...
				//for each attribute available, compute gain!
				int best_attribute = -1;
				
				C45_split_info best_split;
				C45_split_info split_info;
				int first = 1;
				int best_type = 0;							
				
				for ( i = 0; i < n_attributes; i++ ){
					if ( att_available[ i ] > 0 ){
						if ( att_types[ i ] == C45_NODE_SPLIT ){
							//Split over continuous attribute ...
							//printf("Will test split on att %d! \n", i);

							//find best split...
							split_info = c45_find_best_split( p_samples, labels, distribution, i, n_attributes, c_info, split_buffer);
							
						} else {
						    //printf("Will check discrete att %d! \n", i);

							//Discrete
							split_info = c45_discrete_gain_ratio( p_samples, labels, distribution, i, n_attributes, c_info );
						}

						//printf( "Current Split ratio %f \n", split_info.ratio );
						
						if ((first == 1 || best_split.ratio < split_info.ratio ) && split_info.count_two < 2){					
							//new best attribute...						
							best_attribute = i;
							best_type = att_types[ i ];
							best_split = split_info;
							
							first = 0;	//Next won't be the first one
						} 					
					}
				}

				//printf("Creating new node! P = %p \n", parent);

                //Create Node....
				new_node = c45_node_init(parent, n_classes, c_info.majority_class, c_info.absolute_count, c_info.weighted_count);	
			
				//Check split...
				if ( first == 0 && best_split.ratio > 0.0 ){
					//Split accepted ....
					//generate the partitions...
					//...check number of children...
					//printf("Will apply split, preparing... \n");

					int n_children;
					C45_double_array disc_values;
					if ( best_type == C45_NODE_SPLIT){
						//Split over continous (only 2)
						n_children = 2;
					} else {
						//Paritions for Discrete (unknown, must be found ... again!)
						disc_values = c45_find_unique_values( p_samples, best_attribute, n_attributes, c_info.n_samples );
						n_children = disc_values.size;
					}
					
					//...create the arrays...
					int* children_sizes = (int *)calloc( n_children, sizeof(int) );
					double*** children_samples = (double ***)calloc( n_children, sizeof(double **));				
					int** children_labels = (int **)calloc( n_children, sizeof(int));
					double** children_distributions = (double**)calloc( n_children, sizeof(double *));
					//...generate the empty sub-arrays...
					for ( i = 0; i < n_children; i++ ){
						children_samples[i] = (double **)calloc( best_split.max_split_size, sizeof(double *) );
						children_labels[i] = (int *)calloc( best_split.max_split_size, sizeof(int) );
						children_distributions[i] = (double *)calloc( best_split.max_split_size, sizeof(double) );						
					}
					
					//printf("splitting data... \n");
					//... split samples per child....
					int label, to_child, curr_size;
					double* values;
					for ( i = 0; i < n_samples; i++){
						label = labels[i];
						values = p_samples[i];
						
						//check to which child will go the sample...
						if ( best_type == C45_NODE_SPLIT ){
							//For continuous
							to_child = ( values[best_attribute] <= best_split.threshold ? 0 : 1);													
						} else {
							//For discrete ...
							to_child = c45_bin_search_value( disc_values, values[best_attribute] );
						}
						
						//...append...
						curr_size = children_sizes[to_child];
						children_samples[to_child][curr_size] = values;
						children_labels[to_child][curr_size] = label;
						children_distributions[to_child][curr_size] = distribution[i];
						//...expanded...
						children_sizes[to_child]++;									
					}
					
					//printf("split done! setting the parent type... \n");

					//Set split...
					if (best_type == C45_NODE_SPLIT ){
						//For continuous
						c45_node_set_split( new_node, best_attribute, best_split.threshold );
					} else {
						//For discrete
						c45_node_set_discrete( new_node, best_attribute, disc_values.data, disc_values.size );				
					}
										
					//Add children
					C45_node_t* child_node;
					for ( i = 0; i < n_children; i++){
						child_node = c45_tree_construct_rec( children_samples[i], children_labels[i], att_types, children_distributions[i], 
									children_sizes[i], n_attributes, n_classes, c_info.majority_class, 
									new_node, max_splits, c_depth + 1, split_buffer );
						
						c45_node_set_child( new_node, i, child_node );					
					}
										
					if ( best_type == C45_NODE_DISCRETE){
						//Relase array of unique values...
						free( disc_values.data );
					}
										
					//printf("Releasing memory... P = %p \n", parent);
					
					//Release memory of samples from children ...
					//... the sub arrays....
					for ( i = 0; i < n_children; i++ ){
						free( children_samples[i] );
						//printf("Released children samples [%d] P = %p \n", i, parent);
						free( children_labels[i] );
						//printf("Released children labels [%d] P = %p \n", i, parent);												
						free( children_distributions[i] );
                        //printf("Released children distributions [%d] P = %p \n", i, parent);
					}
					//printf("..children samples, labels distributions freed P = %p \n", parent);
					//... the main arrays....
					free( children_sizes );
					//printf("..children sizes P = %p \n", parent);
					free( children_samples );
					//printf("..children samples P = %p \n", parent);
					free( children_labels );
					//printf("..children labels P = %p \n", parent);
					free( children_distributions );
					//printf("..children distributions P = %p, D = %d \n", parent, c_depth);
				}
			}

			//Release other memory taken...
			free( att_available );
			//printf("..Released att_avaiable P = %p, D = %d \n", parent, c_depth);
		}						
	}

	//printf("...node built succesfully! P = %p, D = %d \n", parent, c_depth);
	
	// Return the final node created...
	return new_node;
}

//Function to call externally for tree construction...
C45_node_t* c45_tree_construct( double* samples, int* labels, int* att_types, double* distribution, 
								int n_samples, int n_attributes, int n_classes, int parent_class, 
								int max_splits ) 								
{
    //printf("Tree training shall start! \n");
    //First, create an array of pointers for all samples...
	double ** p_samples = (double **)calloc(n_samples, sizeof(double *));

	int i;
	for ( i = 0; i < n_samples; i++ ){
		p_samples[i] = samples + (n_attributes * i);
	}
	
	//Create buffer for splits...
	C45_split_buffer* split_buffer = c45_split_buffer_create(n_samples, n_classes);

	//printf("starting rec...! \n");
	C45_node_t* root = c45_tree_construct_rec( p_samples, labels, att_types, distribution,
												n_samples, n_attributes, n_classes, parent_class,
												0, max_splits, 1, split_buffer);
												
	//Destroy buffer for splits...
	c45_split_buffer_destroy(split_buffer);
	
	//printf("Finishing! \n");
	//Release at the end...
	free( p_samples);
	
	return root;
}


//========================================================================================
//    Post processing Routines
//		
//		- Pruning
//========================================================================================

/*
	Adaptation from these pages:
		Original JavaScript:
			http://statpages.org/confint.html
			
		python version:
			http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
*/

typedef struct C45_CI_limits {
	double dl;
	double ul;	
} C45_CI_limits;

typedef struct C45_node_error {
	int absolute;
	double weighted;
} C45_node_error;

C45_node_error c45_node_error( C45_node_t* node ){
	C45_node_error local_error;
	
	local_error.absolute = 0;
	local_error.weighted = 0.0;
	
	int i;
	for ( i = 0; i < node->n_classes; i++ ){
		if ( i != node->own_class ){
			local_error.absolute += node->sample_counts[i];
			local_error.weighted += node->sample_weights[i];
		}
	}
	
	return local_error;
}

double c45_pruning_bin_p( double N, double p, double x1, double x2 ){
	double q = p / (1 - p);
	double k = 0.0;
    double v = 1.0;
    double s = 0.0;
	double tot = 0.0;
	double l_val = 1.0e30;
	
	while ( k <= N ){
		tot += v;
		
		if (x1 <= k && k <= x2){
            s += v;
		}
		
		if ( tot > l_val ){
			s = s / l_val;
			tot = tot / l_val;
			v = v / l_val;
		}
		
		k += 1.0;
		v = v * q * ( N + 1 - k)/ k;		
	}
    
	return s / tot;
}

C45_CI_limits c45_pruning_calc_bin(int vx, int vN, double vCL, double vTU)
{
	//Set the confidence bounds
	//double vTU = (100.0 - vCL) / 2.0;
    double vTL = 100.0 - vCL - vTU;
	
	double vP = (double)vx / (double)vN;
	
	double dl, ul, v, p, vsH, vsL;
	
	if( vx == 0 ) {
        dl = 0.0;
	} else {
		v = vP / 2.0;
		vsL = 0.0;
		vsH = vP;
		p = vTL / 100.0;
		
		while ((vsH-vsL) > 1.0e-5){
			if( c45_pruning_bin_p(vN, v, vx, vN) > p){
				vsH = v;
				v = (vsL + v) / 2.0;
			}else{
				vsL = v;
				v = (v + vsH) / 2.0;
			}
		}				
		
		dl = v;
	}
	      
    if ( vx == vN ){
		ul = 1.0;
    } else{
		v = (1.0 + vP) / 2.0;
		vsL = vP;
		vsH = 1.0;
		p = vTU / 100.0;
								
		while( (vsH-vsL) > 1e-5 ){
			if( c45_pruning_bin_p(vN, v, 0, vx) < p){
				vsH = v;
				v = (vsL + v) / 2.0;
			}else{
				vsL = v;
				v = (v + vsH) / 2.0;
			}
		}
		
		ul = v;
    }

	C45_CI_limits result;
	
	result.dl = dl;
	result.ul = ul;
	
	return result;
}

double c45_pruning_U( int E, int N, double CF){
	double n_CF = (1.0 - CF) * 100.0;
	C45_CI_limits result = c45_pruning_calc_bin(N - E, N, n_CF, 0.0);
	
	return 1.0 - result.dl;
}

int c45_prune_tree( C45_node_t* node, int weighted, double CF )
{		
	int i, N, was_pruned;	
	
	//printf( " Enter Node (%d) ... Compute error ... \n", node->total_count );
	
	//...compute leaf predicted error...
	C45_node_error local_error = c45_node_error( node );
	if ( weighted == 0 ){
		N = node->total_count;
		node->predicted_e = node->total_count * c45_pruning_U( local_error.absolute, node->total_count, CF );		
	} else {
		//Determine total of elements on the tree...
		C45_node_t* tempo = node;
		while (tempo->parent > 0){
			tempo = tempo->parent;
		}
		
		N = (int)round( node->total_weight * tempo->total_count );
		int E = (int)ceil( local_error.weighted * tempo->total_count );
		
		if ( E < N ){
			node->predicted_e = N * c45_pruning_U( E, N, CF );
		} else {
			node->predicted_e = E;
		}
	} 
	
	//printf( " Node Check (%d) for children... Compute error ... \n" , node->total_count );
	
	if ( node->n_children == 0 ){
		//No children = leaf = no pruning
		was_pruned = 0;
	} else { 
		//Do recursively, call for children ...
		for ( i = 0; i < node->n_children; i++ ){
			was_pruned += c45_prune_tree( node->children[ i ], weighted, CF );			
		}		
		
		//Check total error as sub-tree...
		double children_e = 0.0;
		for ( i = 0 ; i < node->n_children; i++ ){
			children_e += node->children[i]->predicted_e;
		}
		
		if ( children_e > N ){
			children_e = N;
		}
		
		if ( node->predicted_e < children_e ) {
			//Error would be reduced... prune!
			//printf( "Node prunned! (%d) \n", node->total_count );
			
			//Release children...
			for (i = 0; i < node->n_children; i++){
				//Release the child...
				c45_node_release( node->children[i], 1 );
			}
			free( node->children );
			node->children = 0;
			node->n_children = 0;
			
			//type dependent...
			if ( node->type == C45_NODE_DISCRETE ){
				free( node->disc_values );
				node->disc_values = 0;
			}
			node->split_threshold = 0.0;			
			
			//Make a leaf again...
			node->type = C45_NODE_LEAF;
			node->attribute = -1;
								
			was_pruned ++;			
		} else {
			//The error of the children is smaller, no pruning			
			//The sub-tree error is equal to error of branches
			node->predicted_e = children_e;
		}
						
	}
	
	//printf( " Node Finished!! (%d) \n" , node->total_count );

	return was_pruned;
}

//========================================================================================
//    File Management Routines
//		
//		- Save into a file
//		- Load from a file
//========================================================================================

//Recursively adds the given node and its children to the given file...
int c45_append_node_to_file(C45_node_t* node, FILE* out_file  ) {
	//Check if root
	if ( node->parent == 0 ){
		//Certain info is only stored at root level...
		fwrite( &node->n_classes, sizeof(int), 1, out_file );
	}
	
	int i;
	
	//Own data...
	fwrite( &node->own_class, sizeof(int), 1, out_file );
	fwrite( &node->type, sizeof(int), 1, out_file );
	fwrite( &node->attribute, sizeof(int), 1, out_file );
	fwrite( &node->n_children, sizeof(int), 1, out_file );
	
	//Data that depends on the type...
	if ( node->type == C45_NODE_DISCRETE){
		fwrite( node->disc_values, sizeof(double), node->n_children, out_file );
	} else {
		fwrite( &node->split_threshold, sizeof(double), 1, out_file );
	}
	
	//Distribution data...
	fwrite( node->sample_counts, sizeof(int), node->n_classes, out_file );
	fwrite( node->sample_weights, sizeof(double), node->n_classes, out_file );
	fwrite( &node->total_count, sizeof(int), 1, out_file );
	fwrite( &node->total_weight, sizeof(double), 1, out_file );

	//Now, for children (if any)		
	for ( i = 0; i < node->n_children; i++){
		c45_append_node_to_file(node->children[i], out_file );
	}	
}

int c45_save_to_file( C45_node_t* node, char* file_name ){
	FILE* out_file = fopen( file_name , "wb" );
	if ( out_file == 0 ){
		printf( "Could not write to %s \n", file_name );
		return 0;
	}
	
	c45_append_node_to_file(node, out_file  );
	
	fclose( out_file );
	
	return 1;
}

//Recursively read a node and its children from a file
C45_node_t* c45_load_node_from_file(FILE* in_file, int root, int n_classes ) {
	//Create node...
	C45_node_t* node = (C45_node_t*)calloc( sizeof(C45_node_t), 1);
	
	if ( root == 1){
		//Load values special for the root...
		fread( &node->n_classes, sizeof(int), 1, in_file );						
	} else {
		//Use values for internal nodes and leaves...
		node->n_classes = n_classes;
	}
	
	int i;
	
	//Own data...
	fread( &node->own_class, sizeof(int), 1, in_file );	
	fread( &node->type, sizeof(int), 1, in_file );
	fread( &node->attribute, sizeof(int), 1, in_file );
	fread( &node->n_children, sizeof(int), 1, in_file );
	
	//Data that depends on the type...
	if ( node->type == C45_NODE_DISCRETE){
		node->disc_values = (double *)calloc( sizeof(double), node->n_children );
		fread( node->disc_values, sizeof(double), node->n_children, in_file );
	} else {
		fread( &node->split_threshold, sizeof(double), 1, in_file );
	}
	
	//Distribution data...
	node->sample_counts = (int *)calloc( sizeof(int), node->n_classes );
	node->sample_weights = (double *)calloc( sizeof(double), node->n_classes );
	
	fread( node->sample_counts, sizeof(int), node->n_classes, in_file );
	fread( node->sample_weights, sizeof(double), node->n_classes, in_file );	
	fread( &node->total_count, sizeof(int), 1, in_file );
	fread( &node->total_weight, sizeof(double), 1, in_file );
	
	if ( node->n_children > 0 ){
		node->children = (C45_node_t** )calloc( sizeof(C45_node_t*), node->n_children );
		for ( i = 0; i < node->n_children; i++ ){
			node->children[i] = c45_load_node_from_file(in_file, 0, node->n_classes );
			node->children[i]->parent = node;
		}
	}
	
	return node;
}

C45_node_t* c45_load_from_file( char* file_name ){
	FILE* in_file = fopen( file_name , "rb" );
	if ( in_file == 0 ){
		printf( "Could not read file %s \n", file_name );
		return 0;
	}
	
	C45_node_t* root = c45_load_node_from_file(in_file, 1, 0 );
	
	fclose( in_file );
	
	return root;
}

//========================================================================================
//    Tree Analysis
//========================================================================================

typedef struct C45_Tree_Info {
	int total_leaves;
	int max_depth;
	int min_depth;
	double avg_depth;
} C45_Tree_Info;

C45_Tree_Info c45_analize_tree_rec(C45_node_t* node, int c_depth )
{
	C45_Tree_Info result;
	
	result.total_leaves = 0;
	result.avg_depth = 0.0;	
	
	if ( node->n_children > 0 ){
		//Call for children...
		int i;
		for ( i = 0; i < node->n_children; i++ ){
			C45_Tree_Info child_info = c45_analize_tree_rec( node->children[i], c_depth + 1 );
			
			result.total_leaves += child_info.total_leaves;
			
			if ( i == 0 ){
				//First child....
				result.max_depth = child_info.max_depth;
				result.min_depth = child_info.min_depth;
			} else {
				//...Min...
				if ( child_info.min_depth < result.min_depth ){
					result.min_depth = child_info.min_depth;
				}
				//...Max...
				if ( child_info.max_depth > result.max_depth ){
					result.max_depth = child_info.max_depth;
				}
			}
			
			result.avg_depth += child_info.avg_depth;
		}
	} else {
		//leaf...
		result.total_leaves = 1;
		result.max_depth = c_depth;
		result.min_depth = c_depth;
		result.avg_depth = c_depth;
	}
	
	return result;
}

C45_Tree_Info* c45_analize_tree(C45_node_t* root )
{
	C45_Tree_Info* tree_info = (C45_Tree_Info*)calloc( sizeof(C45_Tree_Info), 1);
	
	*tree_info = c45_analize_tree_rec(root, 1 );
	
	tree_info->avg_depth /= tree_info->total_leaves;
	
	return tree_info;
}

double c45_tree_info_get_avg_depth( C45_Tree_Info* info ){
	return info->avg_depth;
}

int c45_tree_info_get_min_depth( C45_Tree_Info* info ){
	return info->min_depth;
}

int c45_tree_info_get_max_depth( C45_Tree_Info* info ){
	return info->max_depth;
}

void c45_tree_info_release( C45_Tree_Info* info ) {
	free( info );
}
