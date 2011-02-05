#ifndef _ANN_LAYER_JIN
#define _ANN_LAYER_JIN

#include <stdio.h>

/* layer */
typedef struct _NeuralLayer
{
	int n_nodes;
	int n_inputs;

	struct _NeuralLayer* last;
	
	/* vector of [n_nodes] */
	int* result;
	/* matrix of [n_nodes * (last->n_nodes + 1)] */
	int* weights;

	/* arguments */
	int stable;
}NeuralLayer;

/* init the ANN layer; on input layer last_layer = 0 */
void initLayer(NeuralLayer* layer, int n_nodes, NeuralLayer* last_layer);
void releaseLayer(NeuralLayer* layer);

/* store the layer: n_nodes, n_inputs, learning_rate, exp_rate, weights. */
int readLayer(FILE* file, NeuralLayer* layer);
int writeLayer(FILE* file, NeuralLayer* layer);

/* notice: functions below will apply the caculate without recursive;
 * i don't think it convient, however, perhaps it's more flexable in C
 * as you can do what you want between any continus two steps.
 */

/*  */
void caculate(NeuralLayer* layer);

/* directly set value of result, for input layer */
void setLayerValue(NeuralLayer* input_layer, int* value);

#endif	/* end _ANN_LAYER_JIN */
