#ifndef _ANN_LAYER_JIN
#define _ANN_LAYER_JIN

#include <stdio.h>

#define perror(msg) fprintf(stderr, "Error @ %s %d: %s\n", __FILE__, __LINE__, msg)

/* layer */
typedef struct _NeuralLayer
{
	/* int n_nodes = n_inputs */
	int n_inputs;

	/* int* input; */
	/* vector of [n_nodes] */
	int* result;
	/* matrix of [n_nodes * (n_inputs + 1)] */
	int* weights;

}NeuralLayer;

/* init the ANN layer */
void initLayer(NeuralLayer* layer, int n_inputs);
void releaseLayer(NeuralLayer* layer);

/* training: samples as Matrix of size n_inputs * n_samples */
void setLayerWeights(NeuralLayer* layer, int* samples, int n_samples);

/* startup: set input value */
void setInputValue(NeuralLayer* layer, int* value);

/* caculate until stable OR after too many times */
/* return is_stable */
#define MAX_CACUL_TIMES 1024
int caculate(NeuralLayer* layer);

/* as a permanent object: */
/* store the layerhead & weights */
int readLayer(FILE* file, NeuralLayer* layer);
int writeLayer(FILE* file, NeuralLayer* layer);

#endif	/* end _ANN_LAYER_JIN */
