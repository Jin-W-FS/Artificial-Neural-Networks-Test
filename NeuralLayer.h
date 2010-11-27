#ifndef _ANN_LAYER_JIN
#define _ANN_LAYER_JIN

/* layer */
typedef struct _NeuralLayer
{
	int n_nodes;
	int n_inputs;
	
	struct _NeuralLayer * last;

	/* vector of [n_nodes] */
	float* result;
	float* error;
	/* matrix of [n_nodes * (last->n_nodes + 1)] */
	float* weights;

	/* arguments */
	float learning_rate;
	float exp_rate;
}NeuralLayer;

/* init the ANN layer; on input layer last_layer = 0 */
void initLayer(NeuralLayer* layer, int n_nodes, NeuralLayer* last_layer);
void releaseLayer(NeuralLayer* layer);

/* notice: functions below will apply the caculate without recursive;
 * i don't think it convient, however, perhaps it's more flexable in C
 * as you can do what you want between any continus two steps.
 */

/*  */
void caculate(NeuralLayer* layer);

/* by given destination
   return sum of error^2 */
float countFinalError(NeuralLayer* final_layer, float* dest);
/* by per-settled errors: next_layer->error */
void countHiddenError(NeuralLayer* next_layer);

/* use errors already count */
void adjustWeights(NeuralLayer* layer);

/* directly set value of result, for input layer */
void setLayerValue(NeuralLayer* input_layer, float* value);

/* net */
typedef struct _NeuralNet
{
	int n_hidden;
	float epsino;
	
	NeuralLayer* output;
	NeuralLayer* hidden;
	NeuralLayer* input;
}NeuralNet;

/* n_nodes: nodes of each layer, from input towords output, length: n_layers; */
void initNet(NeuralNet* net, int n_layers, int* n_nodes);
void releaseNet(NeuralNet* net);

void caculateNet(NeuralNet* net, float* input);
/* use input[n] & dest[n] to evolve once */
float evolveNet(NeuralNet* net, float* input, float* dest);

#endif	/* end _ANN_LAYER_JIN */
