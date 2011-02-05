#include "NeuralLayer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static void init_weights(int* weights, int len)
{
	memset(weights, 0, len * sizeof(int));
}
/* use layer->n_inputs, layer->n_nodes & layer->last */
void allocLayer(NeuralLayer* layer)
{
	layer->result = (int*)malloc(layer->n_nodes * sizeof(int));
	if (layer->n_inputs)
	{
		layer->weights = (int*)malloc(layer->n_nodes * (layer->n_inputs + 1) * sizeof(int));
	}
	else
	{
		layer->weights = NULL;
	}
}

void initLayer(NeuralLayer* layer, int n_nodes, NeuralLayer* last_layer)
{
	layer->n_nodes = n_nodes;
	layer->n_inputs = (last_layer ? last_layer->n_nodes : 0);
	layer->last = last_layer;

	allocLayer(layer);
	if (layer->weights)
		init_weights(layer->weights, n_nodes * (layer->n_inputs + 1));
}

void releaseLayer(NeuralLayer* layer)
{
	if (layer)
	{
		layer->n_nodes = 0;
		free(layer->result);
		free(layer->weights);
	}
}

/* threshold function */
int sng(int x)
{
	return (x < 0) ? -1 : 1;
}

void caculate(NeuralLayer* layer)
{
	int i, j;
	int sum, sig;
	
	assert(layer);

	layer->stable = 1;
	for (i = 0; i < layer->n_nodes; i++)
	{
		sum = 0;
		for (j = 0; j < layer->n_inputs; j++)
		{
			sum += layer->weights[i * layer->n_inputs + j] * layer->last->result[j];
		}
		/* as threshold: input = 1 */
		sum += layer->weights[i * layer->n_inputs + j] * 1;
		sig = sng(sum);
		
		if (sum && layer->result[i] != sig)
		{
			layer->stable = 0; /* layer changed */
			layer->result[i] = sig;
		}
	}
}

/* directly set value of result, for input layer */
void setLayerValue(NeuralLayer* input_layer, int* value)
{
	memcpy(input_layer->result, value, input_layer->n_nodes * sizeof(int));
}
