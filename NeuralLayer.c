#include "NeuralLayer.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void init_weights(float* weights, int n)
{
	int i;
	for (i = 0; i < n; i++)
		weights[i] = (float)(rand() % 65535) / 65535.0;
}

void initLayer(NeuralLayer* layer, int n_nodes, NeuralLayer* last_layer)
{
	layer->n_nodes = n_nodes;
	layer->n_inputs = (last_layer ? last_layer->n_nodes : 0);
	layer->last = last_layer;

	layer->result = (float*)malloc(n_nodes * sizeof(float));
	if (last_layer)
	{
		layer->error = (float*)malloc(n_nodes * sizeof(float));
		layer->weights = (float*)malloc(n_nodes * (layer->n_inputs + 1) * sizeof(float));

		srand((unsigned int)time(NULL));
		init_weights(layer->weights, n_nodes * (layer->n_inputs + 1));
	}
	else
	{
		layer->error = NULL;
		layer->weights = NULL;
	}
	
	layer->learning_rate = 0.1;
	layer->exp_rate = 1;
}

void releaseLayer(NeuralLayer* layer)
{
	if (layer)
	{
		layer->n_nodes = 0;
		free(layer->result);
		free(layer->error);
		free(layer->weights);
	}
}

/* threshold function */
float threshold(float exp_rate, float x)
{
	return 1 / (1 + expf(-exp_rate * x));
}
float threshold_slope_strick(float exp_rate, float y)
{
 	return exp_rate * y * (1 - y);
}
float threshold_slope_approx(float exp_rate, float y)
{				/* approximately */
	return exp_rate / 4;
}
#ifndef THRESHOLD_SLOPE_USE_STRICK
#	define threshold_slope threshold_slope_approx
#else
#	define threshold_slope threshold_slope_strick
#endif

void caculate(NeuralLayer* layer)
{
	int i, j;
	float sum;
	
	assert(layer && layer->last);
	
	for (i = 0; i < layer->n_nodes; i++)
	{
		sum = 0;
		for (j = 0; j < layer->n_inputs; j++)
		{
			sum += layer->weights[i * layer->n_inputs + j] * layer->last->result[j];
		}
		/* as threshold: input = 1 */
		sum += layer->weights[i * layer->n_inputs + j] * 1;
		layer->result[i] = threshold(layer->exp_rate, sum);
	}
}


/* by given destination */
float countFinalError(NeuralLayer* final_layer, float* dest)
{
	int i;
	float differ, sum_err = 0;
	for (i = 0; i < final_layer->n_nodes; i++)
	{
		differ = dest[i] - final_layer->result[i];
		sum_err += differ * differ;
		
		final_layer->error[i] =				\
			differ * threshold_slope(final_layer->exp_rate, final_layer->result[i]);
	}
	return sum_err;
}

/* by per-settled errors: next_layer->error */
void countHiddenError(NeuralLayer* next_layer);

/* use errors already count */
void adjustWeights(NeuralLayer* layer)
{
	int i, j;
	for (i = 0; i < layer->n_nodes; i++)
	{
		for (j = 0; j < layer->n_inputs; j++)
		{
			layer->weights[i * layer->n_inputs + j] +=	\
				layer->learning_rate * layer->error[i] * layer->last->result[j];
		}
		/* threshold: as input = 1 (OR -1) */
		layer->weights[i * layer->n_inputs + j] +=		\
			layer->learning_rate * layer->error[i] * 1;
	}
}

/* directly set value of result, for input layer */
void setLayerValue(NeuralLayer* input_layer, float* value)
{
	memcpy(input_layer->result, value, input_layer->n_nodes * sizeof(float));
}

/* n_nodes: from input towords output, length: n_hidden + 2  */
void initNet(NeuralNet* net, int n_hidden, int* n_nodes);
void releaseNet(NeuralNet* net);

void evolution(NeuralNet* net, float* input, float* dest);
