#include "NeuralLayer.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void init_weights(float* weights, int n)
{
	int i;
	for (i = 0; i < n; i++)
		weights[i] = (float)((rand() % 65536) / 32768.0 - 1);
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
#define THRESHOLD_SLOPE_USE_STRICK

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
void countHiddenError(NeuralLayer* next_layer)
{
	NeuralLayer* layer = next_layer->last;
	int i, j;
	
	if (!layer) return;
	for (i = 0; i < layer->n_nodes; i++)
	{
		float error = 0;
		for(j = 0; j < next_layer->n_nodes; j++)
		{
			error += next_layer->error[j] * next_layer->weights[j * next_layer->n_nodes + i];
		}
		layer->error[i] = error * threshold_slope(layer->exp_rate, layer->result[i]);
	}
}

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

void initNet(NeuralNet* net, int n_layers, int* n_nodes)
{
	int i;
	NeuralLayer* layers;

	assert(n_layers >= 2);
	
	net->n_hidden = n_layers - 2;
	net->epsino = 0.05;
	
	/* allocate memory */
	layers = (NeuralLayer*)malloc(n_layers * sizeof(NeuralLayer));
	
	net->input = &layers[0];
	net->hidden = (net->n_hidden ? &layers[1] : NULL);
	net->output = &layers[n_layers - 1];

	/* init each Layer */
	initLayer(&layers[0], n_nodes[0], NULL);
	for (i = 1; i < n_layers; i++)
	{
		initLayer(&layers[i], n_nodes[i], &layers[i-1]);
	}

	/* the end */
}
void releaseNet(NeuralNet* net)
{
	int i;
	for (i = 0; i < net->n_hidden + 2; i++)
	{
		releaseLayer(&(net->input[i]));
	}
	free(net->input);
	net->n_hidden = 0;
	net->input = net->hidden = net->output = NULL;
}

void caculateNet(NeuralNet* net, float* input)
{
	int i;
	/* input layer */
	setLayerValue(net->input, input);
	/* hidden layers */
	for (i = 0; i < net->n_hidden; i++)
	{
		caculate(&(net->hidden[i]));
	}
	/* output layer */
	caculate(net->output);
}

float evolveNet(NeuralNet* net, float* input, float* dest)
{
	int i;
	float error;
	/* forward caculate */
	caculateNet(net, input);

	/* backword adjust */
	
 	/* 1. caculate error */
	error = countFinalError(net->output, dest) / 2;
	/* for hidden layers if exist */
	for (i = net->n_hidden; i > 0; i--)
	{
		/* nextlayer = hidden[n_hidden .. 1] */
		countHiddenError(&net->hidden[i]);
	}
	/* 2. adjust wieghts */
	adjustWeights(net->output);
	for (i = 0; i < net->n_hidden; i++)
	{
		adjustWeights(&net->hidden[i]);
	}
	return error;
}


