#include "NeuralLayer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

void initLayer(NeuralLayer* layer, int n_inputs)
{
	assert(n_inputs > 0);
	layer->n_inputs = n_inputs;
	layer->result = (int*)malloc(layer->n_inputs * sizeof(int));
	layer->weights = (int*)malloc(layer->n_inputs * (layer->n_inputs + 1) * sizeof(int));
}

void releaseLayer(NeuralLayer* layer)
{
	if (layer)
	{
		layer->n_inputs = 0;
		free(layer->result);
		free(layer->weights);
	}
}

int sgn(int x)
{
	return ((x == 0) ? 0 : ((x < 0) ? -1 : 1));
}

/* directly set value of result, for input layer */
void setInputValue(NeuralLayer* layer, int* value)
{
	int i;
	for (i = 0; i < layer->n_inputs; i++)
	{
		layer->result[i] = sgn(value[i]);
	}
}

#define weight_at(layer, i_node, i_input) (layer->weights[(i_node) * (layer->n_inputs + 1) + (i_input)])
/* return is_stable */
int cacul_once(NeuralLayer* layer)
{
	int i, j;
	int sum, sig;
	int is_stable;

	assert(layer);

	is_stable = 1;
	for (i = 0; i < layer->n_inputs; i++)
	{
		sum = 0;
		for (j = 0; j < layer->n_inputs; j++)
		{
			sum += weight_at(layer, i, j) * layer->result[j];
		}
		/* the threshold: theta = 1 */
		sum += weight_at(layer, i, j) * 1;
		
		sig = sgn(sum);
		
		if (sig == 0 || sig == layer->result[i])
		{
			/* unchange: is_stable &&= 1 */
		}
		else
		{
			layer->result[i] = sig;
			is_stable = 0;
		}
	}
	return is_stable;
}
int caculate(NeuralLayer* layer)
{
	int n_times;
	for (n_times = 0; n_times < MAX_CACUL_TIMES; n_times++)
	{
		if (cacul_once(layer))
			return 1;
	}
	return 0;
}

/* training: samples as Matrix of size n_inputs * n_samples */
void setLayerWeights(NeuralLayer* layer, int* samples, int n_samples)
{
	int i, j, k;
	for (i = 0; i < layer->n_inputs; i++)
	{
		for (j = 0; j < i; j++)
		{
			int tmp = 0;
			for (k = 0; k < n_samples; k++)
			{
				tmp += samples[k * layer->n_inputs + i] * samples[k * layer->n_inputs + j];
			}
			weight_at(layer, i, j) = tmp;
			weight_at(layer, j, i) = tmp;
		}
		weight_at(layer, i, i) = 0;
		weight_at(layer, i, layer->n_inputs) = 0;
	}
}

	
/* store the layer: n_inputs & weights */
const char* LAYER_HEADER = "%d";

static int readMat(FILE* file, int* mat, int width, int height);
static int writeMat(FILE* file, int* mat, int width, int height);

int writeLayer(FILE* file, NeuralLayer* layer)
{
	int i;
	fprintf(file, LAYER_HEADER, layer->n_inputs);
	fputc('\n', file);
	if (writeMat(file, layer->weights, layer->n_inputs + 1, layer->n_inputs) < 0)
	{
		perror("write mat");
		return -1;
	}
	
	fputc('\n', file);
	return 0;
}
int readLayer(FILE* file, NeuralLayer* layer)
{
	int n_inputs;
	if (fscanf(file, LAYER_HEADER, &n_inputs) != 1)
	{
		perror("scan n_inputs");
		return -1;
	}
	
	/* allocate */
	initLayer(layer, n_inputs);
	/* init weight */
	if (readMat(file, layer->weights, layer->n_inputs + 1, layer->n_inputs) < 0)
	{
		perror("read mat");
		return -1;
	}
	return 0;
}

static int readMat(FILE* file, int* mat, int width, int height)
{
	int i, j;
	if (mat == NULL)	/* input layer */
		return 0;
	
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (fscanf(file, "%d", mat++) != 1)
			{
				perror("read mat");
				return -1;
			}
		}
	}
	return 0;
}
static int writeMat(FILE* file, int* mat, int width, int height)
{
	int i, j;
	if (mat == NULL)	/* input layer */
		return 0;
	
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (fprintf(file, "%d ", *mat++) < 0)
			{
				perror("write mat");
				return -1;
			}
		}
		fputc('\n', file);
	}
	return 0;
}
