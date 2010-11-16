#include "NeuralNode.h"

#include <stdlib.h>

void initNeuralNode(NeuralNode* node)
{
	int i;
	srand((unsigned int)time(NULL));
	for (i = 0; i < MAX_INPUT + 1; i++)
	{
		node->weights[i] = rand() % 2000 / 1000.0 - 1;
	}
}

float adjust_out(float output)
{
	return 1 / (1 + expf(-EXP_RATE * output));
}
float slope_at_strick(float output)
{
 	return EXP_RATE * output * (1 - output);
}

float slope_at_approx(float error)
{				/* approximately */
	return EXP_RATE / 4;
}

void caculate(NeuralNode* node, NeuralNode* inputs)
{
	float sum = 0;
	int i;
	for (i = 0; i < MAX_INPUT; i++)
	{
		sum += inputs[i].result * node->weights[i];
	}
	sum += node->weights[MAX_INPUT];
	node->result =  adjust_out(sum);
}

float count_final_error(NeuralNode* node, float destination)
{
	float error = destination - node->result;
	node->error = error * slope_at(node->result);
	return error;
}

/* give each input node an error rate */
/*
void count_hidden_error(NeuralNode* node, NeuralNode* inputs)
{
	int i;
	float sum = 0, sum_err = adjust_back(node->error);
	for (i = 0; i < MAX_INPUT; i++)
	{
		sum += node->weights[i];
	}
	for (i = 0; i < MAX_INPUT; i++)
	{
		inputs[i].error = sum_err / sum * node->weights[i];
	}
}
*/

void learn(NeuralNode* node, NeuralNode* inputs, float destnation)
{
	int i;
	/* float sum_err = node->error; */
	for (i = 0; i < MAX_INPUT; i++)
	{
		node->weights[i] += (LEARNING_RATE * node->error * inputs[i].result);
	}
	node->weights[MAX_INPUT] += (LEARNING_RATE * node->error * 1);
}
