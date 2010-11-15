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
float adjust_back(float error)
{				/* approximately */
	return error * 4 / EXP_RATE;
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

void count_final_error(NeuralNode* node, float destination)
{
	node->error = destination - node->result;
}
void count_hidden_error(NeuralNode* node, NeuralNode* inputs)
{		/* give each input node an error rate */
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


void learn(NeuralNode* node, NeuralNode* inputs, float destnation)
{
	int i;
	/* float sum_err = node->error; */
	float sum_err = adjust_back(node->error);
	for (i = 0; i < MAX_INPUT; i++)
	{
		node->weights[i] += (LEARNING_RATE * sum_err * inputs[i].result);
	}
	node->weights[MAX_INPUT] += (LEARNING_RATE * sum_err * 1);
}
