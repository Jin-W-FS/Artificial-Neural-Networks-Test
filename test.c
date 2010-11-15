#include <stdio.h>
#include <stdlib.h>

#include "NeuralNode.h"

typedef struct _NeuralWeb
{
	NeuralNode* input;
	NeuralNode* hidden;
	NeuralNode* output;
}NeuralWeb;

void init(NeuralWeb* web);
void release(NeuralWeb* web);

void init_input_Nodes(NeuralNode* inputNode, float* inputf, int n);

void train(NeuralWeb* web);
float apply(NeuralWeb* web);

float model_inputs[4][MAX_INPUT] = {
	0, 0, 0, 1, 1, 0, 1, 1,
};
float model_answer[4] = {
	0, 0, 0, 1
};

int main()
{
	NeuralWeb web;
	
	float input[2];
	
	init(&web);
	train(&web);
	
	printf("now start a test:\n");
	printf("with weights: %f, %f, %f\n",
	       web.output->weights[0], web.output->weights[1], web.output->weights[2]);

	while(scanf("%f %f", &input[0], &input[1]) == 2)
	{
		init_input_Nodes(web.input, input, 2);
		printf("%f && %f = %f\n", input[0], input[1], apply(&web));
	}

	return 0;
}

/* now just a 2-plant web */
void init(NeuralWeb* web)
{
	web->input = (NeuralNode*)malloc(MAX_INPUT * sizeof(NeuralNode));
	web->hidden = NULL;
	web->output = (NeuralNode*)malloc(sizeof(NeuralNode));
}
void release(NeuralWeb* web)
{
	free(web->input);
	free(web->output);
	web->input = NULL;
	web->output = NULL;
}

float apply(NeuralWeb* web)
{
	caculate(web->output, web->input);
	return web->output->result;
}

void init_input_Nodes(NeuralNode* inputNode, float* inputf, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		inputNode[i].result = inputf[i];
	}
}

/* now just a 2-plant web, use models above */
void train(NeuralWeb* web)
{
	int i, pass;

	initNeuralNode(web->output);

	do
	{
		pass = 1;
		for (i = 0; i < 4; i++)
		{
			init_input_Nodes(web->input, model_inputs[i], MAX_INPUT);
			apply(web);

			printf("%f %f\t%f\n",
			       web->input[0].result, web->input[1].result, web->output->result);

			count_final_error(web->output, model_answer[i]);
			
			if (!LITTLE_ERROR(web->output->error))
			{
				pass = 0;
				/* count_hidden_error */
				learn(web->output, web->input, model_answer[i]);
			}
		}
		putchar('\n');
	}while(!pass);
}
