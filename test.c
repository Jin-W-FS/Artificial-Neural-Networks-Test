#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "NeuralLayer.h"

/* train: y = (sin(2 * pi * x) + 1)/2, x in [0, 1) */
#define func(x) ((sin(2 * 3.14159265 * (x)) + 1) / 2)
#define N_SAMPLES 13
#define SUM_ERROR 0.01

const char* save_net = "./NeuralNetwork.log";

void gen_samples(float input[N_SAMPLES], float result[N_SAMPLES])
{
	int i;
	float dim = 1.0 / (N_SAMPLES - 1);
	for (i = 0; i < N_SAMPLES; i++)
	{
		input[i] = dim * i;
		result[i] = func(input[i]);
	}
}

int main(int argc, char* argv[])
{
	NeuralNet net;
	int n_nodes[] = {
		1, 23, 1
	};
	float input, result, error, sumerr;
	int i, n = 0;

	float sample_input[N_SAMPLES];
	float sample_result[N_SAMPLES];
	
	FILE* fl = NULL;

	/* init samples */
	gen_samples(sample_input, sample_result);

	/* init net, or load from file */
	if (argc >= 2 && strcmp(argv[1], "-l") == 0)
	{
		fl = fopen(save_net, "r");
	}

	if (fl)
	{
		readNet(fl, &net);
		fclose(fl);
	}
	else
	{
		initNet(&net, 3, n_nodes);
	}
	
	/* train */
	do
	{
		sumerr = 0;
		for (i = 0; i < N_SAMPLES; i++)
		{
			sumerr += evolveNet(&net, &sample_input[i], &sample_result[i]);
		}
		printf("G %d : sum err = %f\n", n++, sumerr);
	}while(sumerr > SUM_ERROR);

	/* print sample */
	printf("\nnow start a test:\n samples:\n");
	for (i = 0; i < N_SAMPLES; i++)
	{
		printf("sin(2 * pi * %f) = %f\n", sample_input[i], sample_result[i] * 2 - 1);
	}

	/* run test */
	while(scanf("%f", &input) == 1)
	{
		caculateNet(&net, &input);
		result = net.output->result[0];
		printf("sin(2 * pi * %f) = %f\n", input, result * 2 - 1);
	}

	/* print & save Net */
	writeNet(stdout, &net);
	if ((fl = fopen(save_net, "w")) != 0)
		writeNet(fl, &net);
	else
		fprintf(stderr, "Error in Save Net.");

	releaseNet(&net);
	
	return 0;
}

