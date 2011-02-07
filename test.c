#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "NeuralLayer.h"

#ifndef PI
#	define PI 3.14159265
#endif

/* do a liner adjust: from in_sect[] to out_sect[]*/
float SECTION_0_1[2] = { 0.0f, 1.0f };
float adjust(float x, float in_sect[2], float out_sect[2]);

/*
  train: y = (sin(x) + cos(x)) / 2, -pi <= x <= pi,
  x0 = (x + pi) / 2pi, 0 <= x0 <= 1,
  y0 = (y + 1) / 2, 0 <= y0 <=1
*/

/* y = func(x) */
float func(float x);
float input_sect[2] = { -PI, PI };
float output_sect[2] = { -1, 1 };

#define N_SAMPLES 13
#define SUM_ERROR 0.001

const char* save_net = "./NeuralNetwork.log";
#define ERROR_OUTPUT stdout
#define TEST_INPUT stdin
#define TEST_OUTPUT stdout

/* x => x0 */
float in_adj_rev(float x)
{
	return adjust(x, input_sect, SECTION_0_1);
}
/* x0 => x */
float in_adj(float x0){
	return adjust(x0, SECTION_0_1, input_sect);
}
/* y => y0 */
float out_adj_rev(float y){
	return adjust(y, output_sect, SECTION_0_1);
}
/* y0 => y */
float out_adj(float y0){
	return adjust(y0, SECTION_0_1, output_sect);
}

void gen_samples(float input[N_SAMPLES], float result[N_SAMPLES])
{
	int i;
	float dim = 1.0 / (N_SAMPLES - 1);
	for (i = 0; i < N_SAMPLES; i++)
	{
		input[i] = in_adj(dim * i);
		result[i] = out_adj_rev(func(input[i]));
	}
}

int main(int argc, char* argv[])
{
	NeuralNet net;
	int n_nodes[] = {
		1, 23, 1
	};
	float input, error, sumerr;
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
		/* print Generation & errors */
		fprintf(ERROR_OUTPUT, "%d\t%f\n", n++, sumerr);
	}while(sumerr > SUM_ERROR);
	
	/* run test */
	while(fscanf(TEST_INPUT, "%f", &input) == 1)
	{
		caculateNet(&net, &input);
		fprintf(TEST_OUTPUT, "%f\n", out_adj(net.output->result[0]));
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

float func(float x)
{
	return ((sin(x) + cos(x)) / 2);
}

float adjust(float x, float in_sect[2], float out_sect[2])
{
	assert(in_sect[1] - in_sect[0] != 0);
	return out_sect[0] +					\
		(x - in_sect[0]) / (in_sect[1] - in_sect[0]) *	\
		(out_sect[1] - out_sect[0]);
}

