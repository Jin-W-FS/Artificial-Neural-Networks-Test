#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* posix: getopt */
#include <unistd.h>

#include "NeuralLayer.h"

#ifndef PI
#	define PI 3.14159265
#endif

/* do a liner adjust: from in_sect[] to out_sect[]*/
float SECTION_0_1[2] = { 0.0f, 1.0f };
extern float input_sect[2];
extern float output_sect[2];

float adjust(float x, float in_sect[2], float out_sect[2]);

/* x => x0 */
float in_adj_rev(float x){
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

/* Infomation of the net */
/*
  train: y = (sin(x) + cos(x)) / 2, -pi <= x <= pi,
  x0 = (x + pi) / 2pi, 0 <= x0 <= 1,
  y0 = (y + 1) / 2, 0 <= y0 <=1
*/

/* y = func(x) */

#define N_SAMPLES 11
#define N_INPUTS  1
#define N_OUTPUTS 2
#define N_HIDDEN_ADVISE  (N_INPUTS + (int)(0.618 * (N_INPUTS - N_OUTPUTS) + 1))
#define N_HIDDEN  4

#define SUM_ERROR 0.000001

float input_sect[2] = { 0, 1 };
float output_sect[2] = { -PI, PI };

int n_nodes[] = {
	N_INPUTS, N_HIDDEN, N_OUTPUTS
};

void gen_samples(float input[N_SAMPLES][N_INPUTS], float result[N_SAMPLES][N_OUTPUTS]);
int read_input(float input[N_INPUTS], FILE* in);
int write_output(float output[N_OUTPUTS], FILE* out);

/* args analyse */
struct _globalConfig
{
	/* mode */
	int quiet;		/* -q */
	
	/* config file names */
	const char* net_load;	/* -l,-n, default: retrain the net */
	const char* net_save;	/* -s,-n, default: none */
	const char* error_output; /* -e, default: none */
	const char* test_input;	  /* -i, default: stdin */
	const char* test_output;  /* -o, default: stdout */
}globalConfig;
const char* opts_analyse = "qn:l:s:e:i:o:";
int opts_analyser(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	NeuralNet net;
	float error, sumerr;
	int i, n = 0;
	
	float sample_input[N_SAMPLES][N_INPUTS];
	float sample_result[N_SAMPLES][N_OUTPUTS];
	float test_input[N_INPUTS];
	float test_output[N_OUTPUTS];

	FILE *fnet, *ferror, *ftest_in, *ftest_out;
	
	/* opts analyse */
	if (opts_analyser(argc, argv) < 0)
	{
		perror("opts_anal");
		return -1;
	}

	/* init samples */
	if (!globalConfig.quiet)
		fprintf(stderr, "loading samples...");
	gen_samples(sample_input, sample_result);
	
	/* init net, or load from file */
	fnet = NULL;
	if (globalConfig.net_load && (fnet = fopen(globalConfig.net_load, "r")))
	{
		if (!globalConfig.quiet)
			fprintf(stderr, "complete\nloading net...");
		readNet(fnet, &net);
		fclose(fnet);
	}
	else
	{
		initNet(&net, 3, n_nodes);
	}
	
	/* train */
	if (!globalConfig.quiet)
			fprintf(stderr, "complete\ntraining...");
	ferror = NULL;
	if (globalConfig.error_output)
		ferror = fopen(globalConfig.error_output, "w");
	do
	{
		sumerr = 0;
		for (i = 0; i < N_SAMPLES; i++)
		{
			sumerr += evolveNet(&net, sample_input[i], sample_result[i]);
		}
		/* print Generation & Errors */
		if (ferror)
			fprintf(ferror, "%d\t%f\n", n++, sumerr);
		
	}while(sumerr > SUM_ERROR);
	if (ferror)
		fclose(ferror);
	
	/* run test */
	if (!globalConfig.quiet)
		fprintf(stderr, "complete\nNow start a test:\n");

	if (globalConfig.test_input)
		ftest_in = fopen(globalConfig.test_input, "r");
	else
		ftest_in = stdin;

	if (globalConfig.test_output)
		ftest_out = fopen(globalConfig.test_output, "w");
	else
		ftest_out = stdout;
	
	while(read_input(test_input, ftest_in) == N_INPUTS)
	{
		caculateNet(&net, test_input);
		for (i = 0; i < N_OUTPUTS; i++)
		{
			test_output[i] = out_adj(net.output->result[i]);
		}
		write_output(test_output, ftest_out);
	}
	if (ftest_in != stdin)
		fclose(ftest_in);
	if (ftest_out != stdout)
		fclose(ftest_out);

	
	/* print & save Net */
	if (globalConfig.net_save && (fnet = fopen(globalConfig.net_save, "w")))
	{
		if (!globalConfig.quiet)
			fprintf(stderr, "writing net...");
		writeNet(fnet, &net);
		fclose(fnet);
		if (!globalConfig.quiet)
			fprintf(stderr, "complete\nexit.\n");

	}
	
	releaseNet(&net);
	
	return 0;
}

float adjust(float x, float in_sect[2], float out_sect[2])
{
	assert(in_sect[1] - in_sect[0] != 0);
	return out_sect[0] +					\
		(x - in_sect[0]) / (in_sect[1] - in_sect[0]) *	\
		(out_sect[1] - out_sect[0]);
}

void gen_samples(float input[N_SAMPLES][N_INPUTS], float result[N_SAMPLES][N_OUTPUTS])
{
	FILE* sample_input = fopen("samples.log", "r");
	int i, j;
	float tmp;
	for (i = 0; i < N_SAMPLES; i++)
	{
		for (j = 0; j < N_INPUTS; j++)
		{
			fscanf(sample_input, "%f", &tmp);
			input[i][j] = tmp;
		}
		for (j = 0; j < N_OUTPUTS; j++)
		{
			fscanf(sample_input, "%f", &tmp);
			result[i][j] = out_adj_rev(tmp);
		}
	}
}
int read_input(float input[N_INPUTS], FILE* in)
{
	int i;
	for (i = 0; i < N_INPUTS; i++)
	{
		if (fscanf(in, "%f", &input[i]) != 1)
			break;
	}
	return i;
}

int write_output(float output[N_OUTPUTS], FILE* out)
{
	int i;
	for (i = 0; i < N_OUTPUTS; i++)
	{
		if (fprintf(out, "%f ", output[i]) < 0)
			break;
	}
	fputc('\n', out);
	return i;
}

int opts_analyser(int argc, char* argv[])
{
	int opt;

	globalConfig.quiet = 0;
	globalConfig.net_load = NULL;
	globalConfig.net_save = NULL;
	globalConfig.error_output = NULL;
	globalConfig.test_input = NULL;
	globalConfig.test_output = NULL;
	
	while ((opt = getopt(argc, argv, opts_analyse)) != -1)
	{
		switch (opt)
		{
		case 'q':
			globalConfig.quiet = 1;
			break;
		case 'n':
			globalConfig.net_save = globalConfig.net_load = optarg;
			break;
		case 'l':
			globalConfig.net_load = optarg;
			break;
		case 's':
			globalConfig.net_save = optarg;
			break;
		case 'e':
			globalConfig.error_output = optarg;
			break;
		case 'i':
			globalConfig.test_input = optarg;
			break;
		case 'o':
			globalConfig.test_output = optarg;
			break;
		default:
			break;
		}
	}
	return 0;
}
