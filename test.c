#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "NeuralLayer.h"

const char* save_net = "./NeuralNetwork.log";
const char* f_samples = "./samples.log";

#define LINE_LEN 10
/* #line of one image */
#define LINE_IMG 12
#define N_INPUTS (LINE_LEN * LINE_IMG)
#define N_SAMPLES 8
#define SWAP_CHANCE 0.25

const char sample_char[N_SAMPLES] = {
	'0', '1', '2', '3', '4', '6', '*', '9'
};

void gen_samples(int samples[N_SAMPLES][N_INPUTS], FILE* input);
void get_image(int* image, FILE* input);
void show_image(int image[N_INPUTS], FILE* output);
void swap_image(int dst[N_INPUTS], int src[N_INPUTS], float chance);

int main(int argc, char* argv[])
{
	NeuralLayer hnet;

	int *samples = malloc(N_SAMPLES * N_INPUTS * sizeof(int));
	int *test_input = malloc(N_INPUTS * sizeof(int));

	FILE* fl = NULL;
	
	/* get samples */
	FILE* sample_input_file = fopen(f_samples, "r");
	gen_samples((int (*)[N_INPUTS])samples, sample_input_file);
	fclose(sample_input_file);

	/* init net : load from file OR train from samples */
	if (argc >= 2 && strcmp(argv[1], "-l") == 0)
	{
		fl = fopen(save_net, "r");
	}

	if (fl)
	{
		readLayer(fl, &hnet);
		fclose(fl);
	}
	else
	{
		initLayer(&hnet, N_INPUTS);
		setLayerWeights(&hnet, samples, N_SAMPLES);
	}
	
	/* run test */
	printf("now start a test,\n\tinput a char of (012346*9):");
	while (1)
	{
		char c;
		int i;
		scanf(" %c", &c);
		for (i = 0; i < N_SAMPLES; i++)
		{
			if (c == sample_char[i])
				break;
		}
		/* on illegal input, exit while */
		if (i == N_SAMPLES)
			break;
		/* else: caculate net */
		swap_image(test_input, samples + i * N_INPUTS, SWAP_CHANCE);
		setInputValue(&hnet, test_input);
		caculate(&hnet);

		/* show diff */
		show_image(test_input, stdout);
		fputc('\n', stdout);
		show_image(hnet.result, stdout);
		fputc('\n', stdout);
	}

	/* print & save Net */
	fl = fopen(save_net, "w");
	writeLayer(fl, &hnet);
	fclose(fl);

/* release resources */
	releaseLayer(&hnet);
	free(test_input);
	free(samples);
	
	return 0;
}

/* image of size LINE_LEN * LINE_IMG */
void get_image(int* image, FILE* input)
{
	/* +2 'cause fgets stores the '\n\0' at end */
	char line[LINE_LEN + 2];
	int i, j;
	for (i = 0; i < LINE_IMG; i++)
	{
		fgets(line, LINE_LEN + 2, input);
		for (j = 0; j < LINE_LEN; j++)
		{
			/* '#' / +1 : black, ' ' / -1 : white */
			*image++ = (line[j] == '#' ? 1 : -1);
		}
	}
}

void show_image(int* image, FILE* output)
{
	int i, j;
	for (i = 0; i < LINE_IMG; i++)
	{
		for (j = 0; j < LINE_LEN; j++)
		{
			fputc(((*image++) > 0 ? '#' : ' '), output);
		}
		fputc('\n', output);
	}
}

void gen_samples(int samples[N_SAMPLES][N_INPUTS], FILE* input)
{
	int i;
	char line[2];
	for (i = 0; i < N_SAMPLES; i++)
	{
		get_image(samples[i], input);
		/* eat the blank line separating images */
		fgets(line, 2, input);
		assert(line[0] == '\n');
	}
}

#define randf() ((float)(rand() % 65536) / 65536.0f)
void swap_image(int dst[N_INPUTS], int src[N_INPUTS], float chance)
{
	int i;
	for (i = 0; i < N_INPUTS; i++)
	{
		if (randf() < chance)
			dst[i] = -src[i];
		else
			dst[i] = src[i];
	}
}

