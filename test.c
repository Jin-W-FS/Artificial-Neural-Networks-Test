#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "NeuralLayer.h"
char font[6][17] = {
	"****"					\
	"*  *"					\
	"*  *"					\
	"****",			/* 0 */

<<<<<<< HEAD
	"  * "					\
	"  * "					\
	"  * "					\
	"  * ",			/* 1 */
=======
/* train: y = (sin(2 * pi * x) + 1)/2, x in [0, 1) */
#define func(x) ((sin(2 * 3.14159265 * (x)) + 1) / 2)
#define N_SAMPLES 13
#define SUM_ERROR 0.01

const char* save_net = "./NeuralNetwork.log";
>>>>>>> BP-Network

	"****"					\
	"  **" 					\
	"**  "					\
	"****",			/* 2 */

	"****"					\
	"  * "					\
	"   *"					\
	"*** ",			/* 3 */
	
	"* * "					\
	"* * "					\
	"****"					\
	"  * ",			/* 4 */

	"****"					\
	"*** "					\
	"   *"					\
	"*** ",			/* 5 */
};
void print_font(char* font)
{
	int i;
	for (i = 0; i < 4; i++)
	{
		printf("%.4s\n", font + 4 * i);
	}
	putchar('\n');
}

		
void char_to_int(char* src, int* dst)
{
<<<<<<< HEAD
	int i;
	for (i = 0; i < 16; i++)
=======
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
>>>>>>> BP-Network
	{
		dst[i] = ((src[i] == '*') ? 1 : -1);
	}
}
void int_to_char(int* src, char* dst)
{
	int i;
	for (i = 0; i < 16; i++)
	{
		dst[i] = ((src[i] == 1) ? '*' : ' ');
	}
}
int sample[6][16];

void setsample()
{
	int i, j;
	for (i = 0; i < 6; i++)
	{
		char_to_int(font[i], sample[i]);
	}
}

void train(NeuralLayer* hnet, int sample[6][16])
{
	int i, j, k, sum;
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 16; j++)
		{
			if (i == j)
				hnet->weights[i * (hnet->n_inputs + 1) + j] = 0;
			else
			{
				sum = 0;
				for (k = 0; k < 6; k++)
				{
					sum += sample[k][i] * sample[k][j];
				}
				hnet->weights[i * (hnet->n_inputs + 1) + j] = sum;
			}
		}
<<<<<<< HEAD
	}

}

void test(NeuralLayer* hnet, char* input)
{
	int n = 0;
	
	char buffer[17];
	buffer[16] = '\0';
	
	char_to_int(input, hnet->result);
	print_font(input);

	do
=======
		printf("G %d : sum err = %f\n", n++, sumerr);
	}while(sumerr > SUM_ERROR);

	/* print sample */
	printf("\nnow start a test:\n samples:\n");
	for (i = 0; i < N_SAMPLES; i++)
>>>>>>> BP-Network
	{
		caculate(hnet);
		printf("-- %d --\n", ++n);
		int_to_char(hnet->result, buffer);
		print_font(buffer);
	}while(!hnet->stable);
}

<<<<<<< HEAD
char test_sample[17] =
	"****"					\
	"  **"					\
	"**  "					\
	"****";

#define THRESHOLD 0.1
#define REVERSEC(c) (((c) == '*') ? ' ' : '*')

void make_noise(char* test_sample)
{
	int i;
	for (i = 0; i < 16; i++)
=======
	/* run test */
	while(scanf("%f", &input) == 1)
>>>>>>> BP-Network
	{
		if ((rand() % 65536 / 65536.0) < THRESHOLD)
			test_sample[i] = REVERSEC(test_sample[i]);
	}
}

<<<<<<< HEAD
int main(int argc, char* argv[])
{
	int n;
	
	NeuralLayer hnet;
	char buff[17];

	srand((unsigned int)time(NULL));
	initLayer(&hnet, 16, &hnet);
	setsample();

	train(&hnet, sample);

	while (scanf("%d", &n) == 1)
	{
		memcpy(test_sample, font[n % 6], 16);
		/* make_noise(test_sample); */
		test(&hnet, test_sample);
	}
=======
	/* print & save Net */
	writeNet(stdout, &net);
	if ((fl = fopen(save_net, "w")) != 0)
		writeNet(fl, &net);
	else
		fprintf(stderr, "Error in Save Net.");

	releaseNet(&net);
>>>>>>> BP-Network
	

	return 0;
	
}

