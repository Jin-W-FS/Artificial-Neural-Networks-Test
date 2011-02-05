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

	"  * "					\
	"  * "					\
	"  * "					\
	"  * ",			/* 1 */

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
	int i;
	for (i = 0; i < 16; i++)
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
	{
		caculate(hnet);
		printf("-- %d --\n", ++n);
		int_to_char(hnet->result, buffer);
		print_font(buffer);
	}while(!hnet->stable);
}

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
	{
		if ((rand() % 65536 / 65536.0) < THRESHOLD)
			test_sample[i] = REVERSEC(test_sample[i]);
	}
}

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
	

	return 0;
	
}

