#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_INPUT 2
#define MAX_INPUT_EXT (MAX_INPUT + 1)

/* learning rete */
#define yita 0.5
#define epsino 0.01

float weights[MAX_INPUT_EXT];

void init();

float caculate(float* inputs);
float adjust(float output);
/* improve from difference = destination - result(from caculate) */
void learn(float result, float dest, float* inputs);

int main()
{
	float model_inputs[4][MAX_INPUT] = {
		0, 0, 0, 1, 1, 0, 1, 1,
	};
	float model_answer[4] = {
		0, 0, 0, 1
	};
	
	int i, pass;
	float result;

	float input[MAX_INPUT];

	/* training */
	init();

	do
	{
		pass = 1;
		for (i = 0; i < 4; i++)
		{
			result = caculate(model_inputs[i]);

			printf("\t%f && %f = %f\n", model_inputs[i][0], model_inputs[i][1], result);

			if (fabsf(model_answer[i] - result) > epsino)
			{
				pass = 0;
				learn(result, model_answer[i], model_inputs[i]);
			}
		}
		putchar('\n');
	}while(!pass);

	/* test */
	printf("weights:\n");
	for (i = 0; i < MAX_INPUT_EXT; i++)
	{
		printf("\t%f", weights[i]);
	}
	printf("\nNow start test:\n");
	
	while(scanf("%f %f", &input[0], &input[1]) == 2)
	{
		result = caculate(input);
		printf("\t%f && %f = %f\n", input[0], input[1], result);
	}

	return 0;
}


void init()
{
	int i;
	
	srand((unsigned int)time(NULL));
	for (i = 0; i < MAX_INPUT_EXT; i++)
	{
		weights[i] = rand() % 1000 / 1000.0;
	}
}

float adjust(float output)
{
	return ((output > 0.0) ? 1 : 0);
}

float caculate(float* inputs)
{
	float result = 0;
	int i;
	for (i = 0; i < MAX_INPUT; i++)
	{
		result += inputs[i] * weights[i];
	}
	result += weights[i];
	return adjust(result);
}

void learn(float result, float dest, float* inputs)
{
	int i;
	for (i = 0; i < MAX_INPUT; i++)
	{
		weights[i] += (yita * (dest - result) * inputs[i]);
	}
	weights[i] += (yita * (dest - result) * 1);
}
