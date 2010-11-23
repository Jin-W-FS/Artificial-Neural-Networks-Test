#include <stdio.h>
#include <stdlib.h>

#include "NeuralLayer.h"

int input_sample(float sample[4][2], float result[4][1])
{
	int i;
	for (i = 0; i < 4; i++)
	{
		if (scanf("%f %f %f", &sample[i][0], &sample[i][1], &result[i][0]) != 3)
			return -1;
		else
			printf("%f %f %f\n", sample[i][0], sample[i][1], result[i][0]);
	}
	return 0;
}

/* test */
int main()
{
	NeuralLayer input, output;
	float inputf[2];
	float result, error;
	int i, ok, n = 0;

	float sample[4][2];
	float sample_result[4][1];
	
	printf("input sample with result (matrix 4 * 3):\n");
	if (input_sample(sample, sample_result))
	{
		printf("reading error, exit.\n");
		return -1;
	}
	
	initLayer(&input, 2, NULL);
	initLayer(&output, 1, &input);

	do
	{
		printf("\n----%3d ----\n", n++);
		ok = 1;
		for (i = 0; i < 4; i++)
		{
			setLayerValue(&input, sample[i]);
			caculate(&output);
			
			printf("%f\t%f ->\t%f\n", input.result[0], input.result[1], output.result[0]);

			error = countFinalError(&output, sample_result[i]) / 2;
			if (error > 0.001)
			{
				ok = 0;
			}
			adjustWeights(&output);

		}
	}while(!ok);
	
	printf("\nnow start a test:\n");
	while(scanf("%f %f", &inputf[0], &inputf[1]) == 2)
	{
		setLayerValue(&input, inputf);
		caculate(&output);
		result = output.result[0];
		printf("%f || %f = %f\n", inputf[0], inputf[1], result);
	}

	releaseLayer(&input);
	releaseLayer(&output);

	return 0;
}

