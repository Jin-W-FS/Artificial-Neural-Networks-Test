#include <stdio.h>
#include <stdlib.h>

#include "NeuralLayer.h"

/* test */
int main()
{
	NeuralLayer input, output;
	float inputf[2];
	float result, error;
	int i, ok, n = 0;
	
	float sample[4][2] = {
		0, 0, 0, 1, 1, 0, 1, 1
	};
	float sample_result[4][1] = {
		0, 1, 1, 1
	};

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

