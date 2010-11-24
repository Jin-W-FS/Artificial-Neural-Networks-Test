#include <stdio.h>
#include <stdlib.h>

#include "NeuralLayer.h"

/* test */
int main()
{
	NeuralLayer input, hidden, output;
	float inputf[2];
	float result, error;
	int i, ok, n = 0;
	
	float sample[4][2] = {
		0, 0, 0, 1, 1, 0, 1, 1
	};
	float sample_result[4][1] = {
		0, 1, 1, 0
	};

	initLayer(&input, 2, NULL);
	initLayer(&hidden, 2, &input);
	initLayer(&output, 1, &hidden);
	
	do
	{
		printf("\n----%3d ----\n", n++);
		ok = 1;
		for (i = 0; i < 4; i++)
		{
			setLayerValue(&input, sample[i]);
			caculate(&hidden);
			caculate(&output);
			
			printf("%f\t%f ->\t%f\n", input.result[0], input.result[1], output.result[0]);

			error = countFinalError(&output, sample_result[i]) / 2;
			countHiddenError(&output);
			adjustWeights(&output);
			adjustWeights(&hidden);

			ok = ok && (error < 0.05);
		}
	}while(!ok);
	
	printf("\nnow start a test:\n");
	while(scanf("%f %f", &inputf[0], &inputf[1]) == 2)
	{
		setLayerValue(&input, inputf);
		caculate(&hidden);
		caculate(&output);
		
		result = output.result[0];
		printf("%f XOR %f = %f\n", inputf[0], inputf[1], result);
	}

	releaseLayer(&input);
	releaseLayer(&hidden);
	releaseLayer(&output);

	return 0;
}

