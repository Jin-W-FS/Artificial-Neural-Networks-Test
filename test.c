#include <stdio.h>
#include <stdlib.h>

#include "NeuralLayer.h"

/* test */
int main()
{
	NeuralNet net;
	int n_nodes[] = {
		2, 2, 1
	};
	
	float inputf[2];
	float result, error;
	int i, ok, n = 0;
	
	float sample[4][2] = {
		0, 0, 0, 1, 1, 0, 1, 1
	};
	float sample_result[4][1] = {
		0, 1, 1, 0
	};

	
	initNet(&net, 3, n_nodes);

	do
	{
		printf("\n----%3d ----\n", n++);
		ok = 1;
		for (i = 0; i < 4; i++)
		{
			error = evolveNet(&net, sample[i], sample_result[i]);
			printf("%f %f =>\t%f\n",			\
			       net.input->result[0], net.input->result[1], \
			       net.output->result[0]);
			ok = (ok && (error < 0.05));
		}
	}while(!ok);
	
	printf("\nnow start a test:\n");
	while(scanf("%f %f", &inputf[0], &inputf[1]) == 2)
	{
		caculateNet(&net, inputf);
		result = net.output->result[0];
		printf("%f XOR %f = %d\n", inputf[0], inputf[1], result > 0.5);
	}

	releaseNet(&net);

	return 0;
}

