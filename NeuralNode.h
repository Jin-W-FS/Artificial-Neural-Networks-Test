#ifndef NEURAL_NETWORKS_NODES_JIN
#define NEURAL_NETWORKS_NODES_JIN

#include <math.h>

#define MAX_INPUT 2

/* learning rate */
#define LEARNING_RATE 0.5
/* rate k in the output: 1 / (1 + exp(-k * x)
   slope near 0 is k/4
 */
#define EXP_RATE 100.0f

#define EPSINO 0.001

typedef struct _NeuralNode
{
	float weights[MAX_INPUT + 1];
	float result;
	float error;
}NeuralNode;

void initNeuralNode(NeuralNode* node);

/* node: the single node;
 * inputs: an NeuralNode array with MAX_INPUT elements. */
void caculate(NeuralNode* node, NeuralNode* inputs);
float adjust_out(float output);
float adjust_back(float error);

/* improve from error */

/* count error = destination - node.result for a node. */
void count_final_error(NeuralNode* node, float destination);
/* give each input node an error rate from the final node */
void count_hidden_error(NeuralNode* node, NeuralNode* inputs);
#define LITTLE_ERROR(error) (((error) < EPSINO) && ((error) > -EPSINO))

void learn(NeuralNode* node, NeuralNode* inputs, float destnation);

#endif	/* end NEURAL_NETWORKS_NODES_JIN */
