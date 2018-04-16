#include "classifier.h"
//# include <vector>
//using namespace std;


classifier::classifier()
{
}


classifier::~classifier()
{
}


vector<float> classifier::get_outputs()
{
    return outputs;
}

vector<float> classifier::get_weights()
{
    return weights;
}

vector<vector<float>> classifier::get_inputs()
{
    return inputs;
}

void classifier::set_weights(vector<float> w)
{
    weights = w;
}
