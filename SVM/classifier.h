
#pragma once
#include<vector>
using namespace std;

class classifier
{
protected:
    vector<float> weights;
    vector<vector<float>> inputs; // each element in input is a vector (a sample)
    vector<float> outputs;
public:
    classifier();
    virtual ~classifier();
    vector<float> get_weights();
    vector<float> get_outputs();
    vector<vector<float>> get_inputs();
    void set_weights(vector<float> w);
    virtual void update() = 0;
};
