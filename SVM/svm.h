#pragma once
#include "classifier.h"
class svm :
public classifier
{
private:
    float learninig_rate = 0.001;
    float loss = 0.0;
    vector<int> predictions; // the prediction of input samples , 0 or 1.
    float accuracy = 0;
    bool with_bias = true; // set bias, default true
    vector<float> gradients; // the gradients of weights
    vector<int> labels;
public:
    svm();
    ~svm();
    
    void set_with_bias(bool wb = true);
    bool get_with_bias();
    
    void set_learning_rate(float lr);
    
    void init_weight(int num_dim, string mode); // d is the dimension of data, i.e. the number of weights, mode can be "random" or "zeros"
    float get_bias();
    
    void forward(vector<vector<float>> samples); // each element in samples is a vector
    vector<int> get_predictions();
    float get_accuracy();

    void hinge_loss(vector<int> labels);
    void backward();
    float get_loss();
    
    void update();
};


