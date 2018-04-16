#include "svm.h"
#include <random>
#include <iostream>

svm::svm() :classifier()
{
}

svm::~svm()
{
}

void svm::set_with_bias(bool wb)
{
    with_bias = wb;
}

bool svm::get_with_bias()
{
    return with_bias;
}

void svm:: set_learning_rate(float lr)
{
    learninig_rate = lr;
}

void svm::init_weight(int num_dim, string mode)
{

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> normal(0, 1); // mu = 0, sd = 1
    for (int i = 0; i < num_dim; i++)
    {
        float a_w = 0.0;
        if (mode == "random") // initialize the weights randomly
            a_w = normal(gen);
        else if(mode == "zeros")
            a_w = 0.0;
        else
        {
			cout << "Error, plaese pay attentnion the mode you choose: " << endl;// mode << endl;
            exit(1);
        }
        weights.push_back(a_w);
        // gradients.push_back(0); // we initialize the weight simultaneously
    }
    if (with_bias) // put a bias into the weights
    {
        float a_w = normal(gen);
        weights.push_back(a_w);
        // gradients.push_back(0); // we initialize the weight simultaneously
    }
}

float svm::get_bias()
{
    if (with_bias)
    {
        return weights.at(weights.size() - 1);
    }
    else
    {
        cout << "No bias!" <<endl;
        exit(1);
        // throw "No bias!";
    }
}

void svm::forward(vector<vector<float>> samples)
{
    // prepare for forward
    inputs.clear();
    outputs.clear();
    loss = 0.0;
    accuracy = 0.0;
    predictions.clear();
    labels.clear();
    gradients.clear();
    
    // forward
    inputs = samples; // preserve the inputed samples to update the weigths
	int batch_size = int(samples.size());
	for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
	{
		vector<float> a_sample = samples.at(sample_idx);
		float one_output = 0; // the ouput of this sample
		if ( (with_bias && (weights.size() - a_sample.size() == 1)) || (!with_bias && (weights.size() == a_sample.size())) )
		{
			for (int d_idx = 0; d_idx < a_sample.size(); d_idx++)
			{
				one_output += a_sample.at(d_idx)*weights.at(d_idx);
			}
			if (with_bias)
			{
				one_output += weights.at(weights.size() - 1); // pluse the bias
			}
		}
		else
		{
            cout << endl<< "Error the dimensions of input sample and weights are not the same when not equiped with a bias or the dimensions of them are the same when equiped with a bias." << endl;
            exit(1);
			// throw "the dimensions of input sample and weights are not the same when not equiped with a bias or the dimensions of them are the same when equiped with a bias.";
		}
		outputs.push_back(one_output);
        predictions.push_back(one_output >= 0 ? 1:-1);
        one_output = 0; // prepare for the next sample
	}

}

vector<int> svm::get_predictions()
{
    return predictions;
}

float svm:: get_accuracy()
{
    int batch_size = predictions.size();
    int num_correct = 0;
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        if (predictions[sample_idx] == labels[sample_idx])
            num_correct++;
    }
    accuracy = float(num_correct) / batch_size;
    return accuracy;
}

void svm:: hinge_loss(vector<int> lab) // output_svm is the output of svm, not the prediction of input samples
{
    labels = lab;// preserve to compute the gradient
    int batch_size = int(lab.size());
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
        if (predictions.at(sample_idx) != lab.at(sample_idx)) // the prediction is wrong
        {
            loss += 1 - lab[sample_idx] * outputs[sample_idx];
        }
    }
     // average over batch_size ( the number of samples )
    loss = float(loss) / batch_size;
}

void svm:: backward()
{
    int batch_size = int(labels.size());
    int num_dim = int(inputs.at(0).size()); // number of dimensions of the input sample
	int num_weights = num_dim + (with_bias ? 1 : 0);
	gradients = vector<float>(num_weights, 0);
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
    {
		
        if (predictions.at(sample_idx) != labels.at(sample_idx)) // the prediction is wrong
        {
            for (int dim_idx = 0; dim_idx < num_dim; dim_idx++)
            {
                gradients[dim_idx] -= labels[sample_idx] * inputs[sample_idx][dim_idx];
            }
            if (with_bias)
            {
                gradients[gradients.size() - 1] -= labels[sample_idx]; // handling bias
            }

        }
    }
    // average over batch_size ( the number of samples )
    for (int dim_idx = 0; dim_idx < gradients.size(); dim_idx++)
    {
        gradients[dim_idx] = float(gradients[dim_idx]) / batch_size;
    }
}


float svm::get_loss()
{
    return loss;
}

void svm::update()
{
    for (int dim_idx = 0; dim_idx < weights.size(); dim_idx++)
    {
        weights[dim_idx] -= learninig_rate * gradients[dim_idx];
    }
}



