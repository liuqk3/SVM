//
//  main.cpp
//  SVM
//
//  Created by Qiankun Liu on 2018/4/13.
//  Copyright © 2018年 Qiankun Liu. All rights reserved.
//


#include <string>
#include "svm.h"
#include "utils.h"
#include <random>

using namespace std;

int main(int argc, const char * argv[]) {
    
    string train_data_path = "./train.txt";
    data_info info_train = get_data_info(train_data_path, true); // true: shuffle the line index
    int batch_size = 50;
    int num_batch = (info_train.num_samples % batch_size == 0 ) ? int(info_train.num_samples / batch_size) : int(info_train.num_samples / batch_size) + 1;
    
    svm s;
    s.set_with_bias(true); // set bias, default true
    s.init_weight(info_train.num_dims, "zeros"); // initialize the weights
    s.set_learning_rate(0.0003); // set learning rate, default is 0.001
    
    // ################################## train ###########################################
    int epoch = 1;
    for (int epoch_idx = 0; epoch_idx < epoch; epoch_idx++)
    {
        data_info info_train = get_data_info(train_data_path, true); // true: shuffle the line index
        one_batch batch_tmp;
        for (int batch_id = 1; batch_id <= num_batch; batch_id++)
        {
            batch_tmp = get_a_batch(batch_id, batch_size, info_train);
            vector<vector<float>> samples = batch_tmp.samples;
            vector<int> labels = batch_tmp.labels;
            
            s.forward(samples);
            s.hinge_loss(labels); // compute the hinge loss, and the gradients of weights are also computed simutaneously
            s.backward();
            s.update();
            
            // print some statics
            cout<<endl<< "weigths: ";
            vector<float> weigts = s.get_weights();
            for (int w_idx = 0; w_idx < weigts.size(); w_idx++)
                cout<< weigts.at(w_idx)<<"  ";
            cout<<endl;
            
            float loss = s.get_loss();
            float accuracy = s.get_accuracy();
            cout << "Train: "<<"Epoch: "<< epoch_idx << ", batch id: " << batch_id << ", loss = " << loss << ", accuracy = " << accuracy << endl;
        }
    }
    
    // ################################## test #################################
    string test_data_path = "./test.txt";
    data_info info_test = get_data_info(test_data_path, false);
    int test_batch_size = info_test.num_samples;
    int test_num_batch = (info_test.num_samples % test_batch_size == 0) ? int(info_test.num_samples / test_batch_size) : int(info_test.num_samples / test_batch_size) + 1;

    // begin to test s
    one_batch batch_tmp;
    for (int batch_id = 1; batch_id <= test_num_batch; batch_id++)
    {
        batch_tmp = get_a_batch(batch_id, test_batch_size, info_test);
        vector<vector<float>> samples = batch_tmp.samples;
        vector<int> labels = batch_tmp.labels;
        
        s.forward(samples);
        s.hinge_loss(labels);
        float loss = s.get_loss();
        float accuracy = s.get_accuracy();
        cout << endl << "Test: "<< "batch size: "<< test_batch_size << ", batch id: " << batch_id << ", loss = " << loss << ", accuracy = " << accuracy << endl;
    }
    
    return 0;
    
}

