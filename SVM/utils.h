//
//  utils.h
//  SVM
//
//  Created by Qiankun Liu on 2018/4/13.
//  Copyright © 2018年 Qiankun Liu. All rights reserved.
//

#ifndef utils_h
#define utils_h


#endif /* utils_h */

#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

// the information of the data
struct data_info
{
    int num_samples = 0;
    int num_dims = 0;
    vector<int> line_idx; // the line index, starting from 1
    string file_path;
};

// the batch
struct one_batch
{
    vector<vector<float>> samples;
    vector<int> labels;
    vector<int> line_idx; // the line index of the samples
};


data_info get_data_info(string file_path, bool disorder = false)
{
    data_info info;
    //char *fpath = file_path.data();
    FILE *fp;
    int n = 0; // number of lines
    int d = 0; // number of dimensions
    fp = fopen(file_path.data(), "r");
    if (fp == NULL)
    {
        cout << "can not open file: " <<file_path <<endl;
        exit(1);
    }
    else
    {
        char c;
        do{
            c = fgetc(fp);
            if (c == ':' && n == 0)
                d++;
            if (c == '\n')
                n++;
        }while(c!=EOF);
    }

    vector<int> li;
    for (int i = 1; i <= n; i++)
    {
        li.push_back(i);
    }
    if (disorder) // shuffle the line_idx
    {
        random_device rd;
        mt19937 g(rd());
        shuffle(li.begin(), li.end(), g);
    }
    
    info.num_samples = n;
    info.num_dims = d;
    info.line_idx = li;
    info.file_path = file_path;
    return info;
}



one_batch get_a_batch(int batch_id, int batch_size, data_info info)
{
    vector<int> label_tmp;
    vector<vector<float>> sample_tmp;
    
    int num_line = info.num_samples;
    int num_dim = info.num_dims;
    vector<int> line_idx = info.line_idx;
    
    vector<vector<float>> batch;
    int begin_line = (batch_id - 1) * batch_size + 1;
    int end_line = batch_id * batch_size > num_line ? num_line: batch_id * batch_size;
    vector<int> read_line_idx; // the line need to be read
    for (vector<int>::iterator it = line_idx.begin() + begin_line - 1; it != line_idx.begin() + end_line; it++)
    {
        int line = *it;
        read_line_idx.push_back(line);
    }
    
    vector<string> batch_str; // used to store the lines
    string str_tmp;
    int cur_line = 0; // the current line
    ifstream fin(info.file_path);
    if (fin)
    {
        while (getline(fin, str_tmp)) // read a line from the file
        {
            cur_line++; // which line ? the index of line starts from 1...
//            if (begin_line > line_id) // not the line we want, then continue to read the next line...
//                continue;
//            else if (begin_line <= line_id && end_line >= line_id) // the line we want, preserve it.
//                batch_str.push_back(str_tmp);
//            else if (line_id > end_line)
//                break;
            vector<int>::iterator it = find(read_line_idx.begin(), read_line_idx.end(), cur_line); // find the line wether is what we want
            if (it != read_line_idx.end()) // if the line we want
                batch_str.push_back(str_tmp);
        }
    }
    else
    {
        cout<< "can not open file: " << info.file_path << endl;
        exit(1);
    }
    fin.close();
    
    // change the string to samples
    if ( (batch_str.size() != batch_size) && (end_line != num_line))
    {
        cout << "Error: can not load the data successffully!"<< endl;
        exit(1);
    }
    else
    {
        for (int sample_idx = 0; sample_idx < batch_str.size(); sample_idx ++)
        {
            vector<float> one_sample;
            
            // 0 1:4.236298e+00 2:2.198210e+01 3:-3.503797e-01 4:9.752163e+01 ,
            // this is a sample, we need to convert it to what we need
            
            // get the label
            str_tmp = batch_str.at(sample_idx);
            string s;
            if (str_tmp[0] == '0')
                label_tmp.push_back(-1);
            else if(str_tmp[0] == '1')
                label_tmp.push_back(1); // push back to the label_tmp
            
            // get the values of different dimensions
            int pre_idx = str_tmp.find("1:");
            int next_idx = 0;
            for (int d_idx = 1; d_idx <= num_dim; d_idx++)
            {
                int length = 0;
                if (d_idx == num_dim) // the last dimension
                {
                    next_idx = str_tmp.length();
                    length = next_idx - pre_idx - 2;
                }
                else // not the last dmension
                {
                    stringstream ss;
                    ss<<(d_idx+1);
                    string interval = ss.str() + ":"; // the next interval
                    next_idx = str_tmp.find(interval); // find the index of next interval
                    length = next_idx - pre_idx - 3; // get the length of the
                }
                string value_str = str_tmp.substr(pre_idx + 2, length); // add 2 to jump the interval, suach as "1:"
                float value = atof(value_str.data());
                one_sample.push_back(value);
                
                pre_idx = next_idx;
            }
            sample_tmp.push_back(one_sample);
            one_sample.clear(); // prepare for the next sample
        }
    }
    one_batch batch_tmp;
    batch_tmp.samples = sample_tmp;
    batch_tmp.labels = label_tmp;
    batch_tmp.line_idx = read_line_idx;
    return batch_tmp;
    
}


