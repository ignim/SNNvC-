//
//  mnist_to_array.hpp
//  SNN
//
//  Created by 전민기 on 2022/12/13.
//

#ifndef mnist_to_array_hpp
#define mnist_to_array_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

int ReverseInt(int i);
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<float> &arr, char train_or_test);
void ReadMNISTLabel(int NumberOfImages, vector<int> &arr, char train_or_test);
void normalization(vector<float> *In_E, int nInput, int nE);
void spike_total(vector<float> *E_spike, vector<float> *E_spike_total, int nE, int simulate_time, float &E_total_spike);
void weight_save(vector<float> *In_E_weight, int nE, int nInput, int iteration);
//void spike_check(float &interval, float &E_total_spike, vector<float> *E_spike_total, int nE);

#endif /* mnist_to_array_hpp */
