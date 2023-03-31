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
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<double> &arr, char train_or_test);
void ReadMNISTLabel(int NumberOfImages, vector<int> &arr, char train_or_test);
void weight_save(vector<double> &In_E_weight, vector<double> &E_dCon_E, vector<double> &E_dCon_I, vector<double> &E_Con_E, vector<double> &E_Con_I, vector<double> &E_potential, vector<int> &neuron_index, int nE, int nInput, int iteration);
//void spike_check(double &interval, double &E_total_spike, vector<double> &E_spike_total, int nE);

#endif /* mnist_to_array_hpp */
