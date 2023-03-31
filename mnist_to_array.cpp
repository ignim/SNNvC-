//
//  mnist_to_array.cpp
//  SNN
//
//  Created by 전민기 on 2022/12/13.
//

#include "mnist_to_array.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>  
#include <string>
#include <iomanip>
#include <time.h>
#include <random>
#include <thread>
#include <arm_neon.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "/opt/homebrew/Cellar/llvm/15.0.6/lib/clang/15.0.6/include/omp.h"


using namespace std;
using namespace cv;

int ReverseInt(int i)
{
   unsigned char ch1, ch2, ch3, ch4;
   ch1 = i & 255;
   ch2 = (i >> 8) & 255;
   ch3 = (i >> 16) & 255;
   ch4 = (i >> 24) & 255;
   return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<double> &arr, char train_or_test)   // MNIST데이터를 읽어온다.
{
   arr.resize(NumberOfImages * DataOfAnImage);
   if (train_or_test == 1) {
       ifstream file("mnist/train-images-idx3-ubyte", ios::binary);
       if (file.is_open())
       {
           int magic_number = 0;
           int number_of_images = 0;
           int n_rows = 0;
           int n_cols = 0;
    
           file.read((char*)&magic_number, sizeof(magic_number));
           magic_number = ReverseInt(magic_number);
           file.read((char*)&number_of_images, sizeof(number_of_images));
           number_of_images = ReverseInt(number_of_images);
           file.read((char*)&n_rows, sizeof(n_rows));
           n_rows = ReverseInt(n_rows);
           file.read((char*)&n_cols, sizeof(n_cols));
           n_cols = ReverseInt(n_cols);
           for (int i = 0; i<NumberOfImages*n_rows*n_cols; ++i)
           {
               unsigned char temp = 0;
               file.read((char*)&temp, sizeof(temp));
               arr[i] = (double)temp;
  
           }
       }
   }
   else{
       ifstream file("mnist/t10k-images-idx3-ubyte", ios::binary);
       if (file.is_open())
       {
           int magic_number = 0;
           int number_of_images = 0;
           int n_rows = 0;
           int n_cols = 0;
           
           file.read((char*)&magic_number, sizeof(magic_number));
           magic_number = ReverseInt(magic_number);
           file.read((char*)&number_of_images, sizeof(number_of_images));
           number_of_images = ReverseInt(number_of_images);
           file.read((char*)&n_rows, sizeof(n_rows));
           n_rows = ReverseInt(n_rows);
           file.read((char*)&n_cols, sizeof(n_cols));
           n_cols = ReverseInt(n_cols);
           
           for (int i = 0; i<NumberOfImages*n_rows*n_cols; ++i)
           {
               unsigned char temp = 0;
               file.read((char*)&temp, sizeof(temp));
               arr[i] = (double)temp;
  
           }
       }
   }
}

void ReadMNISTLabel(int NumberOfImages, vector<int> &arr, char train_or_test){
   if (train_or_test == 1) {
       ifstream file("mnist/train-labels-idx1-ubyte");
       for (int i = 0; i<NumberOfImages; ++i)
       {
           unsigned char temp = 0;
           file.read((char*)&temp, sizeof(temp));
           if (i > 7){
               arr.push_back((int)temp);
           }
       }
   }
   else{
       ifstream file("mnist/t10k-labels-idx1-ubyte");
       for (int i = 0; i<NumberOfImages; ++i)
       {
           unsigned char temp = 0;
           file.read((char*)&temp, sizeof(temp));
           if (i > 7)
               arr.push_back((int)temp);
    
       }
   }
}

void weight_save(vector<double> &In_E_weight, vector<double> &E_dCon_E, vector<double> &E_dCon_I, vector<double> &E_Con_E, vector<double> &E_Con_I, vector<double> &E_potential, vector<int> &neuron_index, int nE, int nInput, int iteration) {
    int sqrt_nE = (int)sqrt(nE);
    int sqrt_nInput = (int)sqrt(nInput);
    int a  = sqrt_nInput*sqrt_nE;
    int b = nE*nInput;
    Mat img(Size(a,a), CV_64FC1);
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i<sqrt_nE; ++i) {
                for (int j = 0; j<sqrt_nE; ++j){
                    for (int k = 0; k < sqrt_nInput; ++k) {
                        for (int l = 0; l < sqrt_nInput; ++l){
                            img.at<double>(i*sqrt_nInput+k,j*sqrt_nInput+l) = 0;
                        }
                    }
                }
            }
        }
    }
    stringstream ss;
    string type = ".jpg";
    string str1 = to_string(iteration+1);
    string location = "image/";
    ss<<location << str1 <<type;
    string filename = ss.str();
    ss.str("");
    
    unsigned long size_In_E_weight = In_E_weight.size();
    stringstream In_E_weight_f;
    In_E_weight_f<<"weight/" << "In_E_weight_" <<str1<<".bin";
    string In_E_weight_filename = In_E_weight_f.str();
    In_E_weight_f.str("");
    ofstream In_E_weight_outfile(In_E_weight_filename, ios::out | ios::binary);
    In_E_weight_outfile.write(reinterpret_cast<const char*>(In_E_weight.data()), size_In_E_weight * sizeof(double));
    In_E_weight_outfile.close();
    
    unsigned long size_E_dCon_E = E_dCon_E.size();
    stringstream E_dCon_E_f;
    E_dCon_E_f<<"weight/" << "E_dCon_E_" <<str1<<".bin";
    string E_dCon_E_filename = E_dCon_E_f.str();
    E_dCon_E_f.str("");
    ofstream E_dCon_E_outfile(E_dCon_E_filename, ios::out | ios::binary);
    E_dCon_E_outfile.write(reinterpret_cast<const char*>(E_dCon_E.data()), size_E_dCon_E * sizeof(double));
    E_dCon_E_outfile.close();
    
    unsigned long size_E_Con_E = E_Con_E.size();
    stringstream E_Con_E_f;
    E_Con_E_f<<"weight/" << "E_Con_E_" <<str1<<".bin";
    string E_Con_E_filename = E_Con_E_f.str();
    E_Con_E_f.str("");
    ofstream E_Con_E_outfile(E_Con_E_filename, ios::out | ios::binary);
    E_Con_E_outfile.write(reinterpret_cast<const char*>(E_Con_E.data()), size_E_Con_E * sizeof(double));
    E_Con_E_outfile.close();
    
    unsigned long size_E_Con_I = E_Con_I.size();
    stringstream E_Con_I_f;
    E_Con_I_f<<"weight/" << "E_Con_I_" <<str1<<".bin";
    string E_Con_I_filename = E_Con_I_f.str();
    E_Con_I_f.str("");
    ofstream E_Con_I_outfile(E_Con_I_filename, ios::out | ios::binary);
    E_Con_I_outfile.write(reinterpret_cast<const char*>(E_Con_I.data()), size_E_Con_I * sizeof(double));
    E_Con_I_outfile.close();
    
    unsigned long size_E_potential = E_potential.size();
    stringstream E_potential_f;
    E_potential_f<<"weight/" << "E_potential_" <<str1<<".bin";
    string E_potential_filename = E_potential_f.str();
    E_potential_f.str("");
    ofstream E_potential_outfile(E_potential_filename, ios::out | ios::binary);
    E_potential_outfile.write(reinterpret_cast<const char*>(E_potential.data()), size_E_potential * sizeof(double));
    E_potential_outfile.close();
    
    unsigned long size_neuron_index = neuron_index.size();
    stringstream neuron_index_f;
    neuron_index_f<<"weight/" << "neuron_index_" <<str1<<".bin";
    string neuron_index_filename = neuron_index_f.str();
    neuron_index_f.str("");
    ofstream neuron_index_outfile(neuron_index_filename, ios::out | ios::binary);
    neuron_index_outfile.write(reinterpret_cast<const char*>(neuron_index.data()), size_neuron_index * sizeof(unsigned char));
    neuron_index_outfile.close();
    
    unsigned long size_E_dCon_I = E_dCon_I.size();
    stringstream E_dCon_I_f;
    E_dCon_I_f<<"weight/" << "E_dCon_I_" <<str1<<".bin";
    string E_dCon_I_filename = E_dCon_I_f.str();
    E_dCon_I_f.str("");
    ofstream E_dCon_I_outfile(E_dCon_I_filename, ios::out | ios::binary);
    E_dCon_I_outfile.write(reinterpret_cast<const char*>(E_dCon_I.data()), size_E_dCon_I * sizeof(double));
    E_dCon_I_outfile.close();
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i<sqrt_nE; ++i) {
                for (int j = 0; j<sqrt_nE; ++j){
                    for (int k = 0; k < sqrt_nInput; ++k) {
                        for (int l = 0; l < sqrt_nInput; ++l){
                            img.at<double>(i*sqrt_nInput+k,j*sqrt_nInput+l) = In_E_weight.data()[(k*sqrt_nInput+l)*nE + i*sqrt_nE+j];
                        }
                    }
                }
            }
        }
    }
    Mat image(Size(a,a), CV_8UC1);
    convertScaleAbs(img, img, 255.0);
    uchar *dst_data = image.data;
    uchar *src_data = img.data;
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i < b; ++i) {
                dst_data[i] = src_data[i];
            }
        }
    }
    imwrite(filename, image);
}


/*void spike_check(double &interval, double &E_total_spike, vector<double> &E_spike_total, int nE){
    vector<double> temp(nE);
    int check = 0;
#pragma omp parallel
    {
        int check_private = 0;
#pragma omp for
        {
            for (int i = 0; i<nE; ++i) {
                if (E_spike_total.at(i) >= 1) {
                    check_private+=1;
                }
            }
        }
#pragma omp critical
        {
            check+=check_private;
        }
#pragma omp barrier
    }
    cout << "check : " << check << ' ';
    if (check > nE/10) {
        E_total_spike = 0;
        interval /= 2;
    }
    else if (E_total_spike < 5) {
        interval+=1;
    }
    else if (E_total_spike >= 5){
        interval=2;
    }
    
}*/
