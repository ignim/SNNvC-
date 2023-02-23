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
using namespace cv;;

int ReverseInt(int i)
{
   unsigned char ch1, ch2, ch3, ch4;
   ch1 = i & 255;
   ch2 = (i >> 8) & 255;
   ch3 = (i >> 16) & 255;
   ch4 = (i >> 24) & 255;
   return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, vector<float> &arr, char train_or_test)   // MNIST데이터를 읽어온다.
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
               arr[i] = (float)temp;
  
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
               arr[i] = (float)temp;
  
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
               arr.push_back((unsigned char)temp);
    
       }
   }
   
}


void normalization(vector<float> *In_E, int nInput, int nE){
    vector<float> temp(nE);
    float32x4_t scalar_vector = vdupq_n_f32(78.0f);
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i<nE; i+=4) {
                vst1q_f32(&temp[i], vdupq_n_f32(0));
            }
        }
#pragma omp for
        {
            for (int j = 0; j < nE; ++j) {
                for (int i = 0; i < nInput; i+=4) {
                    float32x4_t weight;
                    weight[0]=  In_E->at(i*nE + j);
                    weight[1]=  In_E->at((i+1)*nE + j);
                    weight[2]=  In_E->at((i+2)*nE + j);
                    weight[3]=  In_E->at((i+3)*nE + j);
                    temp[j] += vaddvq_f32(weight);
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i < nE; i+=4) {
                float32x4_t temp4 = vld1q_f32(&temp[i]);
                float32x4_t result = vdivq_f32(scalar_vector, temp4);
                vst1q_f32(&temp[i], result);
            }
        }
#pragma omp for
        {
            for (int j = 0; j < nE; ++j)  {
                for (int i = 0; i < nInput; i+=4) {
                    float32x4_t weight;
                    weight[0]=  In_E->at(i*nE + j);
                    weight[1]=  In_E->at((i+1)*nE + j);
                    weight[2]=  In_E->at((i+2)*nE + j);
                    weight[3]=  In_E->at((i+3)*nE + j);
                    float32x4_t result = vmulq_n_f32(weight, temp[j]);
                    In_E->at(i*nE + j) =  result[0];
                    In_E->at((i+1)*nE + j) =  result[1];
                    In_E->at((i+2)*nE + j) =  result[2];
                    In_E->at((i+3)*nE + j) =  result[3];
                }
            }
        }
    }
}

void spike_total(vector<float> *E_spike, vector<float> *E_spike_total, int nE, int simulate_time, float &E_total_spike){
    float E_total_spike_temp = 0.0f;
#pragma omp parallel
    {
        float E_total_spike_temp_private = 0;
#pragma omp for
        {
            for (int i = 0; i<nE; ++i) {
                for (int j = 0; j<simulate_time; j+=2) {
                    float32x2_t neuron;
                    neuron[0] = E_spike->at(i*simulate_time+j);
                    neuron[1] = E_spike->at(i*simulate_time+j+1);
                    float neuron_add = vaddv_f32(neuron);
                    E_spike_total->at(i) += neuron_add;
                }
            }
        }
#pragma omp for
        {
            for (int i = 0; i<nE; i+=4) {
                float32x4_t total_spike = vld1q_f32(&E_spike_total->at(i));
                E_total_spike_temp_private += vaddvq_f32(total_spike);
            }
        }
#pragma omp critical
            {
                E_total_spike_temp += E_total_spike_temp_private;
            }
    }
    E_total_spike = E_total_spike_temp;
}

void weight_save(vector<float> *In_E_weight, int nE, int nInput, int iteration){
    float data[28*40][28*40];
    int sqrt_nE = (int)sqrt(nE);
    int sqrt_nInput = (int)sqrt(nInput);
    stringstream ss;
    string type = ".jpg";
    string str1 = to_string(iteration+1);
    string location = "image/";
    ss<<location << str1 <<type;
    string filename = ss.str();
    ss.str("");
    // size(width, height)
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i<sqrt_nE; ++i) {
                for (int j = 0; j<sqrt_nE; ++j){
                    for (int k = 0; k < sqrt_nInput; ++k) {
                        for (int l = 0; l < sqrt_nInput; ++l){
                            data[i*sqrt_nInput+k][j*sqrt_nInput+l] = In_E_weight->at((k*sqrt_nInput+l)*nE + i*sqrt_nE+j);
                        }
                    }
                }
            }
        }
    }
    Mat image(Size(28*40, 28*40), CV_8UC1);
    Mat img(Size(28*40, 28*40), CV_32FC1, data);
    convertScaleAbs(img, img, 255.0);
    uchar *dst_data = image.data;
    uchar *src_data = img.data;
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i < 28*40*28*40; ++i) {
                dst_data[i] = src_data[i];
            }
        }
    }
    imwrite(filename, image);
}

/*void spike_check(float &interval, float &E_total_spike, vector<float> *E_spike_total, int nE){
    vector<float> temp(nE);
    int check = 0;
#pragma omp parallel
    {
        int check_private = 0;
#pragma omp for
        {
            for (int i = 0; i<nE; ++i) {
                if (E_spike_total->at(i) >= 1) {
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
