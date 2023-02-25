//
//  main.cpp
//  SNN
//
//  Created by 전민기 on 2022/11/04.
//

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
#include "mnist_to_array.hpp"
#include "cnpy/cnpy.h"
#include "/opt/homebrew/Cellar/llvm/15.0.6/lib/clang/15.0.6/include/omp.h"
#include "SNN.hpp"


using namespace std;

int nE = 1600;
int nI = 1600;
int nInput = 784;

int simulate_time = 350;
int resting_time = 150;
float a = 0.0f;
float b = 0.0f;
float c = 0.0f;
float acc = 0.0f;
float E_total_spike = 0.0f;
float interval = 2.0f;
int main()
{
    
    cnpy::NpyArray In_E_arr = cnpy::npy_load("weight/In_E.npy");
    double* In_E_weight_d = In_E_arr.data<double>();
    vector<float> *In_E_weight = new vector<float>(nInput *nE );
    cnpy::NpyArray E_I_arr = cnpy::npy_load("weight/E_I.npy");
    double* E_I_weight_d = E_I_arr.data<double>();
    vector<float> *E_I_weight = new vector<float>(nE);
    cnpy::NpyArray I_E_arr = cnpy::npy_load("weight/I_E.npy");
    double* I_E_weight_d = I_E_arr.data<double>();
    vector<float> *I_E_weight = new vector<float>(nE * nI);
    for (int i = 0; i<nE * nInput; i++) {
        In_E_weight->at(i) = static_cast<float>(*(In_E_weight_d+i));
    }
    for (int i = 0; i<nE; i++) {
        E_I_weight->at(i) = static_cast<float>(*(E_I_weight_d+i));
    }
    for (int i = 0; i<nE * nI; i++) {
        I_E_weight->at(i) = static_cast<float>(*(I_E_weight_d+i));
    }
    vector<float> *input_data = new vector<float>(nE*simulate_time, 0);
    vector<float> *E_potential = new vector<float>(nE);
    vector<float> *I_potential = new vector<float>(nE);
    
    vector<float> *E_dCon_E = new vector<float>(nE);
    vector<float> *E_dCon_I = new vector<float>(nE);
    vector<float> *E_Con_E = new vector<float>(nE);
    vector<float> *E_Con_I = new vector<float>(nE);
    
    vector<float> *I_dCon_E = new vector<float>(nE);
    vector<float> *I_Con_E = new vector<float>(nE);
    
    vector<float> *E_spike = new vector<float>(nE * simulate_time);
    vector<float> *E_spike_total = new vector<float>(nE);
    vector<float> *I_spike = new vector<float>(nE * (simulate_time+1));
    
    vector<float> *neuron_index = new vector<float>(nE);
    vector<float> *prior = new vector<float>(nE);
    vector<float> *latter = new vector<float>(nE);
    vector<float> *after_save = new vector<float>(nE, 0);
    vector<float> *rate = new vector<float>(nE,1);
    vector<float> *rate_dev = new vector<float>(nE,1);
    vector<float> *dev_initial = new vector<float>(nE,1);
    vector<float> *dev_after = new vector<float>(nE,1);
    int performance_count = 0;
    fill(E_potential->begin(),E_potential->end(),-0.065);
    fill(I_potential->begin(),I_potential->end(),-0.06);
    
    fill(E_dCon_E->begin(),E_dCon_E->end(),1);
    fill(E_dCon_I->begin(),E_dCon_I->end(),7);
    fill(E_Con_E->begin(),E_Con_E->end(),0);
    fill(E_Con_I->begin(),E_Con_I->end(),0);
    
    fill(I_dCon_E->begin(),I_dCon_E->end(),7);
    fill(I_Con_E->begin(),I_Con_E->end(),0);
    cout<< "weight feedback 2 stdp 6 0.01" << endl;
    char train = 1;
    if(train == 1){
        struct timespec begin, end ;
        int total_data = 60000;
        int epoch = 10;
        int train_gap = 10000;
        int train_step = 0;
        vector<float> *neuron_index_num = new vector<float>(total_data*epoch * nE, 0);
        vector<float> train_data;
        vector<int> train_label;
        ReadMNIST(total_data, nInput, train_data, 1);                // 훈련데이터를 불러옴
        ReadMNISTLabel(total_data, train_label, 1);                        // 레이블을 읽어 옴
        /* start simulation */
        SNN stimulation(train_data, input_data, E_potential, I_potential, E_dCon_E, E_dCon_I, E_Con_E, E_Con_I, I_dCon_E, I_Con_E, E_spike, I_spike, In_E_weight, E_I_weight, I_E_weight, E_spike_total, neuron_index_num, neuron_index, nE, nInput, simulate_time, interval, rate, total_data, epoch);
        
        
        clock_gettime(CLOCK_MONOTONIC, &begin);
        for (int i = 0;i < total_data*epoch ; ++i) {
            int iteration = i % 60000;
            normalization(In_E_weight, nInput, nE);
            if (i == 0) {
#pragma omp parallel
                {
                    float sum = 0;
                    float mean = 0;
                    float vari = 0;
                    float stan_dev = 0;
#pragma omp for
                    {
                        for (int j = 0; j < nE ; ++j) {
                            for (int k = 0; k < nInput ; ++k){
                                sum += In_E_weight->at(k*nE+j);
                            }
                            mean = sum / 784;
                            for (int k = 0; k < nInput ; ++k){
                                vari += pow(In_E_weight->at(k*nE+j)-mean,2);
                            }
                            vari = vari / 784;
                            stan_dev = sqrt(vari);
                            dev_initial->at(j) = stan_dev;
                            sum = 0;
                            mean = 0;
                            vari = 0;
                            stan_dev = 0;
                        }
                    }
                }
            }
            else {
#pragma omp parallel
                {
                    float sum = 0;
                    float mean = 0;
                    float vari = 0;
                    float stan_dev = 0;
#pragma omp for
                    {
                        for (int j = 0; j < nE ; ++j) {
                            if (latter->at(j) - prior->at(j) !=0) {
                                for (int k = 0; k < nInput ; ++k){
                                    sum += In_E_weight->at(k*nE+j);
                                }
                                mean = sum / 784;
                                for (int k = 0; k < nInput ; ++k){
                                    vari += pow(In_E_weight->at(k*nE+j)-mean,2);
                                }
                                vari = vari / 784;
                                stan_dev = sqrt(vari);
                                dev_after->at(j) = stan_dev;
                                rate_dev->at(j) = dev_initial->at(j) / dev_after->at(j);
                                dev_initial->at(j) = stan_dev;
                                sum = 0;
                                mean = 0;
                                vari = 0;
                                stan_dev = 0;
                            }
                        }
                    }
                }
            }
            /*float p = 0;
            for (int j = 0; j<nE; ++j) {
                for (int k = 0; k < nInput ; ++k){
                    p+=In_E_weight->at(k*nE+j);
                }
                cout << p << ' ';
                if (j % 40 == 39) {
                    cout << endl;
                }
                p = 0;
            }*/
            
#pragma omp parallel
            {
                float a_private = 0;
#pragma omp for
                {
                    for (int j = 0; j < nE ; ++j) {
                        a_private = 0;
                        for (int k = 0; k<nInput; ++k) {
                            a_private += In_E_weight->at(k*nE+j);
                        }
                        prior->at(j) = a_private;
                    }
                }
            }
            /*if (i %1000 ==  999) {
                struct tm curr_tm;
                time_t curr_time = time(nullptr);

                localtime_r(&curr_time, &curr_tm);

                int curr_year = curr_tm.tm_year + 1900;
                int curr_month = curr_tm.tm_mon + 1;
                int curr_day = curr_tm.tm_mday;
                int curr_hour = curr_tm.tm_hour;
                int curr_minute = curr_tm.tm_min;
                int curr_second = curr_tm.tm_sec;
                cout << "continue" << ' '<< curr_year << "-"<< curr_month <<"-"<<
                curr_day << " "<< curr_hour<< ":" << curr_minute <<":"<< curr_second << endl;
            }*/
            
            stimulation.initializatoin(E_total_spike, simulate_time, iteration);
            //stimulation.setting_for_proceeding(simulate_time);
            stimulation.set_initial(iteration);
            stimulation.poisson_spike_generator(simulate_time);
            stimulation.Stimulation(simulate_time, after_save);
            spike_total(E_spike, E_spike_total, nE, simulate_time, E_total_spike);

            //spike_check(interval, E_total_spike, E_spike_total, nE);
            c += E_total_spike;
            stimulation.process_data(neuron_index_num, train_label[iteration], performance_count, i, total_data, epoch);
            stimulation.STDP(simulate_time, rate_dev);
#pragma omp parallel
            {
                float a_private = 0;
#pragma omp for
                {
                    for (int j = 0; j < nE ; ++j) {
                        a_private = 0;
                        for (int k = 0; k<nInput; ++k) {
                            a_private += In_E_weight->at(k*nE+j);
                        }
                        latter->at(j) = a_private;
                    }
                }
#pragma omp for
                {
                    for (int j = 0; j < nE ; ++j) {
                        if (latter->at(j) - prior->at(j) !=0) {
                            if (latter->at(j) - prior->at(j)<0) {
                                float m = prior->at(j)/latter->at(j);
                                float div;
                                float mod;
                                mod = modf(m, &div);
                                rate->at(j) = div+mod/500;
                                after_save->at(j) = 0;
                            }
                            else {
                                float m = prior->at(j)/latter->at(j);
                                rate->at(j) = m;
                                after_save->at(j) = (1-m)/(500*(1-(E_dCon_E->at(j)*m+1)/2));
                            //cout << "dfsf: " <<(1-m)/(350*(1-(E_dCon_E->at(j)*m+1)/2)) << endl;
                            }
                        }
                    }
                }
            }
            if (E_total_spike <5) {
                stimulation.initializatoin(E_total_spike, simulate_time, iteration);
                interval +=1;
                stimulation.resting(resting_time);
            }
            else {
                interval = 2.0;
            }
            if (i == 0) {
                weight_save(In_E_weight, nE, nInput, i);
            }
            if (i % train_gap == train_gap -1) {
                for (int j = 0; j<nE; ++j) {
                    b += E_dCon_E->at(j);
                }
                float k = 0;
                for (int m = 0; m<nE; ++m) {
                    k += stimulation.learn->at(m);
                    //cout << rate_dev->at(m) << ' ';
                }
                acc = ((float)performance_count / train_gap) * 100;
                cout << "iter : " << i+1 << ' '<< "total : " << c<< " conductance : "<< b/1600 <<" accuracy = " << acc<< "% learn: " << k/1600 <<endl;
                c = 0;
                b = 0;
                weight_save(In_E_weight, nE, nInput, i);
                performance_count = 0;
                stimulation.set_index(train_gap, train_step);
                for (int j = 0; j<nE; ++j) {
                    cout << neuron_index->at(j) << ' ';
                    if (j % 40 == 39) {
                        cout << endl;
                    }
                }
                train_step +=1;
                clock_gettime(CLOCK_MONOTONIC, &end);
                cout << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << endl;
            }
        }
        return 0;
        
    }
    else{
        int total_data = 10000;
        vector<float> test_data;
        vector<int> test_label;
        ReadMNIST(total_data, nInput, test_data, 1);                // 훈련데이터를 불러옴
        ReadMNISTLabel(total_data, test_label, 0);
    }
}
