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

int nE = 6400;
int nI = 6400;
int nInput = 784;
double time_step = 0.0005;
int simulate_time = 700;
int resting_time = 300;
double b = 0;
int c = 0;
double acc = 0;
int E_total_spike = 0;
double interval = 2;
int main()
{
    char train = 1;
    if(train == 1){
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(0.0f, 1.0f);
        struct timespec begin, end ;
        int total_data = 60000;
        int epoch = 10;
        int train_gap = 10000;
        int train_step = 0;
        //cnpy::NpyArray In_E_arr = cnpy::npy_load("weight/In_E.npy");
        //double* In_E_weight_d = In_E_arr.data<double>();
        vector<double> In_E_weight(nInput *nE);
        vector<double> weight_check(nInput *10);
        double* weight_check_ptr = weight_check.data();
        /*cnpy::NpyArray E_I_arr = cnpy::npy_load("weight/E_I.npy");
        double* E_I_weight_d = E_I_arr.data<double>();
        vector<double> *E_I_weight(nE);*/
        
        //cnpy::NpyArray I_E_arr = cnpy::npy_load("weight/I_E.npy");
        //double* I_E_weight_d = I_E_arr.data<double>();
        vector<double> I_E_weight(nE * nI);
#pragma omp parallel
        {
#pragma omp for
            {
                for (int i = 0; i<nE * nInput; i++) {
                    In_E_weight.data()[i] = (dis(gen) + 0.01)*0.3;
                }
            }
            /*for (int i = 0; i<nE; i++) {
             E_I_weight.at(i) = static_cast<double>(*(E_I_weight_d+i));
             }*/
#pragma omp for
            {
                for (int i = 0; i<nE ; i++) {
                    for (int j = 0; j<nI; ++j) {
                        if (i == j) {
                            I_E_weight.data()[i*nE+j] = 0;
                        }
                        else {
                            I_E_weight.data()[i*nE+j] = 21;
                        }
                    }
                }
            }
        }
        cout<< "1.022" << '\n';
        //vector<double> *input_data(nE, 0);
        vector<double> E_potential(nE);
        //vector<double> *I_potential(nE);
        
        vector<double> E_dCon_E(nE);
        vector<double> E_dCon_I(nE);
        vector<double> E_Con_E(nE);
        vector<double> E_Con_I(nE);
        
        /*vector<double> *I_dCon_E(nE);
        vector<double> *I_Con_E(nE);*/
        
        vector<int> E_spike(nE * simulate_time);
        vector<int> E_spike_total(nE);
        //vector<double> *I_spike(nE * (simulate_time+1));
        
        vector<int> neuron_index(nE,0);
        //vector<double> *prior(nE);
        //vector<double> *latter(nE);
        int performance_count = 0;
        fill(E_potential.begin(),E_potential.end(),-0.065);
        //fill(I_potential.begin(),I_potential.end(),-0.06);
        
        fill(E_dCon_E.begin(),E_dCon_E.end(),1);
        fill(E_dCon_I.begin(),E_dCon_I.end(),1);
        fill(E_Con_E.begin(),E_Con_E.end(),0);
        fill(E_Con_I.begin(),E_Con_I.end(),0);
        
        /*fill(I_dCon_E.begin(),I_dCon_E.end(),4);
        fill(I_Con_E.begin(),I_Con_E.end(),0);*/
        vector<double> after_save(nE, 0);
        vector<int> neuron_index_num(train_gap * nE, 0);
        vector<double> train_data;
        
        vector<int> train_label;
        ReadMNIST(total_data, nInput, train_data, 1);                // 훈련데이터를 불러옴
        ReadMNISTLabel(total_data, train_label, 1);                        // 레이블을 읽어 옴
        /* start simulation */
        SNN stimulation(train_data, E_potential, E_dCon_E, E_dCon_I, E_Con_E, E_Con_I, E_spike, In_E_weight, I_E_weight, E_spike_total, neuron_index_num, neuron_index, nE, nInput, simulate_time, interval, total_data, epoch, train_gap, time_step);
        /*SNN stimulation(train_data, input_data, E_potential, I_potential, E_dCon_E, E_dCon_I, E_Con_E, E_Con_I, I_dCon_E, I_Con_E, E_spike, I_spike, In_E_weight, E_I_weight, I_E_weight, E_spike_total, neuron_index_num, neuron_index, nE, nInput, simulate_time, interval, rate, total_data, epoch);*/
        
        int i = 0;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        while(i<total_data*epoch) {
            int iteration = i % 60000;
            stimulation.normalization(In_E_weight, nInput, nE);
            if (i == 0) {
                weight_save(In_E_weight, E_dCon_E, E_dCon_I, E_Con_E, E_Con_I, E_potential, neuron_index, nE, nInput, i);
            }
            /*double pp = 0;
            for(int j = 0 ; j <nE; ++j) {
                pp = 0;
                for(int k =0; k<nInput;++k) {
                    if (In_E_weight.data()[k*nE+j] > pp) {
                        pp = In_E_weight.data()[k*nE+j];
                    }
                }
                cout <<(int)(pp*255)<< ' ';
            }
            cout <<'\n';*/
            stimulation.initializatoin(E_total_spike, simulate_time, iteration);
            stimulation.set_initial(iteration, interval);
            //stimulation.poisson_spike_generator(simulate_time);
            //clock_gettime(CLOCK_MONOTONIC, &begin);
            stimulation.Stimulation(simulate_time, after_save, E_spike_total, E_total_spike, weight_check_ptr, train_label[iteration]);
            //clock_gettime(CLOCK_MONOTONIC, &end);
            //cout << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << '\n';
            //stimulation.spike_total(E_spike, E_spike_total, nE, simulate_time, E_total_spike);
            if (E_total_spike <5) {
                interval +=1.0;
                //clock_gettime(CLOCK_MONOTONIC, &begin);
                stimulation.resting(resting_time, after_save);
                //clock_gettime(CLOCK_MONOTONIC, &end);
                //cout << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << '\n';
            }
            else {
                //int save = performance_count;
                /*cout<<"spike "<<'\n';
                for (int m  = 0; m<nE; ++m) {
                    if (E_spike_total.at(m)>1) {
                        cout << E_spike_total.at(m) << ' ';
                    }
                }*/
                /*for (int m  = 0; m<nE; ++m) {
                    if(E_spike_total.at(m) != 0) {
                        cout << m << ' ';
                    }
                }
                cout <<'\n';
                cout << E_total_spike <<'\n';*/
                /*for (int m  = 0; m<simulate_time; ++m) {
                    int l = 0;
                    for (int p = 0; p<nE; ++p) {
                        if (E_spike.at(p*simulate_time+m) == 1) {
                            l += 1;
                        }
                    }
                    if (l>1) {
                        cout << " time: " << m << " spike: " << l << '\n';
                    }
                }
                cout <<"iter: "<<i <<"E_total_spike: "<< E_total_spike << '\n';*/
                /*vector<double> spikew(nInput,0);
                for (int m  = 0; m<simulate_time; ++m) {
                    for (int p = 0; p<nInput; ++p) {
                        spikew.data()[p] += stimulation.train_data_temp_ptr[m*nInput+p];
                    }
                }
                for (int p = 0; p<nInput; ++p) {
                    cout << spikew.data()[p] << ' ';
                    if (p % 28 == 27) {
                        cout << '\n';
                    }
                }*/
                c+=E_total_spike;
                stimulation.process_data(train_label[iteration], performance_count, i, total_data, epoch, train_gap);
                if (i % train_gap == train_gap -1) {
                    cout << "---------------result-------------"<<'\n';
                    for (int j = 0; j<nE; ++j) {
                        b += E_dCon_E.at(j);
                    }
                    acc = (double)performance_count / (double) train_gap * 100;
                    cout << "iter : " << i+1 << ' '<< "total : " << c<< " conductance : "<< b/nE <<" accuracy = " << acc << "%"<<'\n';
                    c = 0;
                    b = 0;
                    weight_save(In_E_weight, E_dCon_E, E_dCon_I, E_Con_E, E_Con_I, E_potential, neuron_index, nE, nInput, i);
                    performance_count = 0;
                    stimulation.set_index(train_gap, train_step, i);
#pragma omp parallel
                    {
                        float64x2_t temp2 = vdupq_n_f64(0);
#pragma omp for
                        {
                            for (int j = 0; j < nInput*10; j+=2) {
                                vst1q_f64(&weight_check_ptr[j], temp2);
                            }
                        }
                    }
                    for (int j = 0; j<nE; ++j) {
                        for (int k = 0; k<10; ++k) {
                            if (neuron_index.data()[j] == k) {
                                for (int l = 0; l<nInput; ++l) {
                                    weight_check_ptr[l*10+k] += In_E_weight.data()[l*nE+j];
                                }
                                break;
                            }
                        }
                    }
                    stimulation.normalization(weight_check, nInput, 10);
                    for (int k = 0; k<10; ++k) {
                        cout << "number: " << k << '\n';
                        for (int l = 0; l<nInput; ++l) {
                            cout << floor(weight_check_ptr[l*10+k]*10) << ' ';
                            if (l % 28 == 27) {
                                cout << '\n';
                            }
                        }
                    }
                    for (int j = 0; j<nE; ++j) {
                        cout << neuron_index.at(j);
                        if (j % 80 == 79) {
                            cout << '\n';
                        }
                    }
                    cout << "exclusion: " << stimulation.neuron_exclusion<< '\n';
                    train_step +=1;
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    cout << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << '\n';
                    clock_gettime(CLOCK_MONOTONIC, &begin);
                }
                interval = 2.0;
                i+=1;
                /*if (i % train_gap != train_gap -1) {
                    clock_gettime(CLOCK_MONOTONIC, &end);
                    cout << (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / 1000000000.0 << '\n';
                }
                clock_gettime(CLOCK_MONOTONIC, &begin);*/
            }
        }
        return 0;
        
    }
    else{
        int total_data = 10000;
        vector<double> test_data;
        vector<int> test_label;
        ReadMNIST(total_data, nInput, test_data, 1);                // 훈련데이터를 불러옴
        ReadMNISTLabel(total_data, test_label, 0);
    }
}
