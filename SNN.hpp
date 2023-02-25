//
//  SNN.hpp
//  SNN
//
//  Created by 전민기 on 2022/12/26.
//

#ifndef SNN_hpp
#define SNN_hpp

#include <stdio.h>
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
#include <unordered_map>
#include "/opt/homebrew/Cellar/llvm/15.0.6/lib/clang/15.0.6/include/omp.h"


using namespace std;

class SNN{
public:
    int simulation_time_;
    int nE_;
    int nI_;
    int nInput_;
    int data_on = 0;
    int iteration;
    float interval_;
    int while_iter;
    int total_data_;
    int epoch_;
    float time_step = 0.001;
    
    float vI_E = -0.1;
    float vE_E = 0;
    
    float vE_I = 0.0;
    
    float gL = 1.0;

    float v_rest_E = -0.065;
    float v_rest_I = -0.06;
    float v_reset_E = -0.065;
    float v_reset_I = -0.045;

    float v_thresh_E = -0.052;
    float v_thresh_I = -0.04;

    float tau_E = 0.1;
    float tau_I = 0.01;

    float tau_syn_E = 0.001;
    float tau_syn_I = 0.002;
    
    vector<float> train_data_;
    
    vector<float> *input_data_;
    vector<float> *E_potential_;
    vector<float> *I_potential_;
    
    vector<float> *E_dCon_E_;
    vector<float> *E_dCon_I_;
    vector<float> *E_Con_E_;
    vector<float> *E_Con_I_;
    
    vector<float> *I_dCon_E_;
    vector<float> *I_Con_E_;
    
    vector<float> *E_spike_;
    vector<float> *I_spike_;
    vector<float> *E_spike_previous = new vector<float>(nE_ * simulation_time_);
    vector<float> *I_spike_previous = new vector<float>(nE_ * (simulation_time_+1));
    
    vector<float> *In_E_weight_;
    vector<float> *E_I_weight_;
    vector<float> *I_E_weight_;
    
    vector<float> *E_spike_total_;
    vector<float> *train_data_temp = new vector<float>(nInput_*simulation_time_);
    vector<float> *index_num = new vector<float>(total_data_*epoch_);
    vector<float> *neuron_index_num_;
    vector<float> *neuron_index_;
    vector<float> *learn = new vector<float>(nE_,1);
    //vector<float> *I_potential_mem = new vector<float>(nE_);
    //vector<float> *E_Con_E_mem = new vector<float>(nE_);
    //vector<float> *E_Con_I_mem = new vector<float>(nE_);
    //vector<float> *I_Con_E_mem = new vector<float>(nE_);
    //vector<float> *E_dCon_E_mem = new vector<float>(nE_);
    vector<float> *rate_;
protected:
    
    
    
    
public:
    SNN(vector<float> &train_data, vector<float> *input_data, vector<float> *E_potential, vector<float> *I_potential, vector<float> *E_dCon_E, vector<float> *E_dCon_I, vector<float> *E_Con_E, vector<float> *E_Con_I, vector<float> *I_dCon_E, vector<float> *I_Con_E, vector<float> *E_spike, vector<float> *I_spike, vector<float> *In_E_weight, vector<float> *E_I_weight, vector<float> *I_E_weight, vector<float> *E_spike_total, vector<float> *neuron_index_num, vector<float> *neuron_index, int nE, int nInput, int simulation_time, float &interval, vector<float> *rate, int total_data, int epoch) : train_data_(train_data), input_data_(input_data), E_potential_(E_potential), I_potential_(I_potential), E_dCon_E_(E_dCon_E), E_dCon_I_(E_dCon_I), E_Con_E_(E_Con_E), E_Con_I_(E_Con_I), I_dCon_E_(I_dCon_E), I_Con_E_(I_Con_E), E_spike_(E_spike), I_spike_(I_spike), In_E_weight_(In_E_weight), E_I_weight_(E_I_weight), I_E_weight_(I_E_weight), E_spike_total_(E_spike_total), nE_(nE), nI_(nE), nInput_(nInput), simulation_time_(simulation_time), neuron_index_num_(neuron_index_num), neuron_index_(neuron_index), interval_(interval), rate_(rate), total_data_(total_data), epoch_(epoch){}
    void set_initial(int iteration);
    //void setting_for_proceeding(int time);
    void poisson_spike_generator(int time);
    void initializatoin(float &E_total_spike, int time, int iteration);
    void Stimulation(int time, vector<float> *after_save);
    void process_data(vector<float> *neuron_index_num, int train_label, int &performance_count, int iter, int total_data, int epoch);
    void STDP(int time, vector<float> *rate_dev);
    void set_index(int train_gap, int train_step);
    void resting(int time);
};

#endif /* SNN_hpp */

