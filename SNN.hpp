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
    int iteration_;
    double interval_;
    int total_data_;
    int epoch_;
    int train_gap_;
    double time_step_;
    int refractory_e = (int)(0.005/time_step_);
    int subtraction = (int)(0.04/time_step_);
    int addition = (int)(0.02/time_step_);
    
    
    double vI_E = -0.1;
    double vE_E = 0;
    
    double vE_I = 0;
    
    double gL = 1;

    double v_rest_E = -0.065;
    //double v_rest_I = -0.06f;
    double v_reset_E = -0.065;
    //double v_reset_I = -0.045f;

    double v_thresh_E = -0.052;
    //double v_thresh_I = -0.04f;

    double tau_E = 0.1;
    double tau_I = 0.01;

    double tau_syn_E = 0.001;
    double tau_syn_I = 0.002;
    
    vector<double> &train_data_;
    
    //vector<double> &input_data_;
    vector<double> &E_potential_;
    //vector<double> &I_potential_;
    
    vector<double> &E_dCon_E_;
    vector<double> &E_dCon_I_;
    vector<double> &E_Con_E_;
    vector<double> &E_Con_I_;
    
    int neuron_exclusion = 0;
    /*vector<double> &I_dCon_E_;
    vector<double> &I_Con_E_;*/
    
    vector<int> &E_spike_;
    //vector<double> &I_spike_;
    //vector<double> &E_spike_previousvector<double> &(nE_);
    //vector<double> &I_spike_previousvector<double> &(nE_ * (simulation_time_+1));
    vector<double> *I_data = new vector<double>(nE_*simulation_time_,0);
    
    vector<double> &In_E_weight_;
    //vector<double> &E_I_weight_;
    vector<double> &I_E_weight_;
    
    vector<int> &E_spike_total_;
    vector<int> *train_data_temp = new vector<int>(nInput_*simulation_time_,0);
    vector<int> *input_data_copy = new vector<int>(nE_*simulation_time_,0);
    vector<int> *index_num = new vector<int>(train_gap_,0);
    vector<int> &neuron_index_num_;
    vector<int> &neuron_index_;
    vector<double> *learn = new vector<double>(nE_,1);
    vector<int> *learn_stable = new vector<int>(nE_,1);
    int* learn_stable_ptr = learn_stable->data();
    vector<int> *verifying_E = new vector<int>(nE_,0);
    vector<double> *E_potential_capacitance = new vector<double>(nE_*simulation_time_, 0);
    int* E_spike_total_ptr = E_spike_total_.data();
    double* E_potential_capacitance_ptr = E_potential_capacitance->data();
    int* train_data_temp_ptr = train_data_temp->data();
    int* E_spike_ptr = E_spike_.data();
    int* input_data_copy_ptr = input_data_copy->data();
    double* train_data_ptr = train_data_.data();
    double* In_E_weight_ptr = In_E_weight_.data();
    double* E_potential_ptr = E_potential_.data();
    double* E_Con_E_ptr = E_Con_E_.data();
    double* E_dCon_E_ptr = E_dCon_E_.data();
    double* I_data_ptr = I_data->data();
    double* E_Con_I_ptr = E_Con_I_.data();
    double* E_dCon_I_ptr = E_dCon_I_.data();
    double* I_E_weight_prt = I_E_weight_.data();
    int* verifying_E_ptr = verifying_E->data();
    int* neuron_index_ptr = neuron_index_.data();
    int* index_num_ptr = index_num->data();
    int* neuron_index_num_ptr = neuron_index_num_.data();
    //vector<double> &rate_devvector<double> &(nE_,1);
    //vector<double> &dev_initialvector<double> &(nE_,1);
    //vector<double> &dev_aftervector<double> &(nE_,1);
    
    
    
public:
    SNN(vector<double> &train_data, vector<double> &E_potential, vector<double> &E_dCon_E, vector<double> &E_dCon_I, vector<double> &E_Con_E, vector<double> &E_Con_I, vector<int> &E_spike, vector<double> &In_E_weight, vector<double> &I_E_weight, vector<int> &E_spike_total, vector<int> &neuron_index_num, vector<int> &neuron_index, int nE, int nInput, int simulation_time, double &interval, int total_data, int epoch, int train_gap, double time_step) : train_data_(train_data), E_potential_(E_potential),  E_dCon_E_(E_dCon_E), E_dCon_I_(E_dCon_I), E_Con_E_(E_Con_E), E_Con_I_(E_Con_I), E_spike_(E_spike), In_E_weight_(In_E_weight), I_E_weight_(I_E_weight), E_spike_total_(E_spike_total), nE_(nE), nI_(nE), nInput_(nInput), simulation_time_(simulation_time), neuron_index_num_(neuron_index_num), neuron_index_(neuron_index), interval_(interval), total_data_(total_data), epoch_(epoch), train_gap_(train_gap), time_step_(time_step) {}
    /*SNN(vector<double> &train_data, vector<double> &input_data, vector<double> &E_potential, vector<double> &I_potential, vector<double> &E_dCon_E, vector<double> &E_dCon_I, vector<double> &E_Con_E, vector<double> &E_Con_I, vector<double> &I_dCon_E, vector<double> &I_Con_E, vector<double> &E_spike, vector<double> &I_spike, vector<double> &In_E_weight, vector<double> &E_I_weight, vector<double> &I_E_weight, vector<double> &E_spike_total, vector<double> &neuron_index_num, vector<double> &neuron_index, int nE, int nInput, int simulation_time, double &interval, vector<double> &rate, int total_data, int epoch) : train_data_(train_data), input_data_(input_data), E_potential_(E_potential), I_potential_(I_potential), E_dCon_E_(E_dCon_E), E_dCon_I_(E_dCon_I), E_Con_E_(E_Con_E), E_Con_I_(E_Con_I), I_dCon_E_(I_dCon_E), I_Con_E_(I_Con_E), E_spike_(E_spike), I_spike_(I_spike), In_E_weight_(In_E_weight), E_I_weight_(E_I_weight), I_E_weight_(I_E_weight), E_spike_total_(E_spike_total), nE_(nE), nI_(nE), nInput_(nInput), simulation_time_(simulation_time), neuron_index_num_(neuron_index_num), neuron_index_(neuron_index), interval_(interval), rate_(rate), total_data_(total_data), epoch_(epoch){}*/
    void set_initial(int iteration, double interval);
    //void setting_for_proceeding(int time);
    //void poisson_spike_generator(int time);
    void initializatoin(int &E_total_spike, int time, int iteration);
    void Stimulation(int time, vector<double> &after_save, vector<int> &E_spike_total, int &E_total_spike, double* weight_check_ptr, int train_label);
    void process_data(int train_label, int &performance_count, int iter, int total_data, int epoch, int train_gap);
    //void STDP(int time);
    void set_index(int train_gap, int train_step, int iteration);
    void resting(int time, vector<double> &after_save);
    void normalization(vector<double> &In_E, int nInput, int nE);
    //void spike_total(vector<double> &E_spike, vector<double> &E_spike_total, int nE, int simulate_time, double &E_total_spike);
};

#endif /* SNN_hpp */

